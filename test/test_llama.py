import ray
import torch
import time
import argparse
from torch import nn, optim
from torch.profiler import profile, record_function, ProfilerActivity

from src.piper_exec import Task, piper_exec
from src.piper_compile import piper_setup
from src.piper import piper
from src.piper_utils import piper_metadata

from .models.llama import Transformer, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B
from .schedule_helpers import build_1f1b_schedule, build_gpipe_schedule, print_schedule

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLaMA model with pipeline parallelism')
    parser.add_argument('--model', choices=['LLAMA_DEBUG', 'LLAMA_1B', 'LLAMA_3B', 'LLAMA_8B'], default='LLAMA_DEBUG',
                        help='Model configuration: LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, or LLAMA_8B (default: LLAMA_DEBUG)')
    parser.add_argument('--schedule', choices=['gpipe', '1f1b', 'interleaved-1f1b'], default='1f1b',
                        help='Schedule type: gpipe, 1f1b, or interleaved-1f1b (default: 1f1b)')
    parser.add_argument('--devices', type=int, choices=[2, 4], default=2,
                        help='Number of devices/stages (default: 2)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--num_mbs', type=int, default=4,
                        help='Number of microbatches (default: 4)')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Sequence length (default: 512)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations (default: 5)')
    parser.add_argument('--iters', type=int, default=20,
                        help='Number of timing iterations (default: 20)')
    parser.add_argument('--tracing', action='store_true', default=False,
                        help='Enable tracing')
    return parser.parse_args()


def print_cuda_memory_stats(device: str, message: str = "") -> None:
    """Prints CUDA memory usage statistics for the specified device, including available memory.

    Args:
        device (str): The CUDA device identifier (e.g., 'cuda', 'cuda:0').
        message (str): Optional message to describe what is being logged (e.g., "loaded model", "loaded batch").
    """
    torch.cuda.synchronize()
    device_obj = torch.device(device)
    device_idx = device_obj.index if device_obj.index is not None else 0
    BYTES_IN_GB: int = 1024 ** 3

    used_gb: float = torch.cuda.memory_allocated(device_idx) / BYTES_IN_GB
    reserved_gb: float = torch.cuda.memory_reserved(device_idx) / BYTES_IN_GB
    peak_gb: float = torch.cuda.max_memory_allocated(device_idx) / BYTES_IN_GB

    # Get total memory using torch.cuda.get_device_properties
    total_gb: float = torch.cuda.get_device_properties(device_idx).total_memory / BYTES_IN_GB
    available_gb: float = total_gb - reserved_gb

    message_prefix = f":{message}" if message else ""
    print(
        f"[CUDA:{device_idx}] Memory stats{message_prefix} | "
        f"used={used_gb:.1f} GiB, reserved={reserved_gb:.1f} GiB, "
        f"peak={peak_gb:.1f} GiB, available={available_gb:.1f} GiB, total={total_gb:.1f} GiB"
    )


def main(args):
    
    # Set model configuration based on argument
    if args.model == 'LLAMA_DEBUG':
        llama_config = LLAMA_DEBUG
    elif args.model == 'LLAMA_1B':
        llama_config = LLAMA_1B
    elif args.model == 'LLAMA_3B':
        llama_config = LLAMA_3B
    elif args.model == 'LLAMA_8B':
        llama_config = LLAMA_8B
        
    loss_fn = torch.nn.CrossEntropyLoss()
    device = 'cuda'
    
    batch_size = args.batch_size
    num_mbs = args.num_mbs
    num_devices = args.devices
    seq_len = args.seq_len
    warmup = args.warmup
    iters = args.iters

    x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len)).to(device)
    y = torch.zeros((batch_size, llama_config.vocab_size), dtype=torch.long).to(device)

    # print_cuda_memory_stats(device, "after loading data")

    model = Transformer(llama_config)
    model.to(device)

    # print_cuda_memory_stats(device, "after loading model")
    compiled = piper_setup(model, [x], backend=piper)
    # print_cuda_memory_stats(device, "after compiling model")
    
    actors = piper_metadata['actors']
    num_actors = len(actors)
    assert num_actors == num_devices
    ray.get(actors[0].send_input.remote(x))
    ray.get(actors[num_actors-1].send_truth.remote(y))

    schedule = None
    if args.schedule == "interleaved-1f1b":
        from .interleaved_grid_schedules import pp2_interleaved_1f1b_grid_schedule, pp4_interleaved_1f1b_grid_schedule
        schedule = pp2_interleaved_1f1b_grid_schedule if num_devices == 2 else pp4_interleaved_1f1b_grid_schedule
    elif args.schedule == "1f1b":
        schedule = build_1f1b_schedule(num_mbs, num_devices)
    elif args.schedule == "gpipe":
        schedule = build_gpipe_schedule(num_mbs, num_devices)

    print("SCHEDULE:")
    print_schedule(schedule)

    def iter_schedule():
        out = piper_exec(compiled, schedule, [x], y, loss_fn, num_mbs)
        ray.get(out)
        # ray.wait(out, fetch_local=False)

    # set tracing
    ray.get([actor.set_tracing.remote(args.tracing) for actor in actors.values()])

    if args.tracing:
        iter_schedule()

        if args.devices == 2 and args.schedule == "1f1b":
            actor1 = actors[0]
            actor2 = actors[1]
            
            # Collect timing data across multiple iterations
            timing_results = {
                'fwd_0_1': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_1_0': {'times': [], 'sizes': [], 'lengths': []}
            }
            
            for iteration in range(iters):
                # Forward pass timings
                start_time = time.perf_counter()
                ray.get(actor2.time_object_retrieval.remote(actor1.get_object.remote(True, 0)))
                end_time = time.perf_counter()
                fwd_0_1_time = (end_time - start_time) * 1000
                timing_results['fwd_0_1']['times'].append(fwd_0_1_time)
                
                start_time = time.perf_counter()
                ray.get(actor1.time_object_retrieval.remote(actor2.get_object.remote(False, 1)))
                end_time = time.perf_counter()
                bwd_1_0_time = (end_time - start_time) * 1000
                timing_results['bwd_1_0']['times'].append(bwd_1_0_time)

        if args.devices == 2 and args.schedule == "interleaved-1f1b":
            actor1 = actors[0]
            actor2 = actors[1]
            
            # Collect timing data across multiple iterations
            timing_results = {
                'fwd_0_1': {'times': [], 'sizes': [], 'lengths': []},
                'fwd_1_2': {'times': [], 'sizes': [], 'lengths': []},
                'fwd_2_3': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_3_2': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_2_1': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_1_0': {'times': [], 'sizes': [], 'lengths': []}
            }
            
            for iteration in range(iters):
                # Forward pass timings
                start_time = time.perf_counter()
                ray.get(actor2.time_object_retrieval.remote(actor1.get_object.remote(True, 0)))
                end_time = time.perf_counter()
                fwd_0_1_time = (end_time - start_time) * 1000
                timing_results['fwd_0_1']['times'].append(fwd_0_1_time)
                
                start_time = time.perf_counter()
                ray.get(actor1.time_object_retrieval.remote(actor2.get_object.remote(True, 1)))
                end_time = time.perf_counter()
                fwd_1_2_time = (end_time - start_time) * 1000
                timing_results['fwd_1_2']['times'].append(fwd_1_2_time)
                
                start_time = time.perf_counter()
                ray.get(actor2.time_object_retrieval.remote(actor1.get_object.remote(True, 2)))
                end_time = time.perf_counter()
                fwd_2_3_time = (end_time - start_time) * 1000
                timing_results['fwd_2_3']['times'].append(fwd_2_3_time)
                
                # Backward pass timings
                start_time = time.perf_counter()
                ray.get(actor1.time_object_retrieval.remote(actor2.get_object.remote(False, 3)))
                end_time = time.perf_counter()
                bwd_3_2_time = (end_time - start_time) * 1000
                timing_results['bwd_3_2']['times'].append(bwd_3_2_time)
                
                start_time = time.perf_counter()
                ray.get(actor2.time_object_retrieval.remote(actor1.get_object.remote(False, 2)))
                end_time = time.perf_counter()
                bwd_2_1_time = (end_time - start_time) * 1000
                timing_results['bwd_2_1']['times'].append(bwd_2_1_time)
                
                start_time = time.perf_counter()
                ray.get(actor1.time_object_retrieval.remote(actor2.get_object.remote(False, 1)))
                end_time = time.perf_counter()
                bwd_1_0_time = (end_time - start_time) * 1000
                timing_results['bwd_1_0']['times'].append(bwd_1_0_time)

        if args.devices == 4 and args.schedule == "interleaved-1f1b":
            actor1 = actors[0]
            actor2 = actors[1]
            actor3 = actors[2]
            actor4 = actors[3]
            
            # Collect timing data across multiple iterations
            timing_results = {
                'fwd_0_1': {'times': [], 'sizes': [], 'lengths': []},
                'fwd_1_2': {'times': [], 'sizes': [], 'lengths': []},
                'fwd_2_3': {'times': [], 'sizes': [], 'lengths': []},
                'fwd_3_4': {'times': [], 'sizes': [], 'lengths': []},
                'fwd_4_5': {'times': [], 'sizes': [], 'lengths': []},
                'fwd_5_6': {'times': [], 'sizes': [], 'lengths': []},
                'fwd_6_7': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_7_6': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_6_5': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_5_4': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_4_3': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_3_2': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_2_1': {'times': [], 'sizes': [], 'lengths': []},
                'bwd_1_0': {'times': [], 'sizes': [], 'lengths': []}
            }
            
            # microbenchmark p2p communication time
            for iteration in range(iters):
                start_time = time.perf_counter()
                ray.get(actor2.time_object_retrieval.remote(actor1.get_object.remote(True, 0)))
                end_time = time.perf_counter()
                fwd_0_1_time = (end_time - start_time) * 1000
                timing_results['fwd_0_1']['times'].append(fwd_0_1_time)
                
                start_time = time.perf_counter()
                ray.get(actor3.time_object_retrieval.remote(actor2.get_object.remote(True, 1)))
                end_time = time.perf_counter()
                fwd_1_2_time = (end_time - start_time) * 1000
                timing_results['fwd_1_2']['times'].append(fwd_1_2_time)
                
                start_time = time.perf_counter()
                ray.get(actor4.time_object_retrieval.remote(actor3.get_object.remote(True, 2)))
                end_time = time.perf_counter()
                fwd_2_3_time = (end_time - start_time) * 1000
                timing_results['fwd_2_3']['times'].append(fwd_2_3_time)
                
                start_time = time.perf_counter()
                ray.get(actor1.time_object_retrieval.remote(actor4.get_object.remote(True, 3)))
                end_time = time.perf_counter()
                fwd_3_4_time = (end_time - start_time) * 1000
                timing_results['fwd_3_4']['times'].append(fwd_3_4_time)

                start_time = time.perf_counter()
                ray.get(actor2.time_object_retrieval.remote(actor1.get_object.remote(True, 4)))
                end_time = time.perf_counter()
                fwd_4_5_time = (end_time - start_time) * 1000
                timing_results['fwd_4_5']['times'].append(fwd_4_5_time)

                start_time = time.perf_counter()
                ray.get(actor3.time_object_retrieval.remote(actor2.get_object.remote(True, 5)))
                end_time = time.perf_counter()
                fwd_5_6_time = (end_time - start_time) * 1000
                timing_results['fwd_5_6']['times'].append(fwd_5_6_time)

                start_time = time.perf_counter()
                ray.get(actor4.time_object_retrieval.remote(actor3.get_object.remote(True, 6)))
                end_time = time.perf_counter()
                fwd_6_7_time = (end_time - start_time) * 1000
                timing_results['fwd_6_7']['times'].append(fwd_6_7_time)

                # Backward pass timings
                start_time = time.perf_counter()
                ray.get(actor3.time_object_retrieval.remote(actor4.get_object.remote(False, 7)))
                end_time = time.perf_counter()
                bwd_7_6_time = (end_time - start_time) * 1000
                timing_results['bwd_7_6']['times'].append(bwd_7_6_time) 

                start_time = time.perf_counter()
                ray.get(actor2.time_object_retrieval.remote(actor3.get_object.remote(False, 6)))
                end_time = time.perf_counter()
                bwd_6_5_time = (end_time - start_time) * 1000
                timing_results['bwd_6_5']['times'].append(bwd_6_5_time)

                start_time = time.perf_counter()
                ray.get(actor1.time_object_retrieval.remote(actor2.get_object.remote(False, 5)))
                end_time = time.perf_counter()
                bwd_5_4_time = (end_time - start_time) * 1000
                timing_results['bwd_5_4']['times'].append(bwd_5_4_time)

                start_time = time.perf_counter()
                ray.get(actor4.time_object_retrieval.remote(actor1.get_object.remote(False, 4)))
                end_time = time.perf_counter()
                bwd_4_3_time = (end_time - start_time) * 1000
                timing_results['bwd_4_3']['times'].append(bwd_4_3_time)

                start_time = time.perf_counter()
                ray.get(actor3.time_object_retrieval.remote(actor4.get_object.remote(False, 3)))
                end_time = time.perf_counter()
                bwd_3_2_time = (end_time - start_time) * 1000
                timing_results['bwd_3_2']['times'].append(bwd_3_2_time)
                
                start_time = time.perf_counter()
                ray.get(actor2.time_object_retrieval.remote(actor3.get_object.remote(False, 2)))
                end_time = time.perf_counter()
                bwd_2_1_time = (end_time - start_time) * 1000
                timing_results['bwd_2_1']['times'].append(bwd_2_1_time)
                
                start_time = time.perf_counter()
                ray.get(actor1.time_object_retrieval.remote(actor2.get_object.remote(False, 1)))
                end_time = time.perf_counter()
                bwd_1_0_time = (end_time - start_time) * 1000
                timing_results['bwd_1_0']['times'].append(bwd_1_0_time)

        # Calculate and report average statistics
        import numpy as np
        
        print(f"\n=== Average P2P Transfer Statistics ===")
        for transfer_name, data in timing_results.items():
            avg_time = np.mean(data['times'])
            std_time = np.std(data['times'])
            
            print(f"{transfer_name}: {avg_time:.2f} ± {std_time:.2f} ms")
        print()

    # warmup
    for _ in range(warmup):
        iter_schedule()

    # clear timing data
    ray.get([actor.clear_trace_data.remote() for actor in actors.values()])

    # time
    start = time.perf_counter()
    for _ in range(iters):
        iter_schedule()
    end = time.perf_counter()

    def print_mean_timing_data(trace_data: dict, actor_id: int) -> None:
        """Prints mean timing and memory statistics from trace_data for a given actor.

        Args:
            trace_data (dict): Trace data dictionary from the actor containing timing and memory metrics.
            actor_id (int): The ID of the actor.
        """
        import numpy as np

        print(f"\nTiming and Memory statistics for Actor {actor_id}:")
        
        for stage_id, stage_data in trace_data.items():
            if stage_id == 'update':
                # Handle update data (global optimizer step)
                print(f"  Update:")
                for metric, values in stage_data.items():
                    if not values:
                        mean_val = float('nan')
                    else:
                        mean_val = float(np.mean(values))
                    
                    if 'memory' in metric:
                        if 'delta' in metric:
                            print(f"    {metric}: {mean_val:.3f} GB")
                        else:
                            print(f"    {metric}: {mean_val:.3f} GB")
                    else:
                        print(f"    {metric}: {mean_val:.3f} ms")
            else:
                # Handle stage data (forward/backward passes)
                print(f"  Stage {stage_id}:")
                for phase, phase_data in stage_data.items():
                    print(f"    {phase.capitalize()}:")
                    for metric, values in phase_data.items():
                        if not values:
                            mean_val = float('nan')
                        else:
                            mean_val = float(np.mean(values))
                        
                        if 'memory' in metric:
                            if 'delta' in metric:
                                print(f"      {metric}: {mean_val:.3f} GB")
                            else:
                                print(f"      {metric}: {mean_val:.3f} GB")
                        else:
                            print(f"      {metric}: {mean_val:.3f} ms")

    print("THROUGHPUT:")
    print(
        f"{args.schedule} throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
    )
    print(
        f"time: {(end - start)*1000000/iters:.0f} us"
    )

    if args.tracing:
        for actor in actors.values():
            trace_data = ray.get(actor.get_trace_data.remote())
            actor_id = ray.get(actor.id.remote())
            print_mean_timing_data(trace_data, actor_id)
        ray.timeline(f"out/{args.model}-pp{num_devices}-{args.schedule}.json")

if __name__ == "__main__":
    ray.init(include_dashboard=True, log_to_driver=True, namespace="llama")
    torch.manual_seed(0)
    args = parse_args()
    main(args)