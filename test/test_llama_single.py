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
from .schedule_helpers import print_schedule

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLaMA model on single device (no pipeline parallelism)')
    parser.add_argument('--model', choices=['LLAMA_DEBUG', 'LLAMA_1B', 'LLAMA_3B', 'LLAMA_8B'], default='LLAMA_DEBUG',
                        help='Model configuration: LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, or LLAMA_8B (default: LLAMA_DEBUG)')
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


def build_single_device_schedule(n_mbs: int):
    """
    Build a simple schedule for single device (no pipeline parallelism).
    
    For single device, we just do:
    - All forward passes for all microbatches
    - All backward passes for all microbatches
    - Single update step
    
    Args:
        n_mbs: Number of microbatches
    
    Returns:
        A 1xN schedule (1 device, N time steps)
    """
    # Calculate number of time steps: n_mbs forwards + n_mbs backwards + 1 update
    num_steps = n_mbs * 2 + 1
    
    # Single device (stage 0), multiple time steps
    schedule = [[None] * num_steps]
    
    # Forward passes for all microbatches
    for mb_idx in range(n_mbs):
        schedule[0][mb_idx] = Task(
            device_id=0, 
            stage_id=0, 
            mb_idx=mb_idx, 
            is_fwd=True, 
            upd=False
        )
    
    # Backward passes for all microbatches
    for mb_idx in range(n_mbs):
        schedule[0][n_mbs + mb_idx] = Task(
            device_id=0, 
            stage_id=0, 
            mb_idx=mb_idx, 
            is_fwd=False, 
            upd=False
        )
    
    # Single update step at the end
    schedule[0][-1] = Task(
        device_id=0, 
        stage_id=0, 
        mb_idx=0, 
        is_fwd=False, 
        upd=True
    )
    
    return schedule


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
    seq_len = args.seq_len
    warmup = args.warmup
    iters = args.iters

    x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len)).to(device)
    y = torch.zeros((batch_size, llama_config.vocab_size), dtype=torch.long).to(device)

    # print_cuda_memory_stats(device, "after loading data")

    model = Transformer(llama_config)
    model.to(device)

    # print_cuda_memory_stats(device, "after loading model")
    compiled, compilation_metadata = piper_setup(model, [x], backend=piper)
    # print_cuda_memory_stats(device, "after compiling model")
    
    # Update the global piper_metadata with the compilation metadata
    # This is necessary because piper_exec and other functions expect the global metadata to be populated
    piper_metadata.update(compilation_metadata)
    
    actors = compilation_metadata['actors']
    num_actors = len(actors)
    
    print(f"Number of actors (devices): {num_actors}")
    
    # For single device, we expect exactly 1 actor
    if num_actors != 1:
        print(f"WARNING: Expected 1 actor for single-device execution, but got {num_actors} actors.")
        print(f"This may indicate that the model has distributed_stage annotations configured for {num_actors} stages.")
        print(f"Proceeding with execution on actor 0 only.")
    
    # Send input and truth to the first (and only) actor
    ray.get(actors[0].send_input.remote(x))
    ray.get(actors[0].send_truth.remote(y))

    # Build simple single-device schedule
    schedule = build_single_device_schedule(num_mbs)

    print("\nSCHEDULE:")
    print_schedule(schedule)
    print(f"Schedule details: {num_mbs} forward passes, {num_mbs} backward passes, 1 update\n")

    def iter_schedule():
        out = piper_exec(compiled, schedule, [x], y, loss_fn, num_mbs)
        ray.get(out)

    # Set tracing
    ray.get([actor.set_tracing.remote(args.tracing) for actor in actors.values()])

    # Warmup iterations
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        iter_schedule()

    # Clear timing data after warmup
    ray.get([actor.clear_trace_data.remote() for actor in actors.values()])

    # Timed iterations
    print(f"Running {iters} timed iterations...")
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

    print("\n" + "="*60)
    print("THROUGHPUT RESULTS:")
    print("="*60)
    print(
        f"Throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
    )
    print(
        f"Average iteration time: {(end - start)*1000/iters:.2f} ms"
    )
    print(
        f"Total time for {iters} iterations: {(end - start):.2f} seconds"
    )
    print("="*60)

    if args.tracing:
        for actor in actors.values():
            trace_data = ray.get(actor.get_trace_data.remote())
            actor_id = ray.get(actor.id.remote())
            print_mean_timing_data(trace_data, actor_id)
        
        timeline_filename = f"out/{args.model}-single-device.json"
        ray.timeline(timeline_filename)
        print(f"\nRay timeline saved to: {timeline_filename}")

if __name__ == "__main__":
    ray.init(include_dashboard=True, log_to_driver=True, namespace="llama-single")
    torch.manual_seed(0)
    args = parse_args()
    main(args)


