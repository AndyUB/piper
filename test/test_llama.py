import ray
import torch
import time
import argparse
import os
from torch import nn, optim
from torch.profiler import profile, record_function, ProfilerActivity

from src.piper_exec import Task, piper_exec
from src.piper_compile import piper_setup
from src.piper import piper
from src.piper_utils import piper_metadata
from src.piper_coordinator import PiperProgramCoordinator

from .models.llama import Transformer, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B
from .schedule_helpers import (
    build_1f1b_schedule, 
    build_gpipe_schedule, 
    print_schedule,
    pp2_interleaved_1f1b_grid_schedule, 
    pp4_interleaved_1f1b_grid_schedule,
    no_pp_schedule
)


def main(args):
    
    # Set model configuration based on argument
    match args.model:
        case 'LLAMA_DEBUG':
            llama_config = LLAMA_DEBUG
        case 'LLAMA_1B':
            llama_config = LLAMA_1B
        case 'LLAMA_3B':
            llama_config = LLAMA_3B
        case 'LLAMA_8B':
            llama_config = LLAMA_8B
    print(args) 

    loss_fn = torch.nn.CrossEntropyLoss()
    
    batch_size = args.batch_size
    num_mbs = args.num_mbs
    seq_len = args.seq_len
    warmup = args.warmup
    iters = args.iters
    
    x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len))
    y = torch.randn((batch_size, seq_len, llama_config.vocab_size))

    # Generate different input data for each data parallel rank so that model weights get updated differently
    if args.dp_degree > 1:
        dp_rank = int(os.environ['PIPER_DP_RANK'])
        torch.manual_seed(dp_rank)
        x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len))
        torch.manual_seed(0)

    schedule = None
    match args.schedule:
        case "no-pp":
            schedule = no_pp_schedule
        case "interleaved-1f1b":
            schedule = pp2_interleaved_1f1b_grid_schedule if args.pp_degree == 2 else pp4_interleaved_1f1b_grid_schedule
        case "1f1b":
            schedule = build_1f1b_schedule(num_mbs, args.pp_degree)
        case "gpipe":
            schedule = build_gpipe_schedule(num_mbs, args.pp_degree)
    print("Schedule:")
    print_schedule(schedule)

    compiled = piper_setup(
        Transformer, 
        (llama_config, seq_len), 
        torch.optim.Adam, 
        [x], 
        schedule
    )
    

    # Send data to actors ahead of time
    actors = piper_metadata.actors
    ray.get(actors[0].load_input.remote([x]))
    ray.get(actors[len(actors)-1].load_labels.remote(y))

    # Definte one iteration of the schedule
    def iter_schedule():
        losses = piper_exec(compiled, schedule, [x], y, loss_fn)

    # Warmup
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        iter_schedule()

    # Clear tracing data
    ray.get([actor.set_tracing.remote(args.tracing) for actor in actors.values()])

    # Time training steps
    start = time.perf_counter()
    print(f"Running {iters} timed iterations...")
    for _ in range(iters):
        iter_schedule()
    end = time.perf_counter()
    
    print(f"Iteration time: {(end - start)*1e3/iters:.0f} ms")
    print(
        f"{args.schedule} throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
    )

    if args.tracing:
        # Get overall peak memory
        peak_memory = ray.get([actor.get_peak_memory.remote() for actor in actors.values()])
        print("Peak memory:")
        for actor_id, peak_memory in enumerate(peak_memory):
            print(f"\tActor {actor_id}: {peak_memory:.1f} GB")

        # Get tracing data from actors
        for actor in actors.values():
            trace_data = ray.get(actor.get_trace_data.remote())
            actor_id = ray.get(actor.id.remote())
            print_mean_timing_data(trace_data, actor_id)
    
    if args.naive_gradient_sync:
        suffix = "-naive-sync"
    else:
        suffix = ""
    timeline_filename = f"out/{args.model}-pp{args.pp_degree}-dp{args.dp_degree}-{args.schedule}{suffix}.json"
    ray.timeline(timeline_filename)
    print(f"\nRay timeline saved to: {timeline_filename}")


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

                        
def parse_args():
    parser = argparse.ArgumentParser(description='Run LLaMA model with pipeline parallelism')
    parser.add_argument('--model', choices=['LLAMA_DEBUG', 'LLAMA_1B', 'LLAMA_3B', 'LLAMA_8B'], default='LLAMA_DEBUG',
                        help='Model configuration: LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, or LLAMA_8B (default: LLAMA_DEBUG)')
    parser.add_argument('--schedule', choices=['gpipe', '1f1b', 'interleaved-1f1b', 'no-pp'], default='1f1b',
                        help='Schedule type: gpipe, 1f1b, or interleaved-1f1b (default: 1f1b)')
    parser.add_argument('--dp_degree', type=int, default=1,
                        help='Number of data parallel degrees (default: 1)')
    parser.add_argument('--pp_degree', type=int, default=2,
                        help='Number of pipeline parallel degrees (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--num_mbs', type=int, default=4,
                        help='Number of microbatches (default: 4)')
    parser.add_argument('--seq_len', type=int, default=256,
                        help='Sequence length (default: 256)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations (default: 5)')
    parser.add_argument('--iters', type=int, default=20,
                        help='Number of timing iterations (default: 20)')
    parser.add_argument('--tracing', action='store_true', default=False,
                        help='Enable tracing')
    parser.add_argument('--naive_gradient_sync', action='store_true', default=False,
                        help='Enable naive gradient sync')
    return parser.parse_args()

if __name__ == "__main__":
    ray.init(include_dashboard=False, log_to_driver=True, namespace="llama")
    args = parse_args()
    piper_coordinator = PiperProgramCoordinator.remote(dp_degree=args.dp_degree, pp_degree=args.pp_degree)
    ray.get(piper_coordinator.run_program.remote(main, args))
    time.sleep(3)
    ray.shutdown()
