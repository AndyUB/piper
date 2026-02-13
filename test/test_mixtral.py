import ray
import torch
import torch.nn as nn
import argparse
import time
import os
import numpy as np
import itertools

from src.piper_coordinator import PiperProgramCoordinator
from src.piper_compile import piper_setup
from src.piper_exec import piper_exec
from src.piper_utils import piper_metadata

from .models.mixtral import Transformer, ModelArgs
from .schedule_helpers import build_1f1b_schedule, print_schedule
from torch._dynamo.backends.debugging import eager

def main(args):

    world_size = args.dp * args.pp
    mbs = args.mbs
    batch_size = args.batch_size

    match args.model:
        case 'tiny':
            config = ModelArgs.from_name("tiny")
        case 'small':
            config = ModelArgs.from_name("small")
        case 'medium':
            config = ModelArgs.from_name("medium")
        case 'large':
            config = ModelArgs.from_name("Mixtral-8x7B-v0.1")

    x = torch.randint(0, config.vocab_size, (batch_size, config.block_size))
    input_pos = torch.arange(config.block_size)
    y = torch.randn(batch_size, config.block_size, config.vocab_size)

    schedule = build_1f1b_schedule(mbs, args.pp)
    print_schedule(schedule)
    model = piper_setup(
        Transformer, 
        [config], 
        torch.optim.Adam, 
        [x, input_pos], 
        schedule,
        args.naive_gradient_sync,
    )

    # Send data to actors ahead of time
    actors = piper_metadata.actors
    ray.get(actors[0].load_input.remote([x]))
    ray.get(actors[len(actors)-1].load_labels.remote(y))

    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Running {args.warmup} warmup iterations...")
    for _ in range(args.warmup):
        piper_exec(model, schedule, [x, input_pos], y, loss_fn, args.dp)
    
    ray.get([actor.set_tracing.remote(args.tracing) for actor in actors.values()])

    print(f"Running {args.iters} timed iterations...")
    iter_times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        piper_exec(model, schedule, [x, input_pos], y, loss_fn, args.dp)
        end = time.perf_counter()
        iter_times.append(end - start)
    
    dp_rank = int(os.environ['PIPER_DP_RANK'])
    print(
        f"rank {dp_rank} iter time= {np.mean(iter_times):.2f} ± {np.std(iter_times):.2f} s ({len(iter_times)} samples)\n"
        f"rank {dp_rank} throughput= {(args.batch_size * args.mbs * config.block_size)/np.mean(iter_times):.2f} tokens/s ({len(iter_times)} samples)"
    )

    if args.tracing:
        trace_data_dicts = ray.get([actor.get_trace_data.remote() for actor in actors.values()])
        for key in trace_data_dicts[0]:
            all_times = list(itertools.chain.from_iterable(trace_data_dict[key] for trace_data_dict in trace_data_dicts))
            print(f"rank {dp_rank} {key} time= {np.mean(all_times):.2f} ± {np.std(all_times):.2f} ms ({len(all_times)} samples)")

    timeline_filename = f"out/mixtral-pp{args.pp}-dp{args.dp}"
    ray.timeline(timeline_filename)
    print(f"Ray timeline saved to: {timeline_filename}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLaMA model with pipeline parallelism')
    parser.add_argument('--model', choices=['tiny', 'small', 'medium', 'large'], default='tiny',
                        help='Model configuration: tiny, small, medium, or large (default: tiny)')
    parser.add_argument('--dp', type=int, default=2,
                        help='Number of data parallel degrees (default: 2)')
    parser.add_argument('--pp', type=int, default=2,
                        help='Number of pipeline parallel degrees (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--mbs', type=int, default=4,
                        help='Number of microbatches (default: 4)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations (default: 5)')
    parser.add_argument('--iters', type=int, default=10,
                        help='Number of timing iterations (default: 10)')
    parser.add_argument('--tracing', action='store_true', default=False,
                        help='Enable tracing')
    parser.add_argument('--naive_gradient_sync', action='store_true', default=False,
                        help='Enable naive gradient sync')
    return parser.parse_args()

if __name__ == "__main__":
    ray.init(include_dashboard=False, log_to_driver=True, namespace="llama", _temp_dir="/m-coriander/coriander/mfris/tmp/ray")
    args = parse_args()
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=args.pp, dp_degree=args.dp)
    handles = piper_coordinator.run_program.remote(main, args)
    ray.get(handles)
    ray.shutdown()
