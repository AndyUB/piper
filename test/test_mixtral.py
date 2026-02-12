import ray
import torch
import torch.nn as nn
import argparse
import time
import os

from src.piper_coordinator import PiperProgramCoordinator
from src.piper_compile import piper_setup
from src.piper_exec import piper_exec
from src.piper_utils import piper_metadata

from .models.mixtral import Transformer, ModelArgs
from .schedule_helpers import build_1f1b_schedule, print_schedule
from torch._dynamo.backends.debugging import eager

def main(args):

    world_size = args.dp * args.pp

    config = ModelArgs.from_name(args.model)
    dp_rank = int(os.environ['PIPER_DP_RANK'])

    x = torch.randint(0, config.vocab_size, (args.batch_size, config.block_size))
    input_pos = torch.arange(config.block_size)
    y = torch.randn(args.batch_size, config.block_size, config.vocab_size)

    schedule = build_1f1b_schedule(args.mbs, args.pp)
    print_schedule(schedule)
    model = piper_setup(
        Transformer, 
        [config], 
        torch.optim.Adam, 
        [x, input_pos], 
        schedule,
        naive_gradient_sync=args.naive_gradient_sync,
    )

    # Send data to actors ahead of time
    actors = piper_metadata.actors
    ray.get(actors[0].load_input.remote([x]))
    ray.get(actors[len(actors)-1].load_labels.remote(y))

    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(args.warmup):
        piper_exec(model, schedule, [x, input_pos], y, loss_fn)

    start = time.perf_counter()
    for _ in range(args.iters):
        piper_exec(model, schedule, [x, input_pos], y, loss_fn)
    end = time.perf_counter()
    print(f"Iter time: {(end - start)/args.iters:.2f} s")

    ray.timeline(f"out/mixtral.json")

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLaMA model with pipeline parallelism')
    parser.add_argument('--model', choices=['tiny', 'small', 'medium', 'Mixtral-8x7B-v0.1'], default='tiny',
                        help='Model configuration: tiny, small, medium, or Mixtral-8x7B-v0.1 (default: tiny)')
    parser.add_argument('--dp', type=int, default=2,
                        help='Number of data parallel degrees (default: 2)')
    parser.add_argument('--pp', type=int, default=2,
                        help='Number of pipeline parallel degrees (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--mbs', type=int, default=4,
                        help='Number of microbatches (default: 4)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations (default: 3)')
    parser.add_argument('--iters', type=int, default=5,
                        help='Number of timing iterations (default: 5)')
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
