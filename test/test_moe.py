import ray
import torch
import torch.nn as nn
import argparse
import time

from src.piper_coordinator import PiperProgramCoordinator
from src.piper_compile import piper_setup
from src.piper_exec import piper_exec
from src.piper import distributed_stage, piper
from src.piper_actor import get_actor
from src.piper_expert import PiperExpert

from .schedule_helpers import no_pp, print_schedule, build_1f1b_schedule

from .models.moe import MixtureOfExperts, FFNExpert

class MoETransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, experts_per_layer, k, world_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.moe = MixtureOfExperts(hidden_dim, hidden_dim, experts_per_layer, FFNExpert, k=k)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        distributed_stage(0)
        x = self.embedding(x)
        x = self.moe(x)
        return self.output(x)
    

def parse_args():
    parser = argparse.ArgumentParser(description='Run MoE model with pipeline and data parallelism')
    parser.add_argument('--pp_degree', type=int, default=1,
                        help='Pipeline parallel degree (default: 1)')
    parser.add_argument('--dp_degree', type=int, default=1,
                        help='Data parallel degree (default: 1)')
    parser.add_argument('--num_mbs', type=int, default=1,
                        help='Number of microbatches (default: 1)')
    parser.add_argument('--num_stages', type=int, default=1,
                        help='Number of stages (default: 1)')
    return parser.parse_args()


def main(args):

    # Create MoE model
    world_size = args.dp_degree * args.pp_degree
    vocab_size = 2
    batch_size = 2
    seq_len = 2

    x = torch.randint(0, vocab_size, (batch_size,))
    y = torch.randn(batch_size, vocab_size)

    num_stages = args.num_stages
    num_devices = world_size
    num_mbs = args.num_mbs

    model = piper_setup(MoETransformer, (vocab_size, 2, 2, 1, world_size), torch.optim.Adam, [x], num_stages=num_stages, num_devices=num_devices)

    if args.num_stages == 1:
        schedule = no_pp
    else:
        schedule = build_1f1b_schedule(num_mbs, num_stages)
    loss_fn = torch.nn.CrossEntropyLoss()

    print_schedule(schedule)

    # losses = piper_exec(model, schedule, [x], y, loss_fn, num_mbs=num_mbs, num_stages=num_stages)
    
    ray.timeline(f"out/moe.json")

if __name__ == "__main__":
    ray.init(include_dashboard=False, log_to_driver=True, namespace="llama")
    args = parse_args()
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=args.pp_degree, dp_degree=args.dp_degree, world_size=args.dp_degree * args.pp_degree)
    handles = piper_coordinator.run_program.remote(main, args)
    ray.get(handles)
    time.sleep(3)
    ray.shutdown()
