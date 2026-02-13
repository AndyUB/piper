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

def main(dp_degree, pp_degree):

    world_size = dp_degree * pp_degree
    num_mbs = 1
    batch_size = 8

    config = ModelArgs.from_name("tiny")
    dp_rank = int(os.environ['PIPER_DP_RANK'])

    x = torch.randint(0, config.vocab_size, (batch_size, config.block_size))
    input_pos = torch.arange(config.block_size)
    y = torch.randn(batch_size, config.block_size, config.vocab_size)

    schedule = build_1f1b_schedule(num_mbs, pp_degree)
    print_schedule(schedule)
    model = piper_setup(
        Transformer, 
        [config], 
        torch.optim.Adam, 
        [x, input_pos], 
        schedule,
    )

    # Send data to actors ahead of time
    actors = piper_metadata.actors
    ray.get(actors[0].load_input.remote([x]))
    ray.get(actors[len(actors)-1].load_labels.remote(y))

    loss_fn = torch.nn.CrossEntropyLoss()

    start = time.perf_counter()
    piper_exec(model, schedule, [x, input_pos], y, loss_fn)
    end = time.perf_counter()
    print(f"Iter time: {end - start:.2f} s")

    ray.timeline(f"out/mixtral.json")

if __name__ == "__main__":
    ray.init(include_dashboard=False, log_to_driver=True, namespace="llama", _temp_dir="/m-coriander/coriander/mfris/tmp/ray")
    dp_degree, pp_degree = 2, 2
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=pp_degree, dp_degree=dp_degree)
    handles = piper_coordinator.run_program.remote(main, dp_degree, pp_degree)
    ray.get(handles)
    ray.shutdown()
