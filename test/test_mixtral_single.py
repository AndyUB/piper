import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh
import time

from torch.distributed import pipelining as pippy  # adjust if your version differs
from .models.mixtral_a2a import Transformer, ModelArgs


def main() -> None:
    """Run a simple Mixtral tiny pipeline-parallel + data-parallel training step."""
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    config = ModelArgs.from_name("tiny")

    model = Transformer(config, group=None)
    model.to(device)

    batch_size = 16
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=device)
    input_pos = torch.arange(config.block_size, device=device)
    y = torch.randn((batch_size, config.block_size, config.vocab_size), device=device)

    warmup = 3
    iters = 5

    for _ in range(warmup):
        model(x, input_pos)

    start = time.perf_counter()
    for _ in range(iters):
        model(x, input_pos)
    end = time.perf_counter()

    print(f"Iter time: {1e3*(end - start)/iters:.2f} ms")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()