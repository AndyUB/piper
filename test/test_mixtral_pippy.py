import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import pipelining as pippy  # adjust if your version differs
from .models.mixtral_a2a import Transformer, ModelArgs


def main() -> None:
    """Run a simple Mixtral tiny pipeline-parallel + data-parallel training step."""
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dp_degree = 2
    pp_degree = 2
    assert world_size == dp_degree * pp_degree, "This example assumes 2-way DP and 2-way PP (world_size=4)"

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create a 2x2 device mesh for 2-way data parallel (DP) and 2-way pipeline parallel (PP)
    # Mesh layout: [dp_rank, pp_rank]
    mesh = DeviceMesh(
        "cuda",
        torch.arange(world_size).reshape(dp_degree, pp_degree),
        mesh_dim_names=("dp", "pp"),
    )

    # Label DP/PP groups and compute this process's rank within each group
    dp_group = mesh.get_group(mesh_dim="dp")
    pp_group = mesh.get_group(mesh_dim="pp")

    dp_rank = rank // pp_degree
    pp_rank = rank % pp_degree

    dp_devices = [f"cuda:{i}" for i in range(world_size) if i % pp_degree == pp_rank]
    print(f"dp_devices: {dp_devices}")

    print(
        f"Global rank {rank} -> dp_rank={dp_rank}, pp_rank={pp_rank}, "
        f"dp_group={dp_group}, pp_group={pp_group}"
    )

    # Tiny config from your mixtral-a2a.py
    config = ModelArgs.from_name("tiny")
    config.n_layer = config.n_layer // pp_degree

    model = Transformer(config, dp_group)

    if pp_rank == 0:
        model.norm = None
        model.output = None
    if pp_rank == 1:
        model.tok_embeddings = None
    
    model.to(device)

    model = DDP(model, device_ids=dp_devices)

    # NOTE: The exact constructor signature may vary slightly between PyTorch versions.
    stage = pippy.PipelineStage(
        model,
        pp_rank,
        pp_degree,
        device,
        group=pp_group,
    )

    batch_size = 16
    mbs = 1
    loss_fn = nn.CrossEntropyLoss()
    schedule = pippy.Schedule1F1B(stage, mbs, loss_fn)

    x = torch.randint(0, config.vocab_size, (batch_size*mbs, config.block_size), device=device)
    input_pos = torch.arange(config.block_size, device=device)
    y = torch.randn((batch_size*mbs, config.block_size, config.vocab_size), device=device)

    warmup = 3
    iters = 5

    for _ in range(warmup):
        if pp_rank == 0:
            schedule.step(x, input_pos)
        else:
            schedule.step()

    start = time.perf_counter()
    for _ in range(iters):
        if pp_rank == 0:
            schedule.step(x, input_pos)
        else:
            schedule.step()
    end = time.perf_counter()

    print(f"Iter time: {(end - start)/iters:.2f} s")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()