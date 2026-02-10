import ray
import torch
import threading
import os
import gc
import copy

from .piper_actor import _create_actors
from .piper_utils import piper_metadata, create_logger, LOG_LEVEL
from .piper import piper

logger = create_logger("piper_compile", LOG_LEVEL)


def piper_setup(model_class, model_args, optim_fn, example_inputs, schedule, naive_gradient_sync=False):
    """
    Compile a model with the piper backend.
    """
    piper_metadata.naive_gradient_sync = naive_gradient_sync

    num_devices = len(schedule)
    num_mbs = len(set([task.mb_idx for row in schedule for task in row if task is not None]))
    piper_metadata.stage_to_device = {task.stage_id: task.device_id for row in schedule for task in row if task is not None}

    _create_actors(num_devices, optim_fn, num_mbs, naive_gradient_sync)
    ray.get([actor._join_process_groups.remote() for actor in piper_metadata.actors.values()])

    model = model_class(*model_args)

    compiled = torch.compile(model, backend=piper)

    dp_rank = int(os.environ['PIPER_DP_RANK'])
    logger.info(f"DP rank {dp_rank+1} compiling...")

    output = compiled(*example_inputs)

    logger.info(f"DP rank {dp_rank+1} done.")

    from ray.experimental.collective import create_collective_group
    
    create_collective_group(
        list(piper_metadata.actors.values()),
        backend="nccl")
    
    logger.info(f"DP rank {dp_rank} Started NCCL group with {len(piper_metadata.actors)} actors and DAG {piper_metadata.dag}")
    
    return compiled
