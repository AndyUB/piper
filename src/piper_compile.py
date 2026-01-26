import ray
import torch
from .piper_utils import piper_metadata, RemoteTensor, create_logger
from torch._dynamo.backends.debugging import eager
import threading
import os
import gc

logger = create_logger("piper_compile", "INFO")

def setup_data_parallel(local_rank, data_parallel):
    """ Just adds every rank to the same process group"""
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=data_parallel)
    torch.cuda.set_device(local_rank)

def piper_setup(model, example_inputs, num_stages, pp_size, dynamic=False, backend=None):
    """
    Compile a model with the piper backend.

    Args:
        model: A model to compile.
        example_inputs: Example inputs to the model.
        dynamic: Whether to compile in dynamic mode.
        backend: The backend to use for compilation.

    Returns:
        A tuple of (compiled_model, piper_metadata) where piper_metadata contains
        the actors, stage_fns, and other state populated during compilation.
    """
    if not backend:
        backend = eager

    piper_metadata.currently_compiling = True

    compiled = torch.compile(model, dynamic=dynamic, backend=backend)

    from .piper_utils import events_tls
    events_tls.actor_mutexes = dict([(actor_id, threading.Lock()) for actor_id in range(pp_size)])
    events_tls.events = [threading.Event() for _ in range(num_stages)]
    for event in events_tls.events:
        event.set()

    dp_rank = int(os.environ['PIPER_DP_RANK'])
    logger.info(f"DP rank {dp_rank} compiling {num_stages} stages...")

    out = compiled(*example_inputs).get()
    
    piper_metadata.currently_compiling = False

    logger.info(f"DP rank {dp_rank} joining process groups")

    ray.get([actor.join_process_groups.remote() for actor in piper_metadata.actors.values()])

    logger.info(f"DP rank {dp_rank} completed setup for {len(piper_metadata.actors)} actors")

    from ray.experimental.collective import create_collective_group
    
    create_collective_group(
        list(piper_metadata.actors.values()),
        backend="nccl")
    
    logger.info(f"DP rank {dp_rank} Started NCCL group with {len(piper_metadata.actors)} actors")
    
    return compiled
