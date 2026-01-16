import ray
import torch
import threading
import os
import gc

from .piper_actor import create_actor
from .piper_utils import piper_metadata, RemoteTensor, create_logger
from .piper import piper

logger = create_logger("piper_compile", "INFO")

def setup_data_parallel(local_rank, data_parallel):
    """ Just adds every rank to the same process group"""
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=data_parallel)
    torch.cuda.set_device(local_rank)

def piper_setup(model_class, model_args, optim_fn, example_inputs, num_stages, num_devices, dynamic=False):
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
    
    for actor_id in range(num_devices):
        create_actor(actor_id, optim_fn)

    model = model_class(*model_args)

    piper_metadata.currently_compiling = True

    compiled = torch.compile(model, dynamic=dynamic, backend=piper)

    from .piper_utils import events_tls
    events_tls.actor_mutexes = dict([(actor_id, threading.Lock()) for actor_id in range(num_devices)])
    events_tls.events = [threading.Event() for _ in range(num_stages)]
    for event in events_tls.events:
        event.set()

    dp_rank = int(os.environ['PIPER_DP_RANK'])
    logger.info(f"DP rank {dp_rank} compiling {num_stages} stages...")

    out = compiled(*example_inputs).get()

    logger.info(f"DP rank {dp_rank} finished compiling model. DAG: {piper_metadata.dag}")
    
    piper_metadata.currently_compiling = False

    ray.get([actor.join_process_groups.remote() for actor in piper_metadata.actors.values()])

    logger.info(f"Completed DP rank {dp_rank} setup for {len(piper_metadata.actors)} actors")

    from ray.experimental.collective import create_collective_group
    
    create_collective_group(
        list(piper_metadata.actors.values()),
        backend="nccl")
    
    logger.info(f"Started NCCL group with {len(piper_metadata.actors)} actors")
    
    return compiled
