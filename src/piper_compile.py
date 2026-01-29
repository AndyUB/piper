import ray
import torch
import threading
import os
import gc
import copy

from .piper_actor import create_actors
from .piper_utils import piper_metadata, RemoteTensor, create_logger, LOG_LEVEL
from .piper import piper

logger = create_logger("piper_compile", LOG_LEVEL)


def piper_setup(model_class, model_args, optim_fn, example_inputs, num_stages, pp_degree, dynamic=False, check_correct=False):
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
    create_actors(pp_degree, optim_fn)

    model = model_class(*model_args)

    if check_correct:
        model_nocompile = copy.deepcopy(model)

    piper_metadata.currently_compiling = True

    compiled = torch.compile(model, dynamic=dynamic, backend=piper)

    from .piper_utils import events_tls
    events_tls.actor_mutexes = dict([(actor_id, threading.Lock()) for actor_id in range(pp_degree)])
    events_tls.events = [threading.Event() for _ in range(num_stages)]
    for event in events_tls.events:
        event.set()

    dp_rank = int(os.environ['PIPER_DP_RANK'])
    logger.info(f"DP rank {dp_rank+1} compiling {num_stages} stages...")

    output = compiled(*example_inputs).get()

    if check_correct:
        correct_output = model_nocompile(*example_inputs)
        if not torch.allclose(output, correct_output):
            logger.error(f"Model output is not correct")
            logger.error(f"Compiled output: {output}")
            logger.error(f"Correct output: {correct_output}")
        else:
            logger.info(f"Model output is correct")

    logger.info(f"DP rank {dp_rank+1} finished compiling model. DAG: {piper_metadata.dag}")
    
    piper_metadata.currently_compiling = False

    logger.info(f"DP rank {dp_rank} joining process groups")

    ray.get([actor.join_process_groups.remote() for actor in piper_metadata.actors.values()])

    logger.info(f"Completed DP rank {dp_rank+1} setup for {len(piper_metadata.actors)} actors")

    from ray.experimental.collective import create_collective_group
    
    create_collective_group(
        list(piper_metadata.actors.values()),
        backend="nccl")
    
    logger.info(f"DP rank {dp_rank} Started NCCL group with {len(piper_metadata.actors)} actors")
    
    return compiled
