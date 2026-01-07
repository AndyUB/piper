import ray
import torch
from .piper_utils import piper_metadata, RemoteTensor
from torch._dynamo.backends.debugging import eager
import threading
import gc


def piper_setup(model, example_inputs, num_stages, num_devices, dynamic=False, backend=None):
    """
    Compile a model with the piper backend.

    Args:
        model: A model to compile.
        example_inputs: Example inputs to the model.
        dynamic: Whether to compile in dynamic mode.
        backend: The backend to use for compilation.

    Returns:
        A compiled model.
    """
    if not backend:
        backend = eager
    
    compiled = torch.compile(model, dynamic=dynamic, backend=backend)

    piper_metadata.currently_compiling = True
    from .piper_utils import events_tls
    events_tls.actor_mutexes = dict([(actor_id, threading.Lock()) for actor_id in range(num_devices)])
    events_tls.events = [threading.Event() for _ in range(num_stages)]
    for event in events_tls.events:
        event.set()

    print(f"[PIPER] Compiling {num_stages} stages...")

    out = compiled(*example_inputs).get()
    print("[PIPER] Compiled...")

    piper_metadata.currently_compiling = False

    from ray.experimental.collective import create_collective_group
    create_collective_group(
        list(piper_metadata.actors.values()),
        backend="nccl")
    
    print(f"[PIPER] Started NCCL group with {len(piper_metadata.actors)} actors")

    return compiled