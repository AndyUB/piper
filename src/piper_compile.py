import ray
import torch
from torch._dynamo.backends.debugging import eager
import threading
from .piper_utils import piper_tls

def piper_setup(model, example_inputs, num_stages, dynamic=False, backend=None):
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

    piper_tls.currently_compiling = True
    piper_tls.events[None] = [threading.Event() for _ in range(num_stages)]
    for event in piper_tls.events[None]:
        event.set()

    print("[PIPER] Compiling...")

    out = compiled(*example_inputs, dynamo_mb=None).get()

    print("[PIPER] Compiled...")

    piper_tls.currently_compiling = False

    # from ray.experimental.collective import create_collective_group
    # create_collective_group(
    #     list(piper_tls.actors.values()),
    #     backend="nccl")
    
    # print("[PIPER] Started NCCL group")

    return compiled