import ray
import torch
from torch._dynamo.backends.debugging import eager

from .piper_utils import piper_metadata

def piper_setup(model, example_inputs, dynamic=False, backend=None):
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

    piper_metadata['currently_compiling'] = True

    print("[PIPER] Compiling...")

    out = compiled(*example_inputs)

    print("[PIPER] Compiled...")

    piper_metadata['currently_compiling'] = False

    from ray.experimental.collective import create_collective_group
    create_collective_group(
        list(piper_metadata['actors'].values()),
        backend="nccl")
    
    print("[PIPER] Started NCCL group")

    return compiled