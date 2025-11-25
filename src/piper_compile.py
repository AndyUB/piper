
"""
Split forward functions at __resume_at_* globals.
This is to allow for interleaving forward tasks with other tasks.
"""

import dis, functools
from dataclasses import dataclass
from typing import Optional, Callable
from bytecode import Bytecode, Instr, Label

import ray

from .piper_utils import piper_metadata, RemoteTensor

@dataclass(frozen=True)
class CallMarker:
    next_call: str
    args: tuple
    kwargs: dict

def _make_stub(name: str) -> Callable:
    def _stub(*args, **kwargs):
        return CallMarker(name, args, kwargs)
    _stub.__name__ = f"__stub_{name}"
    return _stub

def _find_resume_name(fn: Callable) -> Optional[str]:
    instrs = list(dis.get_instructions(fn))
    for i, ins in enumerate(instrs):
        if isinstance(ins.argval, str) and ins.argval.startswith("__resume_at_"):
            return ins.argval
    return None

def _should_split(code: Bytecode) -> bool:
    """
    Return True iff there is a LOAD_ATTR distributed_stage before
    a call to a __resume_at_* global.
    """
    for instr in code:
        if hasattr(instr, "name") and hasattr(instr, "arg"):
            if instr.name == "LOAD_ATTR" and instr.arg == "distributed_stage":
                return True
    return False

def split_before(fn: Callable):
    """
    Wrap `fn`. If `fn` contains a single call to a name starting with "__resume_at_",
    the wrapper replaces that global with a stub that returns CallMarker(args, kwargs).
    The wrapper accepts either:
      - normal calling form: wrapper(*args, **kwargs)
      - chained form: wrapper(marker) where marker is CallMarker
    In chained form the marker's args/kwargs are used as the call to `fn`.
    """
    code = Bytecode.from_code(fn.__code__)
    g = fn.__globals__
    resume_name = _find_resume_name(fn)
    should_split = _should_split(code)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # allow chained form: single CallMarker positional argument
        if len(args) == 1 and not kwargs and isinstance(args[0], CallMarker):
            call_args, call_kwargs = args[0].args, args[0].kwargs
        else:
            call_args, call_kwargs = args, kwargs

        # if no resume_name found, just invoke the function (still accept marker)
        if not should_split:
            return fn(*call_args, **call_kwargs)
        
        # print(f"Stubbing {fn}")

        assert resume_name is not None
        
        # stub the target resume global, call, then restore
        saved = {}
        if resume_name in g:
            # print("STUBBING", resume_name)
            saved[resume_name] = g[resume_name]
            g[resume_name] = _make_stub(resume_name)
        try:
            return fn(*call_args, **call_kwargs)
        finally:
            # restore original binding if we changed it
            if resume_name in saved:
                g[resume_name] = saved[resume_name]

    return wrapper


import torch
from torch._dynamo.backends.debugging import eager

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

    def make_remote(x):
        return RemoteTensor(x, ray.put(x))
    # example_inputs = list(map(make_remote, example_inputs))
    out = compiled(*example_inputs).get()

    print("[PIPER] Compiled...")
    
    # import inspect
    # output_codes = torch._dynamo.convert_frame.output_codes
    # transformed_fns = {}
    # for name, fn in output_codes.piper_fns.items():
    #     transformed_fns[name] = split_before(fn)

    # out = (model, *example_inputs)
    # next_call = "forward"
    # fwd_fns = []
    # while next_call:
    #     fn = transformed_fns[next_call]
    #     fwd_fns.append(fn)
    #     if next_call == "forward":
    #         out = fn(*out, dynamo_mb=999)
    #     elif isinstance(out, list) or isinstance(out, tuple):
    #         out = fn(*out)
    #     else:
    #         out = fn(out)
    #     if isinstance(out, CallMarker):
    #         next_call = out.next_call
    #     else:
    #         next_call = None

    # setattr(compiled, "_piper_fwd_fns", fwd_fns)

    # # MEMORY CLEANUP
    # torch._dynamo.convert_frame.output_codes.piper_fns.clear()
    # torch._dynamo.convert_frame.output_codes.name_map.clear()
    # import gc
    # gc.collect()

    piper_metadata['currently_compiling'] = False

    from ray.experimental.collective import create_collective_group
    create_collective_group(
        list(piper_metadata['actors'].values()),
        backend="nccl")
    
    print("[PIPER] Started NCCL group")

    return compiled