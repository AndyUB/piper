
"""
Split forward functions at __resume_at_* globals.
This is to allow for interleaving forward tasks with other tasks.
"""

import dis, functools
from dataclasses import dataclass
from typing import Optional, Callable
from bytecode import Bytecode, Instr, Label

import torch.distributed as dist
import os

from .piper_utils import piper_metadata

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
def setup_data_parallel(local_rank, data_parallel):
    """ Just adds every rank to the same process group"""
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=data_parallel)
    torch.cuda.set_device(local_rank)

def piper_setup(model, example_inputs, dynamic=False, backend=None):
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
    #data_parallel_size = piper_metadata['parallelism_configs']['dp']
    #print(f"TEMPORARY: data_parallel_size={data_parallel_size}")
    #local_rank = int(os.environ['LOCAL_RANK'])
    #setup_data_parallel(local_rank, data_parallel_size)
    #print("Verified the setup_data_parallel ...")
    compiled = torch.compile(model, dynamic=dynamic, backend=backend)

    print("[PIPER] Compiling...")

    out = compiled(*example_inputs).get()

    print("[PIPER] Compiled...")
    # TODO: now that we compile everything, make every StageActor process join the process group. question: do the run_dp_rank controller processes have to join the process group
    futures_for_joining_process_group = []
    piper_rank = os.environ.get('PIPER_RANK')
    print(f"[PIPER RANK {piper_rank}]: piper_metadata[actors]={piper_metadata['actors']}")
    print(f"[piper_setup after compile] piper_metadata id: {id(piper_metadata)}, actors dict id: {id(piper_metadata['actors'])}")
    for actor_id, actor in piper_metadata['actors'].items():
        future = actor.join_process_group.remote()
        futures_for_joining_process_group.append(future)
        
    


    # TODO: put code here that synchronizes the StageActors across the dp ranks by invoking piper_utils::actors::join_dp_group or something
    
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
    import ray
    from ray.experimental.collective import create_collective_group
    # TODO: is there an issue here where, if you make this code compile on each DP rank, they each start their own collectives?
    create_collective_group(
        list(piper_metadata['actors'].values()),
        backend="nccl")
    
    print("[PIPER] Started NCCL group")

    # # Set up per-stage DP process groups for gradient synchronization
    # rank = int(os.environ.get('PIPER_RANK', '0'))
    # world_size = int(os.environ.get('PIPER_WORLD_SIZE', '1'))
    # num_stages = int(os.environ.get('NUM_STAGES', '1'))
    
    # if world_size > 1:
    #     print(f"[PIPER] Setting up DP process groups for rank {rank}/{world_size} with {num_stages} stages")
        
    #     # Group actors by their stage_id (without DP offset)
    #     # actor_id = stage_id + rank * num_stages
    #     # So stage_id = actor_id % num_stages (for this simple case)
    #     stage_to_actors = {}
    #     for actor_id, actor in piper_metadata['actors'].items():
    #         # Get the stage_id_for_dp from the actor
    #         stage_id_for_dp = ray.get(actor.id.remote()) % num_stages
    #         if stage_id_for_dp not in stage_to_actors:
    #             stage_to_actors[stage_id_for_dp] = []
    #         stage_to_actors[stage_id_for_dp].append((actor_id, actor))
        
    #     # For now, we'll create a simple per-stage group marker
    #     # In a full implementation, this would set up torch.distributed subgroups
    #     # across all DP ranks. Since torch.distributed setup across Ray actors
    #     # is complex and requires shared rendezvous, we'll set a flag that
    #     # indicates the DP group is ready.
    #     # The actual synchronization will use the default torch.distributed group
    #     # or a more sophisticated mechanism.
        
    #     # For each stage, tell actors they're part of a DP group
    #     for stage_id, actors in stage_to_actors.items():
    #         print(f"[PIPER] Setting up DP group for stage {stage_id} with {len(actors)} actors")
    #         for actor_id, actor in actors:
    #             # Set up the DP process group for this actor
    #             # For now, we pass None as the group, which will use the default group
    #             ray.get(actor.set_dp_process_group.remote(None))

     
    print(f"[PIPER] Completed DP group setup; len(actors)={len(piper_metadata['actors'])}")
    
    # Return both the compiled model and the metadata dict that was populated during compilation
    # This is necessary because torch.compile creates a separate namespace during tracing,
    # so the global piper_metadata in the caller's scope is different from the one used during compilation
    return compiled, piper_metadata
