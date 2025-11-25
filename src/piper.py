import ray
import torch
from .piper_actor import StageActor
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.decorators import _disallow_in_graph_helper
from .piper_utils import RemoteTensor, serialize_graphmodule, piper_metadata

@_disallow_in_graph_helper(throw_if_not_allowed=False)
def distributed_stage(stage_id, actor_id=None, mb=None, optim=None):
    if actor_id is None:
        actor_id = stage_id
    piper_metadata['current_mb'] = mb
    if actor_id not in piper_metadata['actors']:
        actor = StageActor.options(num_gpus=1).remote(actor_id, optim_fn=optim)
        piper_metadata['actors'][actor_id] = actor
        piper_metadata['stage_fns'][stage_id] = None
    piper_metadata['current_stage'] = stage_id
    piper_metadata['current_actor'] = actor_id

@register_backend
def piper(gm, example_inputs, **kwargs):

    placeholders = gm.graph.find_nodes(op="placeholder")
    graphargs = [node.meta["grapharg"] for node in placeholders]

    # make sure example inputs are serializable by turning symbolic
    # ints and fake tensors into concrete values
    serializable_examples = []
    input_idxs = []
    for i, (arg, ex) in enumerate(zip(graphargs, example_inputs)):
        # save indices of input tensors and model parameters
        if 'self' not in str(arg):
            input_idxs.append(i)

        # convert symbolic ints and fake tensors to concrete values
        if isinstance(ex, torch.SymInt):
            serializable_examples.append(int(ex))
        elif isinstance(ex, torch._subclasses.fake_tensor.FakeTensor):
            new = torch.full(
                ex.shape,
                0,
                dtype=ex.dtype,
                device=ex.device,
                layout=ex.layout,
                requires_grad=ex.requires_grad,
            )
            serializable_examples.append(new)
        else:
            serializable_examples.append(ex)

    # serialize the fx.Graph
    payload = serialize_graphmodule(gm)

    # send the fx.Graph and model attributes to the actor
    stage_id = piper_metadata['current_stage']
    actor_id = piper_metadata['current_actor']
    actor = piper_metadata['actors'][actor_id]
    ray.get(
        actor.compile_graph.remote(
            stage_id,
            payload,
            torch._dynamo.backends.debugging.eager,
            serializable_examples,
            input_idxs,
        )
    )

    # get a list of fake tensor outputs from the fx.Graph
    def symint_to_int(x):
        return int(x) if isinstance(x, torch.SymInt) else x
    def int_to_tensor(x):
        return torch.tensor(x) if isinstance(x, int) else x
    fakes = gm(*list(map(symint_to_int, serializable_examples)))
    fakes = list(map(int_to_tensor, fakes))

    # return a wrapper function that runs the fx.Graph on the actor and 
    # returns remote futures for each graph output
    def run_remote_subgraph(*args):
        # ignore model parameter arguments (stored on the actor)
        input_tensors_only = []
        for i in input_idxs:
            input_tensors_only.append(args[i])
        args = input_tensors_only
        
        mb_idx = piper_metadata['current_mb']
        # track stage dependencies
        for arg in args:
            if isinstance(arg, RemoteTensor):
                piper_metadata['dag'].add((arg.get_stage_id(), stage_id))

        # get Ray ObjectRefs from RemoteTensors
        def unwrap(x):
            return x.get_ref() if isinstance(x, RemoteTensor) else x
        args = list(map(unwrap, args))

        if piper_metadata['currently_compiling']:
            # dispatch task without nccl transport
            refs = actor.forward_no_nccl.options(num_returns=len(fakes)).remote(stage_id, mb_idx, *args)
        else:
            # dispatch with nccl transport
            refs = actor.forward.options(num_returns=len(fakes)).remote(stage_id, mb_idx, *args)

        # wrap the remote futures with RemoteTensor
        if isinstance(refs, list):
            assert len(fakes) == len(refs)
            return [RemoteTensor(fake, ref, stage_id) for fake, ref in zip(fakes, refs)]
        else:
            assert len(fakes) == 1
            return [RemoteTensor(fakes[0], refs, stage_id)]
    return run_remote_subgraph