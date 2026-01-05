# Piper

Piper is a PyTorch library for training large models with flexible pipeline parallel schedules.

## Set-up Directions
We assume a Linux-based environment

1. Install UV
Follow the [Installing uv instructions](https://docs.astral.sh/uv/getting-started/installation/) from the uv documentation, or run
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Validate the installation by running
```
uv --version
```

2. Create venv
```
uv venv .venv
source .venv/bin/activate
```

3. Install PyTorch (Nightly)
```
uv pip install --pre \
  torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu129
```

4. Install Remaining Dependencies from `pyproject.toml`
```
uv pip install -e .
```

5. Modify dependencies located in `.venv/lib/`

PyTorch

Piper RemoteTensor is not traceable by TorchDynamo. 
- WIP: following FakeTensor, register operator implementations that will make RemoteTensor transparently traceable by TorchDynamo.
- Modification: Add to the beginning of [`transform` in convert_frame.py](https://github.com/pytorch/pytorch/blob/f9724db4921288a096e331cee835abd43257fbd6/torch/_dynamo/convert_frame.py#L1257):
```
####### PIPER MODIFICATION START #######
# Instead of tracing RemoteTensors, trace their
# underlying FakeTensor
from src.piper_utils import RemoteTensor
for k, v in locals.items():
    if isinstance(v, RemoteTensor):
        locals[k] = v._fake
####### PIPER MODIFICATION END #######
```

The torch.compile backend interface is too limiting. 
Compiler backends accept a graph module and example inputs.
`graphargs` is datastructure describing the arguments to a grpah module, including their source in the top-level module and an example.
The list of example inputs required by the torch.compile backend interface are derived from the graph args datastructure. 
The Piper backend requires source information to differentiate model parameters from inputs. 
- Modification: Change the invocation of [self.call_user_compiler in output_graph.py](https://github.com/pytorch/pytorch/blob/f9724db4921288a096e331cee835abd43257fbd6/torch/_dynamo/output_graph.py#L2217):
```
####### PIPER MODIFICATION START #######
# compiled_fn = self.call_user_compiler(gm, self.example_inputs())
compiled_fn = self.call_user_compiler(gm, self.graphargs)
####### PIPER MODIFICATION END #######
```
 
Ray: Tensor transport backends currently only support 1 return value per task.
- WIP: Upstream this into Ray.
- Modifications (2): Comment out the [assertion in ActorMethod._remote()](https://github.com/ray-project/ray/blob/b70d990db786a1f2259dec0504acccf2590353f3/python/ray/actor.py#L824-L828) and add logic for [handling multiple return values with a GPU object manager](https://github.com/ray-project/ray/blob/b70d990db786a1f2259dec0504acccf2590353f3/python/ray/actor.py#L880-L887).
```
####### PIPER MODIFICATION START #######
# if num_returns != 1:
#     raise ValueError(
#         f"Currently, methods with tensor_transport={tensor_transport.name} only support 1 return value. "
#         "Please make sure the actor method is decorated with `@ray.method(num_returns=1)` (the default)."
#     )
####### PIPER MODIFICATION END #######
```
```
####### PIPER MODIFICATION START #######
gpu_object_manager = ray._private.worker.global_worker.gpu_object_manager
if isinstance(object_refs, ObjectRef):
    object_ref = object_refs
    gpu_object_manager.add_gpu_object_ref(
        object_ref, self._actor, tensor_transport
    )
else:
    for object_ref in object_refs:
        assert isinstance(object_ref, ObjectRef)
        gpu_object_manager.add_gpu_object_ref(
            object_ref, self._actor, tensor_transport
        )
####### PIPER MODIFICATION END #######
```
