# Piper

Piper is a PyTorch library for training large models with flexible pipeline parallel schedules.

## Dependencies

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

Piper RemoteTensor causes recompilation bugs because it's not traceable by TorchDynamo. 
- WIP: same as above
- Modification: Add at the beginning of [`CheckFunctionManager.__init__` in guards.py](https://github.com/pytorch/pytorch/blob/f9724db4921288a096e331cee835abd43257fbd6/torch/_dynamo/guards.py#L3532):
```
####### PIPER MODIFICATION START #######
def filter_guards(guard):
    return not (
        guard.inner_create_fn().__name__ == "TENSOR_MATCH" or 
        guard.name == "L['dynamo_mb']")
guards = list(filter(filter_guards, guards))
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
