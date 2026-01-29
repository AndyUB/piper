# Piper

Piper is a PyTorch library for training large models with flexible pipeline parallel schedules.

## Environment setup: conda
We assume a Linux-based environment

1. Create a conda environment with `python==3.10`
2. Install the requirements in `requirements.txt`
3. Modify PyTorch and Ray dependencies according to the instructions below

## Modifying Ray dependency

**Ray**

Tensor transport backends currently only support 1 return value per task.
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

## Training Llama in Piper
The llama test program `test/test_llama.py` supports GPipe, 1F1B and interleaved 1F1B schedules for 2 or 4 devices.
The `test/models/llama.py` file has example `forward` methods for one stage, two stage, and four stage partitions. 
Ensure that the correct `forward` method is uncommented for the desired schedule (e.g. two stage for 1F1B on 2 devices, four stage for interleaved 1F1B on two devices). 
DP training can also be turned on with the `dp_degree` flag.
Run the Llama test program for the 1F1B schedule for two devices:
```
python3 -m test.test_llama --model LLAMA_DEBUG --schedule 1f1b --num_stages 2 --pp_degree 2 --dp_degree 1
```
