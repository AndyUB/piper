import ray
import torch
import uuid
import inspect
import json, importlib, operator
import torch.fx as fx
import threading
from collections import defaultdict
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from typing import Any, Optional

DEBUG_LOGGING = False


"""
Piper thread local storage for tracking Piper actors, stages, and microbatches
"""

class ThreadLocal(threading.local):
    events = None
    mb_idx = None
    actor_mutexes = None

events_tls = ThreadLocal()

class PiperMetadata:
    actors = dict()
    dag = set()
    currently_compiling = True
    current_stage = None
    current_actor = None
    first_graph_of_stage = None

piper_metadata = PiperMetadata()

"""
Remote tensors wrap Ray ObjectRefs
"""

_fake_tensor_mode = FakeTensorMode()
_fake_tensor_converter = _fake_tensor_mode.fake_tensor_converter

class RemoteTensorKey:
    def __init__(self):
        self.key = str(uuid.uuid4())

class RemoteTensor(torch.Tensor):
    _fake: torch.Tensor
    _stage_id: Optional[int]
    _obj_ref: ray._raylet.ObjectRef
    _resolved: Any

    def __new__(cls, 
                fake: FakeTensor, 
                obj_ref: ray._raylet.ObjectRef,
                stage_id: Optional[int] = None):
        instance = torch.Tensor._make_wrapper_subclass(
            cls,
            fake.size(),
            strides=fake.stride(),
            storage_offset=fake.storage_offset(),
            device=fake.device,  # This is the device of of either input tensor or first tensor of a list
            dtype=fake.dtype,
            layout=fake.layout,
            requires_grad=fake.requires_grad,
        )
        instance._obj_ref = obj_ref
        instance._fake = fake
        instance._stage_id = stage_id
        instance._resolved = None
        instance.key = RemoteTensorKey()
        return instance

    def get_stage_id(self):
        return self._stage_id

    def get(self):
        if self._resolved is None:
            obj = ray.get(self._obj_ref)
            if isinstance(obj, list) or isinstance(obj, tuple):
                assert len(obj) == 1
                self._resolved = obj[0]
            else:
                self._resolved = obj
        return self._resolved
    
    def get_ref(self):
        return self._obj_ref

    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if piper_metadata.currently_compiling:
            def unwrap_fake(x):
                if isinstance(x, RemoteTensor):
                    return x._fake
                return x
            args = torch.utils._pytree.tree_map(unwrap_fake, args)
            kwargs = torch.utils._pytree.tree_map(unwrap_fake, kwargs or {})
            return func(*args, **kwargs)
            
        def unwrap(x):
            if isinstance(x, RemoteTensor):
                return x.get()
            elif isinstance(x, (list, tuple)):
                return type(x)(unwrap(v) for v in x)
            elif isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            else:
                return x

        args = torch.utils._pytree.tree_map(unwrap, args)
        kwargs = torch.utils._pytree.tree_map(unwrap, kwargs or {})

        out = func(*args, **kwargs)
        return out

"""
Serialize/deserialize an fx.GraphModule
"""

def encode_arg(a):
    if isinstance(a, fx.Node):
        return {"__node__": a.name}
    if isinstance(a, torch.device):
        return {"__device__": str(a)}
    if isinstance(a, torch.dtype):
        return {"__dtype__": str(a).replace("torch.", "")}
    if isinstance(a, slice):
        return {"__slice__": True,
                "start": encode_arg(a.start),
                "stop": encode_arg(a.stop),
                "step": encode_arg(a.step)}
    if a is Ellipsis:
        return {"__ellipsis__": True}
    if isinstance(a, tuple):  # <-- preserve tuples
        return {"__tuple__": [encode_arg(x) for x in a]}
    if isinstance(a, list):
        return [encode_arg(x) for x in a]
    if isinstance(a, dict):
        return {k: encode_arg(v) for k, v in a.items()}
    return a

def decode_arg(a, name_to_node):
    if isinstance(a, dict):
        if "__node__" in a:
            return name_to_node[a["__node__"]]
        if "__device__" in a:
            return torch.device(a["__device__"])
        if "__dtype__" in a:
            return getattr(torch, a["__dtype__"])
        if "__slice__" in a:
            return slice(
                decode_arg(a["start"], name_to_node),
                decode_arg(a["stop"], name_to_node),
                decode_arg(a["step"], name_to_node),
            )
        if "__ellipsis__" in a:
            return Ellipsis
        if "__tuple__" in a:  # <-- reconstruct tuples
            return tuple(decode_arg(x, name_to_node) for x in a["__tuple__"])
        # generic dict
        return {k: decode_arg(v, name_to_node) for k, v in a.items()}
    if isinstance(a, list):
        return [decode_arg(x, name_to_node) for x in a]
    return a

def _is_op_overload(obj):
    # Works across PyTorch versions without importing private types directly
    return obj.__class__.__module__.startswith("torch._ops") or obj.__class__.__name__.startswith("OpOverload")

def serialize_target(t):
    # print("SERIALIZING", t)
    # call_method uses a string method name, pass through
    if isinstance(t, str):
        return {"kind": "string", "value": t}

    # Handle _VariableFunctionsClass
    if getattr(t, "__module__", "") == "torch._VariableFunctionsClass":
        public_name = t.__name__
        if hasattr(torch, public_name):
            return {"kind": "py_func", "module": "torch", "qualname": public_name}
        else:
            raise ValueError(f"No public torch alias for {t}")

    # torch.ops.* (aten, prim, etc.)
    if _is_op_overload(t) or (getattr(t, "__module__", "").startswith("torch._ops")):
        return {"kind": "torch_op", "path": str(t)}  # e.g. "aten.add.Tensor" or "aten.add"

    # regular python function or built-in
    if inspect.isfunction(t) or inspect.isbuiltin(t):
        mod = inspect.getmodule(t)
        if mod is None:
            raise ValueError(f"Cannot serialize function without module: {t}")
        return {"kind": "py_func", "module": mod.__name__, "qualname": t.__name__}

    # classes or callables rarely appear as call_function targets, but support anyway
    if inspect.isclass(t):
        mod = t.__module__
        return {"kind": "py_obj", "module": mod, "qualname": t.__qualname__}

    # operator functions (already covered by py_func, but ensure resolvable)
    if t in operator.__dict__.values():
        return {"kind": "py_func", "module": "operator", "qualname": t.__name__}

    # last resort: try module+name
    mod = getattr(t, "__module__", None)
    name = getattr(t, "__name__", None)
    if mod and name:
        return {"kind": "py_func", "module": mod, "qualname": name}

    raise NotImplementedError(f"Unsupported target type: {t} ({type(t)})")

def _resolve_qualname(mod, qualname):
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj

def deserialize_target(payload):
    # print("DESERIALIZING", payload)
    kind = payload["kind"]

    if kind == "string":
        return payload["value"]

    if kind == "py_func":
        mod = importlib.import_module(payload["module"])
        return _resolve_qualname(mod, payload["qualname"])

    if kind == "py_obj":
        mod = importlib.import_module(payload["module"])
        return _resolve_qualname(mod, payload["qualname"])

    if kind == "torch_op":
        obj = torch.ops
        for part in payload["path"].split("."):
            obj = getattr(obj, part)
        return obj

    raise NotImplementedError(f"Unknown target kind: {kind}")

def serialize_graphmodule(gm: fx.GraphModule) -> str:
    nodes = []
    for n in gm.graph.nodes:
        nodes.append({
            "name": n.name,
            "op": n.op,
            "target": serialize_target(n.target) if n.op in ("call_function", "call_method", "call_module", "get_attr") else None,
            "args": encode_arg(n.args),
            "kwargs": encode_arg(n.kwargs),
        })

    data = {
        "nodes": nodes,
        "state_dict": {k: v.detach().cpu().tolist() for k, v in gm.state_dict().items()},
        # save which device parameters were on, optional:
        "param_devices": {k: str(v.device) for k, v in gm.state_dict().items()},
    }
    serialized = json.dumps(data, ensure_ascii=False)

    del data
    del nodes

    return serialized

def _unwrap_output_arg(decoded):
    # FX stores output as (value,), where value may itself be a tuple.
    # Inductor expects the inner tuple directly.
    if isinstance(decoded, (tuple, list)) and len(decoded) == 1 and isinstance(decoded[0], (tuple, list)):
        return decoded[0]
    return decoded

def deserialize_graphmodule(s: str) -> fx.GraphModule:
    data = json.loads(s)
    g = fx.Graph()
    name_to_node = {}

    for n in data["nodes"]:
        op = n["op"]
        if op == "placeholder":
            node = g.placeholder(n["name"])
        elif op == "output":
            decoded = decode_arg(n["args"], name_to_node)
            node = g.output(_unwrap_output_arg(decoded))
        elif op == "call_function":
            target = deserialize_target(n["target"])
            args = decode_arg(n["args"], name_to_node)
            kwargs = decode_arg(n["kwargs"], name_to_node)
            node = g.call_function(target, tuple(args), kwargs)
        elif op == "call_method":
            target = deserialize_target(n["target"])
            args = decode_arg(n["args"], name_to_node)
            kwargs = decode_arg(n["kwargs"], name_to_node)
            node = g.call_method(target, tuple(args), kwargs)
        elif op == "call_module":
            target = deserialize_target(n["target"])
            args = decode_arg(n["args"], name_to_node)
            kwargs = decode_arg(n["kwargs"], name_to_node)
            node = g.call_module(target, tuple(args), kwargs)
        elif op == "get_attr":
            target = deserialize_target(n["target"])
            node = g.get_attr(target)
        else:
            raise NotImplementedError(f"op {op} not handled")
        name_to_node[n["name"]] = node

    gm = fx.GraphModule(torch.nn.Module(), g)
    state = {k: torch.tensor(v) for k, v in data["state_dict"].items()}
    gm.load_state_dict(state, strict=False)
    return gm