import torch
import torch.nn as nn
import torch.fx as fx
import copy
from torch.utils.weak import TensorWeakRef

from visualize_graph import fx_to_graphviz
from test.models.llama import Transformer, LLAMA_DEBUG  # kept for later use
from torch.nn import Parameter

PRINT_FX_GRAPHS = True

torch._dynamo.config.compiled_autograd = True

# Global storage for captured backward graphs
#   CAPTURED["bwd_input"]  = (gm_in, example_inputs_in)
#   CAPTURED["bwd_weight"] = (gm_wt, example_inputs_wt)
CAPTURED = {}


class OneLayerToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TwoLayerToy(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=10, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def is_backward_graph(gm: fx.GraphModule) -> bool:
    """
    Heuristic: compiled-autograd backward graphs usually have call_function
    nodes whose targets live in 'torch._dynamo.compiled_autograd.*'.
    """
    for node in gm.graph.nodes:
        if node.op == "call_function":
            mod = getattr(node.target, "__module__", "")
            if "torch._dynamo.compiled_autograd" in mod:
                return True
    return False

def unwrap_example(ex):
    # If it's a list/tuple, recurse
    if isinstance(ex, (list, tuple)):
        return [unwrap_example(e) for e in ex]

    # Already a tensor or parameter
    if isinstance(ex, torch.Tensor):
        return ex

    # TensorWeakRef case
    if isinstance(ex, TensorWeakRef):
        # Be defensive about method names across versions
        if hasattr(ex, "get_tensor"):
            t = ex.get_tensor()
        elif hasattr(ex, "get"):
            t = ex.get()
        elif callable(ex):
            # some weakref-like objects are callable
            t = ex()
        else:
            t = None
        return t

    # Anything else / underlying tensor GC'd
    return None


def make_capture_backend(tag: str):
    """
    Backend factory that captures the FIRST backward FX graph seen for a given tag.
    It returns the graph unchanged so TorchDynamo / AOTAutograd can still run.
    """
    def backend(gm: fx.GraphModule, example_inputs):
        if is_backward_graph(gm):
            print(f"\n=== CAPTURED BACKWARD GRAPH: {tag} ===")
            if PRINT_FX_GRAPHS:
                gm.print_readable()

            if tag not in CAPTURED:
                CAPTURED[tag] = (gm, example_inputs)
                print(
                    f"Stored backward graph for tag='{tag}' with "
                    f"{len(example_inputs)} example_inputs."
                )
                placeholders = gm.graph.find_nodes(op="placeholder", sort=True)
                print("placeholders:", placeholders)
                print("len(placeholders):", len(placeholders))

                placeholder_types = []
                graphargs = [node.meta["grapharg"] for node in placeholders]
                print("graphargs:", graphargs)
                print("len(graphargs):", len(graphargs))
                for arg in graphargs:
                    arg_ex = arg._example
                    arg_ex = arg_ex if isinstance(arg_ex, (list, tuple)) else [arg_ex]
                    for ex in arg_ex:
                        print(f"example input: {ex}")
                        ex = unwrap_example(ex)
                        print(f"unwrapped: {ex}")

                        if ex is None:
                            continue

                        is_param = any(isinstance(t, torch.nn.Parameter) for t in (ex if isinstance(ex, (list, tuple)) else [ex]))
                        requires_grad = any(getattr(t, "requires_grad", False) for t in (ex if isinstance(ex, (list, tuple)) else [ex]))
                        print(f"parameter: {is_param} | requires_grad: {requires_grad} | shape: {[t.shape for t in (ex if isinstance(ex, (list, tuple)) else [ex]) if isinstance(t, torch.Tensor)]}")
                        placeholder_types.append("param" if is_param else "input")

        else:
            if PRINT_FX_GRAPHS:
                print(f"\n=== FORWARD GRAPH seen in backend '{tag}' ===")
                gm.print_readable()

        return gm

    return backend


# ====== Pure helper: forward prep ======

def forward_prepare(model, x):
    """
    Run a forward pass producing:
      - scalar loss y
      - leaf input x_leaf with requires_grad=True
      - tuple of parameters
    """
    x_leaf = x.clone().detach().requires_grad_(True)
    y = model(x_leaf).sum()
    params = tuple(p for p in model.parameters())
    return y, x_leaf, params


# ====== Compiled backward helpers that call autograd.grad ======

@torch.compile(backend=make_capture_backend("bwd_input"))
def bwd_input_fn(y, x):
    # grads w.r.t. input only
    (gx,) = torch.autograd.grad(y, x)
    return gx


@torch.compile(backend=make_capture_backend("bwd_weight"))
def bwd_weight_fn(y, params):
    # grads w.r.t. parameters only
    gparams = torch.autograd.grad(y, params)
    return gparams


# ====== Utilities for checking correctness ======

def flatten_grads(grads):
    if isinstance(grads, dict):
        return list(grads.values())
    if isinstance(grads, (list, tuple)):
        return list(grads)
    return [grads]


def compare_grad_lists(a, b, atol=1e-6, rtol=1e-6):
    assert len(a) == len(b), f"Different #grads: {len(a)} vs {len(b)}"

    for i, (g1, g2) in enumerate(zip(a, b)):
        if g1 is None or g2 is None:
            print(f"grad[{i}] is None")
            continue

        if not torch.allclose(g1, g2, atol=atol, rtol=rtol):
            max_diff = (g1 - g2).abs().max().item()
            print(f"grad[{i}] mismatch — max diff: {max_diff:.6f}")
        else:
            print(f"grad[{i}] match")


def autograd_grads(model, x):
    """
    Vanilla PyTorch autograd: return [grad_x, grad_param_0, grad_param_1, ...]
    so we can compare against the compiled-backward FX graphs.
    """
    with torch.enable_grad():
        model = copy.deepcopy(model)  # avoid contaminating original model's .grad
        model.zero_grad(set_to_none=True)

        x_leaf = x.clone().detach().requires_grad_(True)
        y = model(x_leaf).sum()
        y.backward()

        grads = [x_leaf.grad]
        grads += [p.grad for _, p in model.named_parameters()]

    return grads


# ====== Main demo: capture & compare backward FX graphs ======

def main():
    torch.manual_seed(0)

    model = TwoLayerToy()
    x = torch.randn(1, 10)

    # 1) Ground truth grads via vanilla autograd
    true_grads = autograd_grads(model, x)
    print("\n=== Vanilla autograd grads collected ===")

    # 2) Run forward + compiled backward for inputs ONCE to capture 'bwd_input' backward FX graph
    y1, x1, params1 = forward_prepare(model, x)
    _ = bwd_input_fn(y1, x1)   # triggers capture for "bwd_input"

    # 3) Run forward + compiled backward for weights ONCE to capture 'bwd_weight' backward FX graph
    y2, x2, params2 = forward_prepare(model, x)
    _ = bwd_weight_fn(y2, params2)  # triggers capture for "bwd_weight"

    assert "bwd_input" in CAPTURED, "Did not capture bwd_input backward graph"
    assert "bwd_weight" in CAPTURED, "Did not capture bwd_weight backward graph"

    gm_in, ex_in = CAPTURED["bwd_input"]
    gm_wt, ex_wt = CAPTURED["bwd_weight"]

    print("\nCaptured backward FX graphs:")
    print("BWD_Input Placeholders:", [n for n in gm_in.graph.nodes if n.op == "placeholder"])
    print("len(BWD_Input Placeholders):", len([n for n in gm_in.graph.nodes if n.op == "placeholder"]))
    print("BWD_Weight Placeholders:", [n for n in gm_wt.graph.nodes if n.op == "placeholder"])
    print("len(BWD_Weight Placeholders):", len([n for n in gm_wt.graph.nodes if n.op == "placeholder"]))


    print("\n=== Using captured backward FX graphs directly ===")
    with torch.no_grad():
        out_in = gm_in(*ex_in)     # grads w.r.t. input
        out_wt = gm_wt(*ex_wt)     # grads w.r.t. params

    out_in_list = flatten_grads(out_in)
    out_wt_list = flatten_grads(out_wt)

    print("bwd_input outputs:", [g.shape for g in out_in_list if isinstance(g, torch.Tensor)])
    print("bwd_weight outputs:", [g.shape for g in out_wt_list if isinstance(g, torch.Tensor)])

    # 4) Compare combined grads vs vanilla autograd
    compiled_grads = out_in_list + out_wt_list

    print("\n=== SPLIT vs PYTORCH AUTOGRAD CHECK ===")
    compare_grad_lists(compiled_grads, true_grads)

    # 5) (Optional) visualize graphs
    if PRINT_FX_GRAPHS:
        fx_to_graphviz(gm_in, filename="backward_input_fx")
        fx_to_graphviz(gm_wt, filename="backward_weight_fx")
        print("Saved backward_input_fx and backward_weight_fx graphs to files.")


if __name__ == "__main__":
    main()
