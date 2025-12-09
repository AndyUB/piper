import torch
import torch.nn as nn
import torch.fx as fx
from torch.utils.weak import TensorWeakRef
import copy
from visualize_graph import fx_to_graphviz
from test.models.llama import Transformer, LLAMA_DEBUG
from torch.nn import Parameter

PRINT_FX_GRAPHS = False

torch._dynamo.config.compiled_autograd = True

placeholder_types = []

SPLIT_GRADS = None

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



def extract_subgraph(gm: fx.GraphModule, output_nodes):
    graph = gm.graph

    needed = set(output_nodes)
    queue = list(output_nodes)

    while queue:
        node = queue.pop()
        for inp in node.all_input_nodes:
            if inp not in needed:
                needed.add(inp)
                queue.append(inp)

    for node in graph.nodes:
        if node.op == "placeholder":
            needed.add(node)

    new_graph = fx.Graph()
    node_copies = {}

    for node in graph.nodes:
        if node.op == "output":
            continue
        if node in needed:
            new_node = new_graph.node_copy(node, lambda n: node_copies[n])
            node_copies[node] = new_node

    new_outputs = [node_copies[n] for n in output_nodes]
    if len(new_outputs) == 1:
        new_graph.output((new_outputs[0],))
    else:
        new_graph.output(tuple(new_outputs))

    new_gm = fx.GraphModule(gm, new_graph)
    new_gm.recompile()
    return new_gm


def split_backward_graph(
    gm: fx.GraphModule,
    input_grad_indices,
    param_grad_indices,
):
    graph = gm.graph
    output_node = None
    for n in graph.nodes:
        if n.op == "output":
            output_node = n
            break
    assert output_node is not None, "No output node in FX graph?"

    outputs_tuple = output_node.args[0]
    assert isinstance(outputs_tuple, (tuple, list)), "Expected tuple of outputs"

    input_grad_nodes = [outputs_tuple[i] for i in input_grad_indices]
    param_grad_nodes = [outputs_tuple[i] for i in param_grad_indices]

    gm_input = extract_subgraph(gm, input_grad_nodes)
    gm_weight = extract_subgraph(gm, param_grad_nodes)
    return gm_input, gm_weight


def is_backward_graph(gm: fx.GraphModule) -> bool:
    for node in gm.graph.nodes:
        if node.op == "call_function":
            mod = getattr(node.target, "__module__", "")
            if "torch._dynamo.compiled_autograd" in mod:
                return True
    return False

def forward_backend(gm: fx.GraphModule, example_inputs):
    global placeholder_types

    print("FORWARD GRAPH")
    if PRINT_FX_GRAPHS:
        gm.print_readable()

    placeholders = gm.graph.find_nodes(op="placeholder", sort=True)
    print("placeholders:", placeholders)

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

    print("placeholder_types:", placeholder_types)
    return gm


def backward_backend(gm: fx.GraphModule, example_inputs):
    if not is_backward_graph(gm):
        global placeholder_types

        print("FORWARD GRAPH")
        if PRINT_FX_GRAPHS:
            gm.print_readable()

        placeholders = gm.graph.find_nodes(op="placeholder", sort=True)
        print("placeholders:", placeholders)

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

        print("placeholder_types:", placeholder_types)
        return gm

    print("FULL BACKWARD GRAPH")
    if PRINT_FX_GRAPHS:
        gm.print_readable()
    # print("placeholder_types:", placeholder_types)

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

    print("placeholder_types:", placeholder_types)

    return gm

    fx_to_graphviz(gm, filename="full_backward_graph")
    print("Saved full backward graph visualization to file.")

    graph = gm.graph
    output_node = next(n for n in graph.nodes if n.op == "output")
    outputs_tuple = output_node.args[0]

    if not isinstance(outputs_tuple, (tuple, list)):
        outputs_tuple = (outputs_tuple,)

    num_outputs = len(outputs_tuple)

    print(outputs_tuple)
    for out in outputs_tuple:
        print("output:", type(out), out.name, out.target, out.op, out.args, out.kwargs)

    assert num_outputs == len(placeholder_types), (
        f"Mismatch: backward outputs ({num_outputs}) "
        f"vs forward grad targets ({len(placeholder_types)})"
    )

    num_inputs_grads = placeholder_types.count("input")
    num_parameters_grads = placeholder_types.count("param")
    input_grad_indices = range(num_inputs_grads)
    param_grad_indices = range(num_inputs_grads, num_inputs_grads + num_parameters_grads)

    print("input_grad_indices:", input_grad_indices)
    print("param_grad_indices:", param_grad_indices)
    

    # input_grad_indices = [1]
    # param_grad_indices = [0, 2, 3, 4]

    gm_input, gm_weight = split_backward_graph(
        gm,
        input_grad_indices=input_grad_indices,
        param_grad_indices=param_grad_indices,
    )

    print("BACKWARD INPUT SUBGRAPH")
    if PRINT_FX_GRAPHS:
        gm.print_readable()

    print("BACKWARD WEIGHT SUBGRAPH")
    if PRINT_FX_GRAPHS:
        gm.print_readable()

    with torch.no_grad():
        full_out = gm(*example_inputs)
        in_out = gm_input(*example_inputs)
        w_out = gm_weight(*example_inputs)

    print("BACKWARD OUTPUT CHECK")
    print("full_out shapes:", [t.shape for t in full_out])
    print("input_out shapes:", [t.shape for t in in_out])
    print("weight_out shapes:", [t.shape for t in w_out])

    full_out_list = flatten_grads(full_out)
    split_out_list = flatten_grads(in_out) + flatten_grads(w_out)

    global SPLIT_GRADS
    SPLIT_GRADS = split_out_list

    print("SPLIT vs FULL GRAPH CHECK")
    compare_grad_lists(full_out_list, split_out_list)


    fx_to_graphviz(gm_input, filename="backward_input_subgraph")
    fx_to_graphviz(gm_weight, filename="backward_weight_subgraph")
    print("Saved subgraph visualizations to files.")

    return gm

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
            print(f"grad[{i}] mismatch — max diff: {(g1 - g2).abs().max().item():.6f}")
        else:
            print(f"grad[{i}] match")


def autograd_grads(model, x):
    with torch.enable_grad():
        copy.deepcopy(model)
        model.zero_grad(set_to_none=True)

        x = x.clone().detach().requires_grad_(True)
        y = model(x).sum()
        y.backward()

        grads = [x.grad]
        grads += [p.grad for _, p in model.named_parameters()]

    return grads


@torch.compile(backend=backward_backend)
def train(model, x):
    x = x.clone().detach().requires_grad_(True)
    y = model(x).sum()
    y.backward()
    return y



def main():
    global MODEL, X, Y
    torch.manual_seed(0)

    # llama_config = LLAMA_DEBUG
    # loss_fn = torch.nn.CrossEntropyLoss()
    # # device = 'cuda'
    
    # batch_size = 8
    # seq_len = 512
    # x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len))
    # y = torch.zeros((batch_size, llama_config.vocab_size), dtype=torch.long)

    # print_cuda_memory_stats(device, "after loading data")

    model = TwoLayerToy()
    x = torch.randn(1, 10)


    loss = train(model, x)
    print("Loss:", loss.item())

    # true_grads = autograd_grads(model, x)

    # print("SPLIT vs PYTORCH AUTOGRAD CHECK")
    # compare_grad_lists(SPLIT_GRADS, true_grads)

if __name__ == "__main__":
    main()
