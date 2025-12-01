import torch
import torch.nn as nn
import torch.fx as fx
import copy
from visualize_graph import fx_to_graphviz

torch._dynamo.config.compiled_autograd = True

SPLIT_GRADS = None

class OneLayerToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

class TwoLayerToy(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=8, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


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


def custom_backend(gm: fx.GraphModule, example_inputs):
    if not is_backward_graph(gm):
        print("FORWARD GRAPH")
        gm.print_readable()
        return gm

    print("FULL BACKWARD GRAPH")
    gm.print_readable()

    fx_to_graphviz(gm, filename="full_backward_graph")
    print("Saved full backward graph visualization to file.")

    input_grad_indices = [0]
    param_grad_indices = [1,2]

    gm_input, gm_weight = split_backward_graph(
        gm,
        input_grad_indices=input_grad_indices,
        param_grad_indices=param_grad_indices,
    )

    print("BACKWARD INPUT SUBGRAPH")
    gm_input.print_readable()

    print("BACKWARD WEIGHT SUBGRAPH")
    gm_weight.print_readable()

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


@torch.compile(backend=custom_backend)
def train(model, x):
    x = x.clone().detach().requires_grad_(True)
    y = model(x).sum()
    y.backward()
    return y


def main():
    global MODEL, X
    torch.manual_seed(0)
    model = OneLayerToy()
    x = torch.randn(1, 10)

    loss = train(model, x)
    print("Loss:", loss.item())

    true_grads = autograd_grads(model, x)

    print("SPLIT vs PYTORCH AUTOGRAD CHECK")
    compare_grad_lists(SPLIT_GRADS, true_grads)

if __name__ == "__main__":
    main()
