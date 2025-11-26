import torch
import torch.fx as fx
import graphviz


def fx_to_graphviz(
    gm: fx.GraphModule,
    filename: str = "fx_graph",
    format: str = "png",
    show_attrs: bool = True,
):
    """
    Convert a torch.fx.GraphModule to a Graphviz visualization.

    Args:
        gm (fx.GraphModule): The traced model
        filename (str): Output file name (no extension)
        format (str): 'png', 'pdf', 'svg', etc.
        show_attrs (bool): Show full node target names
    """
    dot = graphviz.Digraph(format=format, graph_attr={"rankdir": "LR"})

    for node in gm.graph.nodes:
        label = f"{node.op}\n{node.name}"

        if show_attrs and node.target not in ("output", "placeholder"):
            label += f"\n{node.target}"

        dot.node(node.name, label)

        for user in node.users:
            dot.edge(node.name, user.name)

    dot.render(filename, cleanup=True)
    return dot
