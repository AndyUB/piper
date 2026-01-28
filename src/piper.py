import ray
import torch
import os
from .piper_actor import get_actor, dispatch_expert_ray
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.decorators import _disallow_in_graph_helper
from .piper_utils import RemoteTensor, serialize_graphmodule, piper_metadata, create_logger, print_backward_graph, LOG_LEVEL
import threading

logger = create_logger("piper_backend", LOG_LEVEL)


@torch.compiler.disable
def distributed_stage(stage_id, actor_id=None):
    """
    Annotation for stage boundaries, causes torch.compile graph break
    and sets metadata appropriately at compile time
    """
    dp_rank = int(os.environ['PIPER_DP_RANK'])
    world_size = int(os.environ['PIPER_WORLD_SIZE'])
    dp_degree = int(os.environ['PIPER_DP_DEGREE'])

    if actor_id is None:
        actor_id = stage_id

    piper_metadata.current_stage = stage_id
    piper_metadata.current_actor = actor_id
    piper_metadata.first_graph_of_stage = True



def split_gm_by_experts(gm, stage_id, pp_degree):
    """
    Transform a graph module with expert annotations by extracting
    expert computations into submodules and distributing the expert
    submodules to Ray actors.
    """
    import torch.fx as fx
    from collections import defaultdict
    import operator
    
    # Collect all nodes with custom metadata
    nodes_with_metadata = []
    expert_metadata = {}
    for node in gm.graph.nodes:
        if 'custom' in node.meta:
            custom_meta = node.meta['custom']
            # Extract expert and batch_idx from metadata
            if isinstance(custom_meta, dict):
                global_expert_id = custom_meta.get('global_expert_id')
                local_expert_id = custom_meta.get('local_expert_id')
                batch_idx = custom_meta.get('batch_idx')
                if global_expert_id is not None:
                    expert_metadata[global_expert_id] = (local_expert_id, batch_idx)
                    nodes_with_metadata.append((node, global_expert_id, local_expert_id, batch_idx))
    
    assert nodes_with_metadata, "GraphModule must have 'expert' metadata"
    
    # Group nodes by expert ID for creating expert modules
    expert_code = defaultdict(list)  # expert_id -> list of all nodes for this expert
    
    for node, global_expert_id, _, _ in nodes_with_metadata:
        expert_code[global_expert_id].append(node)
    
    expert_modules = {}
    
    for expert_id, expert_nodes in expert_code.items():
        expert_graph = fx.Graph()
        expert_node_mapping = {}
        
        expert_node_set = set(expert_nodes)
        
        # Find all inputs needed by this expert (nodes that are not in the expert set)
        expert_inputs = set()
        for node in expert_nodes:
            for arg in fx.graph.map_arg(node.args, lambda n: n):
                if isinstance(arg, fx.Node) and arg not in expert_node_set:
                    expert_inputs.add(arg)
        
        # Create placeholders for inputs
        input_placeholders = {}
        for input_node in expert_inputs:
            placeholder = expert_graph.placeholder(input_node.name)
            expert_node_mapping[input_node] = placeholder
            input_placeholders[input_node] = placeholder

        # Find outputs of this expert (nodes used outside the expert set)
        expert_outputs = []
        for node in expert_nodes:
            for user in node.users:
                if user not in expert_node_set:
                    if node not in expert_outputs:
                        expert_outputs.append(node)
        
        # If no external users, use the last node(s) as output
        if not expert_outputs:
            expert_outputs = [expert_nodes[-1]] if expert_nodes else []

        # Topological sort of expert nodes
        def get_dependencies(node):
            deps = set()
            for arg in fx.graph.map_arg(node.args, lambda n: n):
                if isinstance(arg, fx.Node) and arg in expert_node_set:
                    deps.add(arg)
            return deps
        remaining = set(expert_nodes)
        ordered_expert_nodes = []
        while remaining:
            for node in list(remaining):
                deps = get_dependencies(node)
                if deps.issubset(set(ordered_expert_nodes)):
                    ordered_expert_nodes.append(node)
                    remaining.remove(node)
                    break
        
        # Copy expert nodes to the expert graph in topological order
        for node in ordered_expert_nodes:
            new_expert_node = expert_graph.node_copy(
                node,
                lambda n: expert_node_mapping.get(n, input_placeholders.get(n))
            )
            expert_node_mapping[node] = new_expert_node

        # Create output node
        if expert_outputs:
            output_values = [expert_node_mapping[node] for node in expert_outputs]
            if len(output_values) == 1:
                expert_graph.output(output_values[0])
            else:
                expert_graph.output(tuple(output_values))
        else:
            # No outputs, create a dummy output
            expert_graph.output(expert_graph.placeholder('dummy'))
        
        expert_gm = fx.GraphModule(torch.nn.Module(), expert_graph)
        
        # Store the expert module and related info
        for node in ordered_expert_nodes:
            if node.op == "placeholder":
                input_placeholders[node] = node

        # Load the module on the corresponding actor
        input_idxs, param_idxs = [], []
        params = []
        placeholders = expert_gm.graph.find_nodes(op="placeholder")
        for i, placeholder in enumerate(placeholders):
            if "grapharg" in placeholder.meta:
                if 'self' in str(placeholder.meta["grapharg"]):
                    param_idxs.append(i)
                    params.append(placeholder.meta["grapharg"]._example())
                else:
                    input_idxs.append(i)
                    params.append(None)
            else:
                input_idxs.append(i)
                params.append(None)

        # logger.debug(f"Submodule {expert_id} input_idxs: {input_idxs} param_idxs: {param_idxs}, params: {[p.shape if p is not None else None for p in params]}")
        local_expert_id, batch_idx = expert_metadata[expert_id]
        expert_metadata[expert_id] = (local_expert_id, batch_idx, input_idxs)
        actor_id = local_expert_id % pp_degree
        actor = get_actor(actor_id)
        ray.get(actor.load_expert.remote(stage_id, expert_id, local_expert_id, batch_idx, expert_gm, input_idxs, param_idxs, params))

        expert_modules[expert_id] = (expert_gm, input_placeholders, expert_outputs, expert_nodes, ordered_expert_nodes)

    # Create a new top-level graph and replace expert nodes with call_module
    new_graph = fx.Graph()
    node_mapping = {}
    
    # Copy placeholders
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
            node_mapping[node] = new_node
    
    # Track which expert calls have been replaced
    expert_call_replaced = {}
    
    # Copy all get_attr nodes (parameters) so they're available for expert calls
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            if node not in node_mapping:
                new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
                node_mapping[node] = new_node
    
    # Process nodes in the original graph order to preserve structure
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # Already handled
            continue
        
        if node.op == "output":
            # Handle output separately
            continue
        
        if node.op == "get_attr":
            # Already handled above
            continue
        
        if node in node_mapping:
            # Already processed
            continue
        
        # Check if this node belongs to an expert and find which expert call it's part of
        in_expert = False
        for expert_id, expert_nodes in expert_code.items():
            if node in expert_nodes:
                in_expert = True
                
                # Get the expert module info
                expert_gm, input_placeholders, expert_outputs, expert_nodes, ordered_expert_nodes = expert_modules[expert_id]
                
                # Check if we've already created the call_module for this expert call
                if expert_id not in expert_call_replaced:
                    # Find first node in this expert's nodes (in original graph order)
                    first_node = None
                    for n in gm.graph.nodes:
                        if n in expert_nodes:
                            first_node = n
                            break
                    
                    if first_node is None:
                        first_node = expert_nodes[0]
                    
                    # Map inputs in the same order as placeholders
                    # Use the inputs that the first node of this expert needs
                    mapped_args = []
                    for input_node in input_placeholders.keys():
                        if input_node in node_mapping:
                            mapped_args.append(node_mapping[input_node])
                        else:
                            # Input node not yet mapped, need to copy it first
                            new_input_node = new_graph.node_copy(input_node, lambda n: node_mapping.get(n, n))
                            node_mapping[input_node] = new_input_node
                            mapped_args.append(new_input_node)
                    
                    # Dispatch expert module to a Ray actor at the position of the first expert node for this expert
                    module_name = f"expert_{expert_id}"
                    local_expert_id, batch_idx, input_idxs = expert_metadata[expert_id]
                    input_tensors_only = []
                    for i in input_idxs:
                        input_tensors_only.append(mapped_args[i])
                    # expert_module_node = new_graph.get_attr(module_name)
                    ray_call_args = [expert_id, local_expert_id, batch_idx, pp_degree] + input_tensors_only
                    call_node = new_graph.call_function(dispatch_expert_ray, tuple(ray_call_args))
                    expert_call_replaced[expert_id] = (call_node, expert_outputs)
                
                # Map expert nodes to the call_module output
                call_node, expert_outputs = expert_call_replaced[expert_id]
                
                # Only create output mapping for output nodes (nodes used outside expert)
                if node in expert_outputs:
                    if len(expert_outputs) == 1:
                        # Single output, use call_node directly
                        node_mapping[node] = call_node
                    else:
                        # Multiple outputs, need to getitem
                        output_idx = expert_outputs.index(node)
                        getitem_node = new_graph.call_function(
                            operator.getitem,
                            (call_node, output_idx)
                        )
                        node_mapping[node] = getitem_node
                else:
                    node_mapping[node] = call_node
                break
        
        if not in_expert:
            # Regular node, copy it
            new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
            node_mapping[node] = new_node
    
    
    # Handle output node
    output_node = None
    for node in gm.graph.nodes:
        if node.op == "output":
            output_node = node
            break
    
    if output_node:
        output_args = fx.graph.map_arg(output_node.args, lambda n: node_mapping.get(n, n))
        if isinstance(output_args, tuple) and len(output_args) == 1:
            new_graph.output(output_args[0])
        else:
            new_graph.output(output_args)
    
    # Create root module and add expert modules as submodules
    root_module = torch.nn.Module()
    for expert_id, (expert_gm, _, _, _, _) in expert_modules.items():
        module_name = f"expert_{expert_id}"
        root_module.add_module(module_name, expert_gm)
    
    new_gm = fx.GraphModule(root_module, new_graph)
    
    # Copy parameters, buffers, and modules from original module
    for name, param in gm.named_parameters(recurse=False):
        if name not in new_gm._parameters:
            new_gm.register_parameter(name, param)
    
    for name, buffer in gm.named_buffers(recurse=False):
        if name not in new_gm._buffers:
            new_gm.register_buffer(name, buffer)
    
    for name, module in gm.named_children():
        if name not in [f"expert_{expert_id}" for expert_id in expert_modules.keys()]:
            new_gm.add_module(name, module)
    
    new_gm.recompile()
    
    return new_gm

@register_backend
def piper(gm, example_inputs, **kwargs):
    """
    torch.compile backend loads the graph module on 
    a Ray actor and returns a callback that remotely
    runs the graph module. 
    """
    logger.debug(f"Compiling subgraph {id(gm)}")

    if not piper_metadata.currently_compiling:
        gm.print_readable()
        assert False, "Piper backend called outside of compilation"

    # Distribute expert submodules to actors if there are expert annotations
    stage_id = piper_metadata.current_stage
    pp_degree = int(os.environ['PIPER_PP_DEGREE'])
    original_gm = gm
    gm = split_gm_by_experts(gm, stage_id, pp_degree)

    # For the top-level graph, log which arguments are input tensors
    # vs parameter tensors and make sure all example inputs are serializable

    placeholders = gm.graph.find_nodes(op="placeholder")
    graphargs = [node.meta["grapharg"] for node in placeholders]

    # make sure example inputs are serializable by turning symbolic
    # ints and fake tensors into concrete values
    serializable_examples = []
    input_idxs = []
    param_idxs = []
    for i, (arg, ex) in enumerate(zip(graphargs, example_inputs)):
        # save indices of input tensors and model parameters
        if 'self' not in str(arg):
            input_idxs.append(i)
        else:
            param_idxs.append(i)
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
    stage_id = piper_metadata.current_stage
    actor_id = piper_metadata.current_actor
    actor = get_actor(actor_id)

    dp_rank = int(os.environ['PIPER_DP_RANK'])
    dp_degree = int(os.environ['PIPER_DP_DEGREE'])
    global_rank = dp_rank * dp_degree + actor_id
    graph_id = id(gm)
    original_param_id = sum([id(example_inputs[i]) for i in param_idxs])
    
    ray.get(
        actor.load_graph.remote(
            stage_id,
            graph_id,
            original_param_id,
            payload,
            torch._dynamo.backends.debugging.eager,
            serializable_examples,
            input_idxs,
        )
    )

    # Get fake tensor representations of the graph output(s)
    def symint_to_int(x):
        return int(x) if isinstance(x, torch.SymInt) else x
    def int_to_tensor(x):
        return torch.tensor(x) if isinstance(x, int) else x
    example_inputs = list(map(symint_to_int, serializable_examples))
    fakes = original_gm(*example_inputs)
    fakes = list(map(int_to_tensor, fakes))

    # logger.debug(f"Stage {stage_id} backward graph:")
    # print_backward_graph(logger.debug, fakes[0])

    # wait for a signal to run this graph if it's the first graph of the stage
    first_graph_of_stage = piper_metadata.first_graph_of_stage
    if first_graph_of_stage:
        piper_metadata.first_graph_of_stage = False
    
    # return a wrapper function that runs the fx.Graph on the actor and 
    # returns remote futures for each graph output
    def run_remote_subgraph(*args):

        logger.debug(f"Running subgraph {id(gm)}")

        from .piper_utils import events_tls

        mb_idx = events_tls.mb_idx

        # wait for a signal to run the partial graph
        if first_graph_of_stage:
            logger.debug(f"Thread {mb_idx} global rank {global_rank} waiting for stage {stage_id}")
            events_tls.events[stage_id].wait()
            logger.debug(f"Thread {mb_idx} global rank {global_rank} running stage {stage_id}")

        # Mutex ensures that only one thread submits a task to this actor at a time
        logger.debug(f"Thread {mb_idx} global rank {global_rank} waiting for actor mutex {actor_id}")
        with events_tls.actor_mutexes[actor_id]:
            logger.debug(f"Thread {mb_idx} global rank {global_rank} got actor mutex {actor_id}")

            # clear the event for the next stage
            if first_graph_of_stage:
                events_tls.events[stage_id].clear()

            # use the correct set of parameters, send them to the actor if we haven't seen them before
            param_id = sum([id(args[i]) for i in param_idxs])
            if param_id != original_param_id:
                logger.debug(f"Calling subgraph with new params")
                if piper_metadata.currently_compiling:
                    ray.get(actor.add_param_group.remote(stage_id, graph_id, param_id, args, input_idxs))

            # ignore model parameter arguments (stored on the actor)
            input_tensors_only = []
            for i in input_idxs:
                input_tensors_only.append(args[i])
            args = input_tensors_only
            
            # track stage dependencies
            for arg in args:
                if isinstance(arg, RemoteTensor):
                    prev_stage = arg.get_stage_id()
                    if prev_stage != stage_id:
                        piper_metadata.dag.add((prev_stage, stage_id))

            # get Ray ObjectRefs from RemoteTensors
            def unwrap(x):
                return x.get_ref() if isinstance(x, RemoteTensor) else x
            args = list(map(unwrap, args))

            if piper_metadata.currently_compiling:
                # dispatch task without nccl transport
                refs = actor.forward_cpu.options(num_returns=len(fakes)).remote(stage_id, graph_id, param_id, mb_idx, *args)
            else:
                # dispatch with nccl transport
                refs = actor.forward.options(num_returns=len(fakes)).remote(stage_id, graph_id, param_id, mb_idx, *args)

            # wrap the remote futures with RemoteTensor
            # if piper_metadata.currently_compiling:
            #     return [t.to('cpu') for t in ray.get(refs)]

            if isinstance(refs, list):
                assert len(fakes) == len(refs)
                return [RemoteTensor(fake, ref, stage_id) for fake, ref in zip(fakes, refs)]
            else:
                assert len(fakes) == 1
                return [RemoteTensor(fakes[0], refs, stage_id)]
    return run_remote_subgraph