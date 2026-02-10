import ray
import torch
import torch.fx as fx
import torch.distributed as dist
from collections import defaultdict
from typing import Optional
import operator
import os
from torch.autograd import Function

from .piper_utils import create_logger, LOG_LEVEL, piper_metadata

logger = create_logger("piper_graph_transform", LOG_LEVEL)


class CommOp:
    def __init__(self, tensor_id: int, name: str, dep: str, pass_type: str, op: str, group: str):
        self.tensor = None # this gets set on the actor
        self.tensor_id = tensor_id
        self.name = name
        self.dep = dep # "pre" or "post"
        self.pass_type = pass_type # "forward" or "backward"
        self.op = op # "allreduce", "allgather", "scatter", "alltoall"
        self.group = group # "dp" or "pp"

def _get_dp_comm_ops(example_inputs, placeholders):
    dp_comm_ops = []
    ids = []
    for p, t in zip(placeholders, example_inputs):
        ids.append(id(t))
        if isinstance(t, torch.nn.Parameter) and t.requires_grad:
            dp_comm_ops.append(CommOp(id(t), p.name, "post", "backward", "allreduce", "dp"))
    logger.debug(f"Got {len(dp_comm_ops)} dp comm ops for {len(example_inputs)} example inputs")
    return dp_comm_ops, ids


class AllToAllSingleFunction(torch.autograd.Function):
    """
    Custom autograd function for all_to_all_single that ensures gradients flow correctly.
    
    This wraps torch.distributed.all_to_all_single to ensure that gradients
    from the input tensor properly flow to the output buffer.
    The backward pass performs another all_to_all_single to reverse the communication.
    """
    @staticmethod
    def forward(ctx, output, input_tensor):
        """
        Forward pass: performs all_to_all_single communication.
        
        Args:
            ctx: Context for storing information for backward pass
            output: Output buffer (will be modified in-place)
            input_tensor: Input tensor to communicate
            group: Process group (optional)
        """
        # Store group for backward pass
        from .piper_utils import piper_metadata
        actor_self = piper_metadata.actor_self
        ctx.group = actor_self.dp_group
        
        # Ensure input_tensor is contiguous (all_to_all_single requires contiguous tensors)
        # TODO: performance cost?
        input_tensor = input_tensor.contiguous()
        
        # Perform the communication (modifies output in-place)
        dist.all_to_all_single(output, input_tensor, group=ctx.group)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: performs reverse all_to_all_single to propagate gradients.
        
        The backward of all_to_all_single is another all_to_all_single operation
        that reverses the communication pattern.
        """
        if grad_output is None:
            return None, None, None
        
        # Ensure grad_output is contiguous (all_to_all_single requires contiguous tensors)
        # TODO: performance cost?
        grad_output = grad_output.contiguous()
        
        # Create a buffer for the gradient input
        grad_input = torch.empty_like(grad_output)
        
        # Reverse the communication: all_to_all_single in backward
        # This propagates gradients from output back to input
        if ctx.group is None:
            dist.all_to_all_single(grad_input, grad_output)
        else:
            dist.all_to_all_single(grad_input, grad_output, group=ctx.group)
        
        # Return gradients: grad_output flows to grad_input, None for group
        return grad_input, grad_input, None

def _dispatch_a2a_single(output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Dispatch a Ray remote call to run an all_to_all_single operation
    using a custom autograd Function. This function is called
    from the FX graph, so it needs to be allowed in the graph.
    """
    return AllToAllSingleFunction.apply(output, input_tensor)

# Allow the dispatch function in the graph
torch.compiler.allow_in_graph(_dispatch_a2a_single)

def _split_gm_by_stages(gm) -> tuple[fx.GraphModule, list[tuple[int, fx.GraphModule, list[int], list]]]:
    """
    Transform a graph module with stage annotations by extracting
    stage computations into submodules and distributing the stage
    submodules to Ray actors.
    
    Returns:
        A tuple of (new_graph_module, list_of_stage_info) where:
        - new_graph_module: The transformed top-level graph module
        - list_of_stage_info: List of (stage_id, submodule, input_idxs, all_inputs) tuples for each extracted stage
    """
    # Collect all nodes with custom metadata
    nodes_with_metadata = []
    for node in gm.graph.nodes:
        if 'custom' in node.meta:
            custom_meta = node.meta['custom']
            # Extract stage from metadata
            if isinstance(custom_meta, dict):
                stage_annotation_id = custom_meta.get('stage')
                if stage_annotation_id is not None:
                    nodes_with_metadata.append((node, stage_annotation_id))
    
    if not nodes_with_metadata:
        logger.info("No stage nodes found in graph")
        return gm, []
    else:
        logger.info(f"Found stage nodes in graph")
    
    # Group nodes by stage ID for creating stage modules
    stage_code = defaultdict(list)  # stage_id -> list of all nodes for this stage
    
    for node, stage_annotation_id in nodes_with_metadata:
        stage_code[stage_annotation_id].append(node)
    
    stage_modules = {}
    
    # Process stages in sorted order to track outputs from previous stages
    for stage_annotation_id, stage_nodes in sorted(stage_code.items(), key=lambda x: x[0]):
        stage_graph = fx.Graph()
        stage_node_mapping = {}
        
        stage_node_set = set(stage_nodes)
        
        # Collect all nodes from other stages to identify stage boundaries
        other_stage_nodes = set()
        for other_stage_id, other_stage_nodes_list in stage_code.items():
            if other_stage_id != stage_annotation_id:
                other_stage_nodes.update(other_stage_nodes_list)
        
        # Check if any nodes are outputs from previous stages
        prev_stage_outputs = set()
        for prev_stage_id in range(stage_annotation_id):
            if prev_stage_id in stage_modules:
                _, _, _, prev_outputs, _, _, _, _, _ = stage_modules[prev_stage_id]
                prev_stage_outputs.update(prev_outputs)
        
        # Find all inputs needed by this stage (nodes that are not in the stage set)
        # Check both args and kwargs to find all dependencies
        stage_inputs = set()
        get_attr_nodes = set()  # Track get_attr nodes separately
        
        def extract_nodes_from_arg(arg):
            """Recursively extract all fx.Node objects from an argument"""
            nodes = []
            if isinstance(arg, fx.Node):
                nodes.append(arg)
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    nodes.extend(extract_nodes_from_arg(item))
            elif isinstance(arg, dict):
                for value in arg.values():
                    nodes.extend(extract_nodes_from_arg(value))
            return nodes
        
        for node in stage_nodes:
            # Check args - use both map_arg and explicit extraction for robustness
            for arg in fx.graph.map_arg(node.args, lambda n: n):
                if isinstance(arg, fx.Node) and arg not in stage_node_set:
                    if arg.op == "get_attr":
                        get_attr_nodes.add(arg)
                    elif arg.op == "placeholder":
                        # Placeholders are always inputs
                        stage_inputs.add(arg)
                    elif arg in other_stage_nodes or arg in prev_stage_outputs:
                        # Nodes from other stages or outputs from previous stages are inputs
                        stage_inputs.add(arg)
                    else:
                        # Could be computed in this stage or an input - will determine later
                        stage_inputs.add(arg)
            
            # Check kwargs - explicitly extract nodes from dict values
            # fx.graph.map_arg should work, but let's also manually check kwargs dict
            if node.kwargs:
                # Use map_arg first (should handle most cases)
                for arg in fx.graph.map_arg(node.kwargs, lambda n: n):
                    if isinstance(arg, fx.Node) and arg not in stage_node_set:
                        if arg.op == "get_attr":
                            get_attr_nodes.add(arg)
                        elif arg.op == "placeholder":
                            # Placeholders are always inputs
                            stage_inputs.add(arg)
                        elif arg in other_stage_nodes or arg in prev_stage_outputs:
                            # Nodes from other stages or outputs from previous stages are inputs
                            stage_inputs.add(arg)
                        else:
                            # Could be computed in this stage or an input - will determine later
                            stage_inputs.add(arg)
                
                # Also explicitly extract nodes from kwargs dict values as a fallback
                for kwarg_value in extract_nodes_from_arg(node.kwargs):
                    if kwarg_value not in stage_node_set:
                        if kwarg_value.op == "get_attr":
                            get_attr_nodes.add(kwarg_value)
                        elif kwarg_value.op == "placeholder":
                            stage_inputs.add(kwarg_value)
                        elif kwarg_value in other_stage_nodes or kwarg_value in prev_stage_outputs:
                            stage_inputs.add(kwarg_value)
                        else:
                            stage_inputs.add(kwarg_value)
        
        # Copy get_attr nodes (parameters) to the stage graph first
        # These should be available directly in the stage, not passed as inputs
        for node in gm.graph.nodes:
            if node in get_attr_nodes:
                new_get_attr = stage_graph.node_copy(node, lambda n: n)
                stage_node_mapping[node] = new_get_attr
        
        # Create placeholders for inputs in original graph order to ensure correct argument ordering
        # Preserve all metadata from original nodes
        input_placeholders = {}
        input_order = []  # Track order of inputs as they appear in original graph
        for node in gm.graph.nodes:
            if node in stage_inputs and node not in input_placeholders:
                # If the input is already a placeholder, use node_copy to preserve all metadata
                if node.op == "placeholder":
                    placeholder = stage_graph.node_copy(node, lambda n: n)
                else:
                    # For computed nodes, create a placeholder but preserve metadata
                    placeholder = stage_graph.placeholder(node.name)
                    # Copy all metadata from the original node
                    placeholder.meta.update(node.meta)
                stage_node_mapping[node] = placeholder
                input_placeholders[node] = placeholder
                input_order.append(node)
        
        # Find outputs of this stage (nodes used outside the stage set)
        # First, collect all nodes from other stages to identify stage boundaries
        other_stage_nodes = set()
        for other_stage_id, other_stage_nodes_list in stage_code.items():
            if other_stage_id != stage_annotation_id:
                other_stage_nodes.update(other_stage_nodes_list)
        
        # Check if any nodes are outputs from previous stages
        prev_stage_outputs = set()
        for prev_stage_id in range(stage_annotation_id):
            if prev_stage_id in stage_modules:
                _, _, _, prev_outputs, _, _, _, _, _ = stage_modules[prev_stage_id]
                prev_stage_outputs.update(prev_outputs)
        
        # Find all nodes computed in this stage (dependencies of stage_nodes)
        # These are nodes that are arguments to stage_nodes but not inputs from other stages
        # We need to recursively collect all dependencies
        stage_computed_nodes = set(stage_nodes)
        visited = set()
        
        def collect_dependencies(node):
            """Recursively collect all node dependencies that are computed in this stage"""
            if node in visited:
                return
            visited.add(node)
            
            # Skip if already in computed nodes
            if node in stage_computed_nodes:
                return
            
            # Skip placeholders, get_attr, nodes from other stages, and outputs from previous stages
            if (node.op == "placeholder" or node.op == "get_attr" or 
                node in other_stage_nodes or node in prev_stage_outputs):
                return
            
            # This node is computed in this stage (it's a dependency of a stage node)
            if node in stage_inputs:
                # Remove from inputs if it's computed in this stage
                stage_inputs.discard(node)
            stage_computed_nodes.add(node)
            
            # Recursively collect dependencies of this node from args
            for arg in fx.graph.map_arg(node.args, lambda n: n):
                if isinstance(arg, fx.Node):
                    collect_dependencies(arg)
            
            # Recursively collect dependencies from kwargs - use both map_arg and explicit extraction
            if node.kwargs:
                # Use map_arg first
                for arg in fx.graph.map_arg(node.kwargs, lambda n: n):
                    if isinstance(arg, fx.Node):
                        collect_dependencies(arg)
                
                # Also explicitly extract nodes from kwargs dict values as a fallback
                for kwarg_node in extract_nodes_from_arg(node.kwargs):
                    collect_dependencies(kwarg_node)
        
        # Collect all dependencies of stage nodes
        for stage_node in stage_nodes:
            # Collect from args
            for arg in fx.graph.map_arg(stage_node.args, lambda n: n):
                if isinstance(arg, fx.Node):
                    collect_dependencies(arg)
            
            # Collect from kwargs - use both map_arg and explicit extraction
            if stage_node.kwargs:
                for arg in fx.graph.map_arg(stage_node.kwargs, lambda n: n):
                    if isinstance(arg, fx.Node):
                        collect_dependencies(arg)
                
                # Also explicitly extract nodes from kwargs dict values
                for kwarg_node in extract_nodes_from_arg(stage_node.kwargs):
                    collect_dependencies(kwarg_node)
        
        # Find outputs: nodes computed in this stage that are used by other stages
        # Check if users have stage annotations from other stages, or if they're in already-processed stages
        stage_outputs = []
        for node in stage_computed_nodes:
            for user in node.users:
                # Check if user has a stage annotation from a different stage
                user_in_other_stage = False
                if 'custom' in user.meta:
                    custom_meta = user.meta['custom']
                    if isinstance(custom_meta, dict):
                        user_stage = custom_meta.get('stage')
                        if user_stage is not None and user_stage != stage_annotation_id:
                            user_in_other_stage = True
                
                # Also check if user is in an already-processed stage's computed nodes
                if not user_in_other_stage:
                    for other_stage_id in range(stage_annotation_id):
                        if other_stage_id in stage_modules:
                            _, _, _, _, _, other_ordered_nodes, _, _, _ = stage_modules[other_stage_id]
                            if user in other_ordered_nodes:
                                user_in_other_stage = True
                                break
                
                # Check if user is in another stage (annotated nodes)
                if not user_in_other_stage and user in other_stage_nodes:
                    user_in_other_stage = True
                
                if user_in_other_stage:
                    if node not in stage_outputs:
                        stage_outputs.append(node)
                # Also check if user is the output node (which uses stage outputs)
                elif user.op == "output":
                    if node not in stage_outputs:
                        stage_outputs.append(node)
        
        # If no external users, use the last node(s) as output
        if not stage_outputs:
            # Use the last computed node in topological order
            ordered_computed = [n for n in gm.graph.nodes if n in stage_computed_nodes]
            stage_outputs = [ordered_computed[-1]] if ordered_computed else []
        
        # Preserve original graph order by iterating through nodes in original order
        # This ensures correctness of computation order
        # Include ALL computed nodes, not just annotated ones
        ordered_stage_nodes = []
        for node in gm.graph.nodes:
            if node in stage_computed_nodes:
                ordered_stage_nodes.append(node)
        
        # Copy stage nodes to the stage graph in original graph order
        for node in ordered_stage_nodes:
            new_stage_node = stage_graph.node_copy(
                node,
                lambda n: stage_node_mapping.get(n, input_placeholders.get(n))
            )
            stage_node_mapping[node] = new_stage_node

        # Create output node
        if stage_outputs:
            print(f"Stage {stage_annotation_id} outputs: {[node.name for node in stage_outputs]}")
            output_values = [stage_node_mapping[node] for node in stage_outputs]
            if len(output_values) == 1:
                stage_graph.output(output_values[0])
            else:
                stage_graph.output(tuple(output_values))
        else:
            # No outputs, create a dummy output
            stage_graph.output(stage_graph.placeholder('dummy'))
        
        stage_gm = fx.GraphModule(torch.nn.Module(), stage_graph)
        
        # Store the stage module and related info
        for node in ordered_stage_nodes:
            if node.op == "placeholder":
                input_placeholders[node] = node

        # Load the module on the corresponding actor
        # Track which inputs come from previous stage outputs
        input_idxs = []
        params = []
        placeholders = stage_gm.graph.find_nodes(op="placeholder")
        
        # Create reverse mapping from placeholder to original node
        placeholder_to_original = {placeholder: orig_node for orig_node, placeholder in input_placeholders.items()}
        
        # Get outputs from previous stage (if it exists)
        prev_stage_outputs = set()
        prev_stage_id = stage_annotation_id - 1
        if prev_stage_id in stage_modules:
            _, _, _, prev_stage_outputs_list, _, _, _, _, _ = stage_modules[prev_stage_id]
            prev_stage_outputs = set(prev_stage_outputs_list)
        
        for i, placeholder in enumerate(placeholders):
            # For the first stage, the input indices are everything that's not an attribute
            if stage_annotation_id == 0:
                if 'self' in placeholder.name:
                    params.append(placeholder.meta["grapharg"]._example())
                else:
                    input_idxs.append(i)
                    params.append(None)
            # For subsequent stages, check if this placeholder corresponds to an output from the previous stage
            else:
                orig_node = placeholder_to_original.get(placeholder)
                is_from_prev_stage = orig_node is not None and orig_node in prev_stage_outputs
                # Check if this is from previous stage output
                if is_from_prev_stage:
                    input_idxs.append(i)
                    params.append(None)
                else:
                    params.append(placeholder.meta["grapharg"]._example())
        logger.debug(f"Stage {stage_annotation_id} inputs: {input_idxs}, params: {[param.shape if param is not None else None for param in params]}")
        stage_modules[stage_annotation_id] = (
            stage_gm,
            input_placeholders,
            input_order,
            stage_outputs,
            stage_nodes,
            ordered_stage_nodes,
            stage_annotation_id,
            input_idxs,
            params,
        )

    # Create a new top-level graph and replace stage nodes with call_module
    new_graph = fx.Graph()
    node_mapping = {}
    
    # Copy placeholders
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
            node_mapping[node] = new_node
    
    # Track which stage calls have been replaced
    stage_call_replaced = {}
    
    # Copy all get_attr nodes (parameters) so they're available for stage calls
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
        
        # Check if this node belongs to a stage and find which stage call it's part of
        in_stage = False
        for stage_annotation_id, stage_nodes in stage_code.items():
            if node in stage_nodes:
                in_stage = True
                
                # Get the stage module info
                stage_gm, input_placeholders, input_order, stage_outputs, stage_nodes, ordered_stage_nodes, _, _, _ = stage_modules[stage_annotation_id]
                
                # Check if we've already created the call_module for this stage call
                if stage_annotation_id not in stage_call_replaced:
                    # Find first node in this stage's nodes (in original graph order)
                    first_node = None
                    for n in gm.graph.nodes:
                        if n in stage_nodes:
                            first_node = n
                            break
                    
                    if first_node is None:
                        first_node = stage_nodes[0]
                    
                    # Map inputs in the same order as placeholders (original graph order)
                    mapped_args = []
                    for input_node in input_order:
                        if input_node in node_mapping:
                            mapped_args.append(node_mapping[input_node])
                        else:
                            # Input node not yet mapped, need to copy it first
                            new_input_node = new_graph.node_copy(input_node, lambda n: node_mapping.get(n, n))
                            node_mapping[input_node] = new_input_node
                            mapped_args.append(new_input_node)
                    
                    # Create call_module node for the stage
                    module_name = f"stage_{stage_annotation_id}"
                    call_node = new_graph.call_module(module_name, tuple(mapped_args), {})
                    stage_call_replaced[stage_annotation_id] = (call_node, stage_outputs)
                
                # Map stage nodes to the call_module output
                call_node, stage_outputs = stage_call_replaced[stage_annotation_id]
                
                # Only create output mapping for output nodes (nodes used outside stage)
                if node in stage_outputs:
                    if len(stage_outputs) == 1:
                        # Single output, use call_node directly
                        node_mapping[node] = call_node
                    else:
                        # Multiple outputs, need to getitem
                        output_idx = stage_outputs.index(node)
                        getitem_node = new_graph.call_function(
                            operator.getitem,
                            (call_node, output_idx)
                        )
                        node_mapping[node] = getitem_node
                else:
                    node_mapping[node] = call_node
                break
        
        if not in_stage:
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
    
    # Create root module and add stage modules as submodules
    root_module = torch.nn.Module()
    submodule_list = []
    for stage_annotation_id, (stage_gm, placeholders, _, _, _, _, _, input_idxs, params) in stage_modules.items():
        module_name = f"stage_{stage_annotation_id}"
        root_module.add_module(module_name, stage_gm)
        submodule_list.append((stage_annotation_id, stage_gm, input_idxs, params, placeholders))
    
    new_gm = fx.GraphModule(root_module, new_graph)
    
    # Copy parameters, buffers, and modules from original module
    for name, param in gm.named_parameters(recurse=False):
        if name not in new_gm._parameters:
            new_gm.register_parameter(name, param)
    
    for name, buffer in gm.named_buffers(recurse=False):
        if name not in new_gm._buffers:
            new_gm.register_buffer(name, buffer)
    
    for name, module in gm.named_children():
        if name not in [f"stage_{stage_annotation_id}" for stage_annotation_id in stage_modules.keys()]:
            new_gm.add_module(name, module)
    
    new_gm.recompile()
    
    return new_gm, submodule_list



def _insert_comm_ops(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Transform a graph module by inserting communication operations on annotated nodes.
    
    This function finds nodes annotated with torch.fx.traceback.annotate containing
    "collective": "all_to_all_single" and inserts the corresponding communication
    operations into the graph.
    
    For each annotated node, it inserts:
    1. (Optional) Reshape input if "reshape": ("input", shape) is specified
    2. buf = torch.empty_like(annotated_tensor)
    3. torch.distributed.all_to_all_single(buf, annotated_tensor)  # Uses default process group
    4. (Optional) Reshape output if "reshape": ("output", shape) is specified
    5. Replaces all uses of annotated_tensor with the result
    
    The communication uses the default distributed process group (no group parameter specified).
    
    The reshape annotation format is: ("input"|"output", shape_tuple)
    where shape_tuple is a tuple of dimensions (can include -1 for inferred dimension).
    
    Args:
        gm: The graph module to transform
        
    Returns:
        A new graph module with communication operations inserted
    """
    # Find all nodes with collective communication annotations
    # We need to identify contiguous blocks of annotations, not just group by metadata
    # Each contiguous block should get its own communication operation
    
    # First, collect all annotated nodes with their metadata
    annotated_nodes = []
    node_list = list(gm.graph.nodes)
    for idx, node in enumerate(node_list):
        if 'custom' in node.meta:
            custom_meta = node.meta['custom']
            if isinstance(custom_meta, dict):
                collective = custom_meta.get('collective')
                if collective == 'all_to_all_single':
                    reshape = custom_meta.get('reshape')
                    annotation_key = tuple(sorted(custom_meta.items()))
                    annotated_nodes.append((idx, node, annotation_key, reshape))
    
    if not annotated_nodes:
        logger.info("No communication annotations found in graph")
        return gm
    
    # Group annotated nodes into contiguous blocks
    # A contiguous block is a sequence of nodes with the same annotation_key
    # that appear sequentially in the graph. Each block gets its own communication operation.
    # We split blocks when:
    # 1. The annotation_key changes (different annotation pattern)
    # 2. The same annotation_key appears again after being interrupted (new instance of same pattern)
    #    This handles the case where the same annotation appears in different layers
    annotation_blocks = []
    current_block = None
    MAX_GAP_FOR_CONTIGUOUS = 50  # Maximum number of non-annotated nodes allowed in a block
    
    for idx, node, annotation_key, reshape in annotated_nodes:
        if current_block is None:
            # Start a new block
            current_block = {
                'annotation_key': annotation_key,
                'nodes': [(node, reshape)],
                'last_idx': idx
            }
        elif current_block['annotation_key'] == annotation_key:
            # Same annotation - check if it's still part of the same contiguous block
            gap = idx - current_block['last_idx'] - 1
            if gap <= MAX_GAP_FOR_CONTIGUOUS:
                # Contiguous (or close enough), add to current block
                current_block['nodes'].append((node, reshape))
                current_block['last_idx'] = idx
            else:
                # Too large a gap - this is a new instance of the same annotation pattern
                # (e.g., a different layer). Finish current block and start new one.
                annotation_blocks.append(current_block)
                logger.debug(f"Split block due to gap of {gap} nodes for annotation {annotation_key}")
                current_block = {
                    'annotation_key': annotation_key,
                    'nodes': [(node, reshape)],
                    'last_idx': idx
                }
        else:
            # Different annotation, finish current block and start new one
            annotation_blocks.append(current_block)
            current_block = {
                'annotation_key': annotation_key,
                'nodes': [(node, reshape)],
                'last_idx': idx
            }
    
    # Don't forget the last block
    if current_block is not None:
        annotation_blocks.append(current_block)
    
    logger.info(f"Found {len(annotation_blocks)} contiguous annotation blocks")
    
    # For each contiguous block, find the output node (the one used outside the annotation)
    # This is the node that should have communication applied to it
    annotated_output_nodes = []
    for block_idx, block in enumerate(annotation_blocks):
        nodes_in_block = block['nodes']
        nodes_set = {node for node, _ in nodes_in_block}
        
        if len(nodes_in_block) == 1:
            # Single node, use it directly
            annotated_output_nodes.append(nodes_in_block[0])
        else:
            # Multiple nodes in the same block
            # Find the one that is used by nodes outside the annotation block
            # This is the output of the annotated block
            output_candidates = []
            for node, reshape in nodes_in_block:
                # Check if this node is used by any node not in the annotation block
                for user in node.users:
                    if user not in nodes_set:
                        output_candidates.append((node, reshape))
                        break
            
            if output_candidates:
                # Use the first output candidate (should be the output of the annotated block)
                annotated_output_nodes.append(output_candidates[0])
                logger.debug(f"Block {block_idx}: Using output node {output_candidates[0][0].name}")
            else:
                # No external users found, use the last node in topological order within the block
                # This happens if the annotated block's output isn't used yet
                last_node = max(nodes_in_block, key=lambda x: node_list.index(x[0]))
                annotated_output_nodes.append(last_node)
                logger.debug(f"Block {block_idx}: No external users found, using last node: {last_node[0].name}")
    
    logger.info(f"Found {len(annotated_output_nodes)} annotation blocks requiring communication operations")
    
    # Create a new graph to build the transformed version
    new_graph = fx.Graph()
    node_mapping = {}
    
    # Copy placeholders first
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
            node_mapping[node] = new_node
    
    # Copy all get_attr nodes (parameters) so they're available for use
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            if node not in node_mapping:
                new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
                node_mapping[node] = new_node
    
    # Process nodes in topological order
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # Already handled
            continue
        
        if node.op == "output":
            # Handle output separately at the end
            continue
        
        if node.op == "get_attr":
            # Already handled above
            continue
        
        # Check if this node needs communication inserted
        # This should be the output node of an annotated block
        needs_comm = False
        reshape_info = None
        for annotated_node, reshape in annotated_output_nodes:
            if node == annotated_node:
                needs_comm = True
                reshape_info = reshape
                break
        
        if needs_comm:
            # First, copy the node to get its output
            new_node = new_graph.node_copy(
                node,
                lambda n: node_mapping.get(n, n)
            )
            
            # Handle input reshape if specified
            # reshape_info format: ("input"|"output", shape_tuple)
            if reshape_info is not None:
                reshape_type, reshape_shape = reshape_info
                if reshape_type == "input":
                    # Reshape before communication
                    new_node = new_graph.call_function(
                        torch.reshape,
                        (new_node, reshape_shape)
                    )
                    logger.debug(f"Applied input reshape {reshape_shape} for node {node.name}")
            
            # Insert communication operations after this node
            # Following the exact pattern from the comments in mixtral.py:
            # buf = torch.empty_like(expert_inputs)
            # torch.distributed.all_to_all_single(buf, expert_inputs, group=device_mesh.get_group())
            # buf = buf.reshape(...)  # if output reshape
            # expert_inputs = buf
            
            # 1. Create buffer: buf = torch.empty_like(new_node)
            buf_node = new_graph.call_function(
                torch.empty_like,
                (new_node,)
            )
            
            # 2. Call all_to_all_single using custom autograd function
            # Use AllToAllSingleFunction to ensure gradients flow correctly
            # This wraps all_to_all_single with proper backward pass
            buf_node = new_graph.call_function(
                _dispatch_a2a_single,
                (buf_node, new_node)
            )
            # After all_to_all_single, buf_node contains the result
            # and maintains gradient connection from new_node through the custom Function
            
            # Handle output reshape if specified
            if reshape_info is not None:
                reshape_type, reshape_shape = reshape_info
                if reshape_type == "output":
                    # Reshape after communication
                    buf_node = new_graph.call_function(
                        torch.reshape,
                        (buf_node, reshape_shape)
                    )
                    logger.debug(f"Applied output reshape {reshape_shape} for node {node.name}")
            
            # 3. Replace all uses of the original node with the buffer
            # The buffer now contains the result after communication (and optional reshape)
            node_mapping[node] = buf_node
            
            logger.debug(f"Inserted all_to_all_single communication for node {node.name}")
        else:
            # Regular node, just copy it
            new_node = new_graph.node_copy(
                node,
                lambda n: node_mapping.get(n, n)
            )
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
    
    # Create new graph module with a fresh root module
    root_module = torch.nn.Module()
    
    # Copy parameters, buffers, and modules from original module
    for name, param in gm.named_parameters(recurse=False):
        root_module.register_parameter(name, param)
    
    for name, buffer in gm.named_buffers(recurse=False):
        root_module.register_buffer(name, buffer)
    
    for name, module in gm.named_children():
        root_module.add_module(name, module)
    
    new_gm = fx.GraphModule(root_module, new_graph)
    
    new_gm.recompile()
    
    return new_gm