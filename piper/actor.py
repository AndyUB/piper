import ray
import torch
import logging
import os
import time
import threading
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union, Callable
import gc
from torch._guards import CompileId
from torch.nn import Parameter
from torch.autograd.graph import GradientEdge, Node
import torch.distributed as dist
from collections import defaultdict
from piper.utils import deserialize_graphmodule, piper_metadata, create_logger, RemoteTensor, print_backward_graph, LOG_LEVEL
from piper.backward_utils import get_param_groups, construct_reverse_graph, _get_grad_fn_or_grad_acc

torch.set_float32_matmul_precision('high')
torch._dynamo.config.compiled_autograd = False

CLEANUP_MEMORY = False
torch.autograd.set_detect_anomaly(True)

logger = create_logger("piper_actor", LOG_LEVEL)

def create_actors(num_actors, optim_class):
    dp_rank = int(os.environ['PIPER_DP_RANK'])
    world_size = int(os.environ['PIPER_WORLD_SIZE'])
    dp_degree = int(os.environ['PIPER_DP_DEGREE'])
    pp_degree = int(os.environ['PIPER_PP_DEGREE'])

    from piper.utils import piper_metadata
    for actor_id in range(num_actors):
        global_rank = dp_rank * dp_degree + actor_id
        actor = PiperActor.options(num_gpus=0.9, max_concurrency=2).remote(actor_id, optim_class, world_size, dp_rank=dp_rank, dp_degree=dp_degree, pp_degree=pp_degree)
        piper_metadata.actors[actor_id] = actor

    ray.get([actor.load_actor_handles.remote(piper_metadata.actors) for actor in piper_metadata.actors.values()])

def get_actor(actor_id):
    from piper.utils import piper_metadata
    return piper_metadata.actors[actor_id]

class ExpertRayFunction(torch.autograd.Function):
    """
    Custom autograd Function that dispatches expert calls to Ray actors.
    """
    
    @staticmethod
    def forward(ctx, global_expert_id: int, local_expert_id: int, batch_idx: int, pp_degree: int, *args):
        """
        Dispatch a forward call to the expert `global_expert_id` 
        on the actor `local_expert_id % pp_degree`. 
        Future work will allow custom actor placement. 
        """
        # Get the actor for this expert
        actor_id = local_expert_id % pp_degree
        actor = get_actor(actor_id)
        current_actor = ray.get_runtime_context().current_actor
        
        # Store metadata for backward pass
        ctx.global_expert_id = global_expert_id
        ctx.local_expert_id = local_expert_id
        ctx.batch_idx = batch_idx
        ctx.pp_degree = pp_degree
        ctx.actor_id = actor_id

        from piper.utils import piper_metadata
        actor_self = piper_metadata.actor_self
        
        # Store which inputs require gradients
        ctx.input_requires_grad = [arg.requires_grad if isinstance(arg, torch.Tensor) else False for arg in args]
        ctx.num_inputs = len(args)
        
        # Dispatch a remote call if actor is not current actor
        if actor == current_actor:
            output = actor_self.run_expert(global_expert_id, batch_idx, *args)
        else:
            ref = actor.run_expert.remote(global_expert_id, batch_idx, *args)
            output = ray.get(ref)

        # Detach the output so the graph is not carried over when 
        # the expert runs on the current actor. 
        if isinstance(output, (list, tuple)):
            output = [t.detach() for t in output]
        else:
            output = output.detach()
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Dispatch a backward expert call corresponding to the 
        forward expert call from the ctx. 
        """
        # Get the actor for this expert
        actor = get_actor(ctx.actor_id)
        current_actor = ray.get_runtime_context().current_actor
        logger.debug(f"Expert {ctx.global_expert_id} {ctx.batch_idx} backward on actor: {current_actor} calling actor: {actor}")
        num_inputs = ctx.num_inputs

        from piper.utils import piper_metadata
        actor_self = piper_metadata.actor_self
        
        # Dispatch a remote call if actor is not current actor
        if actor == current_actor:
            grad_inputs = actor_self.backward_expert(ctx.global_expert_id, ctx.batch_idx, grad_output)
        else:
            refs = actor.backward_expert.options(num_returns=num_inputs).remote(ctx.global_expert_id, ctx.batch_idx, grad_output)
            grad_inputs = ray.get(refs)
        
        # Return gradients for inputs (None for metadata arguments)
        result = [None, None, None, None] + list(grad_inputs)
        
        return tuple(result)

def dispatch_expert_ray(global_expert_id: int, local_expert_id: int, batch_idx: int, pp_degree: int, *args):
    """
    Dispatch a Ray remote call to run an expert GraphModule
    using a custom autograd Function. This function is called
    from the FX graph, so it needs to be allowed in the graph.
    """
    return ExpertRayFunction.apply(global_expert_id, local_expert_id, batch_idx, pp_degree, *args)

# Allow the dispatch function in the graph
torch.compiler.allow_in_graph(dispatch_expert_ray)

@ray.remote(enable_tensor_transport=True)
class PiperActor:
    def __init__(self, actor_id, optim_class, world_size, dp_rank=0, dp_degree=1, pp_degree=1):
        self.logger = create_logger("piper_actor", LOG_LEVEL)

        self.actor_id = actor_id
        self.optim_class = optim_class
        
        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.pp_degree = pp_degree
        self.world_size = world_size

        self.dp_group = None
        self.device = 'cuda'

        if dp_degree == 1:
            self.global_rank = actor_id
        elif pp_degree == 1:
            self.global_rank = dp_rank
        else:
            self.global_rank = dp_rank * dp_degree + actor_id

        self.logger.info(f"Initializing Ray actor {actor_id} global rank {self.global_rank} GPU {os.environ['CUDA_VISIBLE_DEVICES']}")

        self.input = None
        self.truth = None

        # map stage id -> compiled fx.Graph function
        self.forward_fns = dict()
        # map stage id -> original GraphModule (for hook registration)
        self.graph_modules = dict()
        # map stage id -> model parameters used by the fx.Graph with holes (None values) for input tensors
        self.parameters = dict()
        # map stage id -> indices of the input tensors (as opposed to model parameters) used by the fx.Graph
        self.input_idxs = dict()
        # map stage id -> optimizer for the fx.Graph
        self.optims = dict()
        # map stage id -> mb_idx -> previous activation (if this stage is not first)
        self.inp_activation = defaultdict(dict)
        # map stage id -> mb_idx -> current activation
        self.out_activation = defaultdict(dict)
        # accumuate loss for each microbatch
        self.loss = []
        # map expert id -> expert
        self.experts = dict()
        # map expert id -> expert parameters
        self.expert_parameters = dict()
        # map expert id -> expert input indices
        self.expert_input_idxs = dict()
        # map expert id -> expert module
        self.expert_modules = dict()
        # map expert id -> expert input activations
        self.expert_input_activations = defaultdict(dict)
        # map (expert_id, batch_idx) -> output activation for backward
        self.expert_output_activations = defaultdict(dict)
        # stage_id -> mb_idx -> param_groups
        self._bw_param_groups = defaultdict(dict)

        # Timing infrastructure
        self.tracing = False  # Toggle for timing and memory tracing
        self.trace_data = {'update': {'total': [], 'peak_memory_delta': [], 'peak_memory': []}}

        from piper.utils import piper_metadata
        piper_metadata.actor_self = self

    def load_actor_handles(self, actor_handles):
        from piper.utils import piper_metadata
        piper_metadata.actors = actor_handles
    
    def id(self):
        return self.actor_id

    def send_input(self, tensor):
        self.input = tensor.to(self.device)
        return "done"
    
    def send_truth(self, tensor):
        self.truth = tensor.to(self.device)
        return "done"

    def get_peak_memory(self):
        return torch.cuda.max_memory_allocated() / (1024**3)

    # @ray.method
    def join_process_groups(self):
        master_addr = os.environ.get('PIPER_MASTER_ADDR', "127.0.0.1")
        master_port = os.environ.get('PIPER_MASTER_PORT', "10000")
        init_method = f"tcp://{master_addr}:{master_port}"

        dist.init_process_group("nccl", init_method=init_method, rank=self.global_rank, world_size=self.world_size)
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} has GPU {os.environ['CUDA_VISIBLE_DEVICES']}, joined the global process group")

        if self.dp_degree > 1:
            self.join_dp_process_group()
    
    def join_dp_process_group(self):
        # Every process needs to participate in every subgroup creation
        num_dp_groups = self.world_size // self.dp_degree
        for dp_group_id in range(num_dp_groups):
            group_ranks = [(dp_group_id + num_dp_groups * i) for i in range(self.dp_degree)]
            process_group = dist.new_group(ranks=group_ranks, backend='nccl')
            if self.global_rank % num_dp_groups == dp_group_id:
                self.dp_group = process_group
                self.logger.info(f"Global rank {self.global_rank} joined its dp group {dp_group_id} along with ranks {group_ranks}")

    def load_graph(self, stage_id: int, gm_data, compiler_fn, graphargs, input_idxs):
        self.logger.debug(f"Compiling graph on actor {self.actor_id} for stage id: {stage_id} with inputs: {len(graphargs)} and input indices: {input_idxs}")

        # set up tracing data structure
        self.trace_data[stage_id] = {
            'forward': {
                'forward': [],
                'total': [],
                'peak_memory_delta': [],
                'peak_memory': []
            },
            'backward': {
                'backward': [],
                'total': [],
                'peak_memory_delta': [],
                'peak_memory': []
            },
            'backward_input': {
                'time': [],
                'graph_pruning': [],
            },
            'backward_weight': {
                'time': [],
            },
        }

        # compile the graph with the given graphargs
        gm = deserialize_graphmodule(gm_data)

        # Store GraphModule reference
        self.graph_modules[stage_id] = gm
        self.forward_fns[stage_id] = gm.forward
        
        # save the parameters and initialize the optimizer
        self.add_param_group(stage_id, graphargs, input_idxs)

        del gm_data

    def add_param_group(self, stage_id: int, params, input_idxs):
        self.logger.debug(f"Adding param group for stage {stage_id}")

        if stage_id in self.parameters:
            self.logger.debug(f"Param group already exists for stage {stage_id}")
            return
        
        # place parameters on the device
        def move_to_device(idx, arg):
            if idx not in input_idxs:
                return arg.to(self.device).detach().requires_grad_(True)
            else:
                return arg.to(self.device)
        params = list(map(move_to_device, range(len(params)), params))

        # discard the graphargs that correspond to input tensors
        for i in input_idxs:
            params[i] = None

        # save the parameters
        self.input_idxs[stage_id] = input_idxs
        self.parameters[stage_id] = params

        # add the parameters to the optimizer for this stage
        if stage_id not in self.optims:
            self.optims[stage_id] = self.optim_class([param for param in params if param is not None])
        else:
            self.optims[stage_id].add_param_group({'params': [param for param in params if param is not None]})

    def load_expert(self, stage_id, expert_id, local_expert_id, batch_idx, expert_module, input_idxs, param_idxs, params):
        self.logger.debug(f"Loading expert {expert_id} on actor {self.actor_id} global rank {self.global_rank}")
        self.experts[expert_id] = expert_module

        # place parameters on the device
        def move_to_device(idx, arg):
            if arg is None:
                return None
            if idx not in input_idxs:
                return arg.to(self.device).detach().requires_grad_(True)
            else:
                return arg.to(self.device)
        params = list(map(move_to_device, range(len(params)), params))

        self.expert_input_idxs[expert_id] = input_idxs
        self.expert_parameters[expert_id] = params
        self.expert_modules[expert_id] = expert_module

        # add the parameters to the optimizer for this stage
        if stage_id not in self.optims:
            self.optims[stage_id] = self.optim_class([param for param in params if param is not None])
        else:
            self.optims[stage_id].add_param_group({'params': [param for param in params if param is not None]})

    def run_expert(self, expert_id, batch_idx, *args):
        logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} forward expert {expert_id} batch element {batch_idx}")

        # Place args on the device and detach existing grad_fn
        def place(arg):
            if isinstance(arg, torch.Tensor):
                if arg.device == torch.device('cpu'):
                    out = arg.to(self.device)
                else:
                    out = arg
                if out.grad_fn is not None:
                    out = out.detach().requires_grad_(True)
            return out
        args = list(map(place, args))

        # place input tensors in the correct indices
        for i, arg in zip(self.expert_input_idxs[expert_id], args):
            self.expert_parameters[expert_id][i] = arg

        # save input activations
        self.expert_input_activations[expert_id][batch_idx] = args
        
        # run expert module, track grad fn
        with torch.set_grad_enabled(True):
            out = self.expert_modules[expert_id](*self.expert_parameters[expert_id])
        
        # save output activation
        if isinstance(out, (list, tuple)):
            self.expert_output_activations[expert_id][batch_idx] = out[0]
        else:
            self.expert_output_activations[expert_id][batch_idx] = out
        
        # clear the input tensors
        for i in self.expert_input_idxs[expert_id]:
            self.expert_parameters[expert_id][i] = None
        del args

        return out
    
    def backward_expert(self, expert_id, batch_idx, grad_output):
        logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} backward expert {expert_id} batch element {batch_idx}")

        if expert_id not in self.expert_output_activations or batch_idx not in self.expert_output_activations[expert_id]:
            raise ValueError(f"No output activation found for expert {expert_id} batch {batch_idx}")
        
        # Get the stored output activation
        output_activation = self.expert_output_activations[expert_id][batch_idx]
        
        self.logger.debug(f"Expert {expert_id} backward graph:")
        print_backward_graph(self.logger.debug, output_activation)

        if isinstance(grad_output, torch.Tensor) and grad_output.device != self.device:
            grad_output = grad_output.to(self.device)

        output_activation.backward(gradient=grad_output)

        # Get the gradients for the input activations
        input_activations = self.expert_input_activations[expert_id][batch_idx]
        grad_inputs = []
        for i, input_activation in enumerate(input_activations):
            if input_activation.requires_grad:
                grad_inputs.append(input_activation.grad)
            else:
                grad_inputs.append(None)
        
        # Clean up
        del output_activation
        del input_activations
        
        return grad_inputs
    
    # @ray.method(tensor_transport="nccl")
    def forward(self, stage_id: int, mb_idx: int, *args):
        self.logger.debug(f"Calling forward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

        if self.tracing:
            beginning_event = torch.cuda.Event(enable_timing=True)
            forward_start_event = torch.cuda.Event(enable_timing=True)
            forward_end_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            beginning_event.record()
        
        def pre_loaded_input(param):
            if param is None:
                return self.input
            else:
                return param
        args = list(map(pre_loaded_input, args))

        # Ray object refs resolve to a single element list
        def unwrap(x):
            if isinstance(x, list) or isinstance(x, tuple):
                assert len(x) == 1
            return x[0] if isinstance(x, list) or isinstance(x, tuple) else x
        args = list(map(unwrap, args))

        def place(arg):
            if arg.device == torch.device('cpu'):
                return arg.to(self.device)
            else:
                return arg
        args = list(map(place, args))

        # place input tensors in the correct indices
        for i, arg in zip(self.input_idxs[stage_id], args):
            self.parameters[stage_id][i] = arg

        # save first input as previous activation
        # if args[0].dtype == torch.float:
        #     args[0].requires_grad_().retain_grad()
        # self.inp_activation[stage_id][mb_idx] = args[0]
        if stage_id != 0:
            if not args[0].requires_grad:
                args[0].requires_grad_()
            self.inp_activation[stage_id][mb_idx] = args[0]

        # Record start event for forward timing
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            forward_start_memory = torch.cuda.memory_allocated()
            forward_start_event.record()

        # Call compiled function
        output = self.forward_fns[stage_id](*self.parameters[stage_id])
        assert isinstance(output, (list, tuple)) and len(output) == 1, "Piper only supports one output per subgraph"
        output = output[0]

        # Record end event and calculate forward timing
        if self.tracing:
            forward_end_event.record()
            torch.cuda.synchronize()
            forward_time = forward_start_event.elapsed_time(forward_end_event)
            forward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - forward_start_memory) / (1024**3)
            forward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # save output as output activation
        self.logger.debug(f"Saving output activation {output.shape} for stage {stage_id} mb {mb_idx}")
        self.out_activation[stage_id][mb_idx] = output

        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.parameters[stage_id][i] = None
        del args

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = forward_start_event.elapsed_time(end_event)
            self.trace_data[stage_id]['forward']['forward'].append(forward_time)
            self.trace_data[stage_id]['forward']['total'].append(forward_time)
            self.trace_data[stage_id]['forward']['peak_memory_delta'].append(forward_peak_memory_delta_gb)
            self.trace_data[stage_id]['forward']['peak_memory'].append(forward_peak_memory_gb)

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return output

    def forward_cpu(self, stage_id: int, mb_idx: int, *args):
        self.logger.debug(f"Calling cpu forward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

        def pre_loaded_input(param):
            if param is None:
                return self.input
            else:
                return param
        args = list(map(pre_loaded_input, args))

        # Ray object refs resolve to a single element list
        def unwrap(x):
            if isinstance(x, list) or isinstance(x, tuple):
                assert len(x) == 1
                return x[0]
            else:
                return x
        args = list(map(unwrap, args))
        
        def place(arg):
            if arg.device == torch.device('cpu'):
                return arg.to(self.device)
            else:
                return arg
        args = list(map(place, args))

        # place input tensors in the correct indices
        for i, arg in zip(self.input_idxs[stage_id], args):
            self.parameters[stage_id][i] = arg

        # save first input as input activation
        # if args[0].dtype == torch.float:
        #     args[0].requires_grad_().retain_grad()
        # self.inp_activation[stage_id][mb_idx] = args[0]
        if stage_id != 0:
            if not args[0].requires_grad:
                args[0].requires_grad_()
            self.inp_activation[stage_id][mb_idx] = args[0]

        out = self.forward_fns[stage_id](*self.parameters[stage_id])

        # save output as output activation
        activation_tensor = out[0] if isinstance(out, (list, tuple)) else out
        self.out_activation[stage_id][mb_idx] = activation_tensor
        
        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.parameters[stage_id][i] = None
        del args
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return out

    # @ray.method(tensor_transport="nccl")
    def backward(self, stage_id: int, mb_idx: int, inp, truth=None, loss_fn=None):
        self.logger.debug(f"Calling backward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

        if self.tracing:
            beginning_event = torch.cuda.Event(enable_timing=True)
            backward_start_event = torch.cuda.Event(enable_timing=True)
            backward_end_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            beginning_event.record()
            torch.cuda.reset_peak_memory_stats()
            backward_start_memory = torch.cuda.memory_allocated()
            backward_start_event.record()

        if isinstance(inp, list) or isinstance(inp, tuple):
            assert len(inp) == 1
            inp = inp[0]

        # Get the activations for this stage and microbatch
        out_activation = self.out_activation[stage_id][mb_idx]
        inp_activation = self.inp_activation[stage_id][mb_idx]
        self.logger.debug(f"Stage {stage_id} backward graph:")
        print_backward_graph(self.logger.debug, out_activation)

        # compute loss with the final activation of the final stage. 
        # use the saved activation rather than inp because the saved activation stores the computation graph
        if loss_fn is not None:
            assert out_activation.shape == inp.shape
            if self.truth is not None:
                labels = self.truth
            else:
                labels = truth.to(self.device)
            assert out_activation.shape == labels.shape
            loss = loss_fn(out_activation, labels)
            loss.backward()
            self.loss.append(loss.item())
        # if not the last stage, backprop on the stored activation given 
        # the input gradient from the subsequent stage
        else:
            assert inp is not None
            assert out_activation.shape == inp.shape
            out_activation.backward(gradient=inp)

        del out_activation

        # propagate the gradient backwards if not the first stage
        if stage_id == 0:
            ret = "done"
        else:
            ret = inp_activation.grad
        
        del inp_activation

        # Record end event and calculate backward timing
        if self.tracing:
            backward_end_event.record()
            torch.cuda.synchronize()
            
            # Calculate total backward time
            backward_time = backward_start_event.elapsed_time(backward_end_event)
            
            backward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - backward_start_memory) / (1024**3)
            backward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

            end_event.record()
            torch.cuda.synchronize()
            total_time = beginning_event.elapsed_time(end_event)
            self.trace_data[stage_id]['backward']['backward'].append(backward_time)
            self.trace_data[stage_id]['backward']['total'].append(total_time)
            self.trace_data[stage_id]['backward']['peak_memory_delta'].append(backward_peak_memory_delta_gb)
            self.trace_data[stage_id]['backward']['peak_memory'].append(backward_peak_memory_gb)
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return [ret] * 2

    def synchronize_gradients(self):
        """Synchronize gradients across all DP ranks for this stage using all-reduce."""     
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} synchronizing gradients")
        
        # Iterate over all stages on this actor and synchronize their parameters
        for stage_id, parameters in self.parameters.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.dp_group)
        self.logger.info(f"Actor {self.actor_id} global rank {self.global_rank} finished synchronizing gradients")

    def update(self, *done_mbs):
        self.logger.debug(f"Calling update on actor {self.actor_id} global rank {self.global_rank}")
        
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.reset_peak_memory_stats()
            update_start_memory = torch.cuda.memory_allocated()

            start_event.record()
        
        # if dp degree > 1, synchronize the gradients
        if self.dp_degree > 1:
            self.synchronize_gradients()
        
        # step the optimizer for each stage
        for _, optim in self.optims.items():
            optim.step()
            optim.zero_grad()
        losses = self.loss
        self.loss.clear()

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event)
            update_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - update_start_memory) / (1024**3)
            update_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

            self.trace_data['update']['total'].append(total_time)
            self.trace_data['update']['peak_memory_delta'].append(update_peak_memory_delta_gb)
            self.trace_data['update']['peak_memory'].append(update_peak_memory_gb)

        return losses

    def clear_trace_data(self) -> None:
        """
        Clear all collected timing data.
        """
        for stage_id in self.trace_data:
            self.trace_data[stage_id] = {
                'forward': {
                    'forward': [],
                    'total': [],
                    'peak_memory_delta': [],
                    'peak_memory': []
                },
                'backward': {
                    'backward': [],
                    'total': [],
                    'peak_memory_delta': [],
                    'peak_memory': []
                },
                'backward_input': {
                    'time': [],
                    'graph_pruning': [],
                },
                'backward_weight': {
                    'time': [],
                },
            }
        self.trace_data['update'] = {
            'total': [],
            'peak_memory_delta': [],
            'peak_memory': []
        }
    
    def get_trace_data(self) -> dict:
        return self.trace_data

    def set_tracing(self, enabled: bool) -> None:
        """
        Enable or disable timing and memory tracing.
        
        Args:
            enabled (bool): True to enable tracing, False to disable.
        """
        self.tracing = enabled
        self.logger.info(f"Actor {self.actor_id}: Tracing {'enabled' if enabled else 'disabled'}")

    def start_mem_tracing(self) -> None:
        torch.cuda.memory._record_memory_history()
        return "done"
    
    def stop_mem_tracing(self) -> None:
        torch.cuda.memory._dump_snapshot(f"actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        self.logger.info(f"Saved memory snapshot to actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return "done"

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()
        return "done"

    def get_peak_memory(self):
        return torch.cuda.max_memory_allocated() / (1024**3)
    
    # @ray.method(tensor_transport="nccl")
    def get_object(self, is_fwd, stage_id):
        if is_fwd:
            return self.fwd_objs[stage_id]
        else:
            return self.bwd_objs[stage_id]

    def time_object_retrieval(self, obj) -> dict:
        objs = []
        if isinstance(obj, tuple) or isinstance(obj, list):
            objs = [t + 1 for t in obj]
        else:
            objs = [obj + 1]
        self.objs = objs
        return "done"
    
    # @ray.method(tensor_transport="nccl")
    def backward_input(self, stage_id: int, mb_idx: int, inp, truth=None, loss_fn=None):
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        is_last_stage = loss_fn is not None
        activation = self.out_activation[stage_id][mb_idx]

        # Use loss instead of activation if this is the backward_input call for the last stage
        if is_last_stage:
            labels = self.truth if self.truth is not None else truth.to(self.device)
            loss = loss_fn(activation, labels)
            self.loss.append(loss)

            activation_or_loss = loss
            upstream_grads = torch.ones_like(loss)

        else:
            activation_or_loss = activation
            upstream_grads = inp

        # In first stage, we treat the backward_input call like a NOOP
        if stage_id == 0:
            gx = torch.zeros_like(activation) # May get slight performance boost if we don't send the full ones tensor
            if self.tracing:
                end_event.record()
                torch.cuda.synchronize()
                total_time = start_event.elapsed_time(end_event)
                self.trace_data[stage_id]['backward_input']['time'].append(total_time)
            return [gx, upstream_grads]

        stage_input = self.inp_activation[stage_id][mb_idx]
        stage_params = [p for p in self.parameters[stage_id] if p is not None and p.requires_grad]

        # Lists of autograd nodes for the layer outputs, inputs, and parameters
        # Tracing for graph pruning computation time
        if self.tracing:
            graph_pruning_start = torch.cuda.Event(enable_timing=True)
            graph_pruning_end = torch.cuda.Event(enable_timing=True)
            graph_pruning_start.record()
        
        output_nodes = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [activation_or_loss]) if n is not None]
        input_nodes  = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [stage_input]) if n is not None]
        param_nodes   = [n for n in (_get_grad_fn_or_grad_acc(p) for p in stage_params) if n is not None]

        # Use the autograd graph with edges reversed to compute parameter groups, which are groups 
        # of parameters that share the same intermediate nodes. Intermediate nodes are the nodes that
        # lie on both (1) a backward path from the output node(s) to the stage input nodes and 
        # (2) in a path from the output node(s) a parameter node/gradient accumulator
        reverse_edges = construct_reverse_graph(output_nodes)
        param_groups = get_param_groups(input_nodes, param_nodes, reverse_edges)

        pgs = param_groups
        with_int = [pg for pg in pgs if pg.get("intermediates")]
        self.logger.debug(
            f"stage {stage_id} mb {mb_idx}: groups={len(pgs)} with_intermediates={len(with_int)} "
            f"intermediates_sizes={[len(pg.get('intermediates', [])) for pg in with_int]}"
        )

        def ancestors(start_nodes, reverse_edges):
            # reverse_edges: dict[node] -> list[node] giving "incoming" edges when traversing backward
            seen = set()
            stack = list(start_nodes)
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                for p in reverse_edges.get(n, []):
                    stack.append(p)
            return seen

        # build ancestor sets for a handful of groups
        anc_sets = []
        for pg in param_groups[:10]:  # sample first 10 to keep it cheap
            ints = pg.get("intermediates", [])
            anc_sets.append(ancestors(ints, reverse_edges))

        shared = 0
        for i in range(len(anc_sets)):
            for j in range(i+1, len(anc_sets)):
                shared += len(anc_sets[i].intersection(anc_sets[j])) > 0

        self.logger.info(f"stage {stage_id} mb {mb_idx}: sampled_pairs_with_shared_ancestors={shared}")

        # Hooks to capture grads at intermediate nodes. In backward_weight,
        # we'll backprop from these intermediate values
        handles = []
        for pg in param_groups:
            intermediates = pg["intermediates"]
            if not intermediates:
                continue

            pg["grads"] = [None] * len(intermediates)

            for i, intermediate_node in enumerate(intermediates):
                def make_hook(group: Dict[str, Any], idx: int):
                    def hook(grad_inputs):
                        group["grads"][idx] = grad_inputs
                    return hook
                handles.append(intermediate_node.register_prehook(make_hook(pg, i)))

        if self.tracing:
            graph_pruning_end.record()
            torch.cuda.synchronize()
            graph_pruning_time = graph_pruning_start.elapsed_time(graph_pruning_end)
            if 'graph_pruning' not in self.trace_data[stage_id]['backward_input']:
                self.trace_data[stage_id]['backward_input']['graph_pruning'] = []
            self.trace_data[stage_id]['backward_input']['graph_pruning'].append(graph_pruning_time)

        gx = torch.autograd.grad(
            outputs=activation_or_loss,
            inputs=stage_input,
            grad_outputs=upstream_grads,
            retain_graph=True,
            allow_unused=True,
        )

        gx = gx[0] # Take the gx gradient out of the tuple returned by autograd.grad
        if gx is not None and stage_input.requires_grad:
            if stage_input.grad is None:
                stage_input.grad = gx
            else:
                stage_input.grad.add_(gx)

        # Free output tensors between the output nodes and the intermediate nodes
        if not isinstance(activation_or_loss, list):
            activation_or_loss = [activation_or_loss]

        for t in activation_or_loss:
            t.detach_()

        for h in handles:
            h.remove()

        # Save parameter groups for use in backward_weight
        self._bw_param_groups[stage_id][mb_idx] = param_groups

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event)
            self.trace_data[stage_id]['backward_input']['time'].append(total_time)

        return [gx, upstream_grads]

    # @ray.method(tensor_transport="nccl")
    def backward_weight(self, stage_id: int, mb_idx: int, gx, upstream, truth=None, loss_fn=None):
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        stage_params = [p for p in self.parameters[stage_id] if p is not None and p.requires_grad]

        # We only pass to enforce temporal dependency.
        # upstream is only required for the stage 0 case
        gx = gx
        upstream = upstream

        # Special case to handle stage 0 since backward_input is a NOOP, 
        # meaning no parameter groups are created
        if stage_id == 0:
            is_last_stage = loss_fn is not None
            activation = self.out_activation[stage_id][mb_idx]

            if is_last_stage:
                labels = self.truth if self.truth is not None else truth.to(self.device)
                loss = loss_fn(activation, labels)

                gparams = torch.autograd.grad(
                    outputs=loss,
                    inputs=stage_params,
                    retain_graph=False,
                )

            else:
                gparams = torch.autograd.grad(
                    outputs=activation,
                    inputs=stage_params,
                    grad_outputs=upstream,
                    retain_graph=False,
                )

            assert len(gparams) == len(stage_params), (
                f"Stage {stage_id}: mismatch #param grads {len(gparams)} vs params {len(stage_params)}"
            )
            
            for p, pg in zip(stage_params, gparams):
                if p.grad is None:
                    p.grad = pg.clone()
                else:
                    p.grad.add_(pg)

            del self.out_activation[stage_id][mb_idx]

            if self.tracing:
                end_event.record()
                torch.cuda.synchronize()
                total_time = start_event.elapsed_time(end_event)
                self.trace_data[stage_id]['backward_weight']['time'].append(total_time)

            return [None] + ["done"]

        # Create mapping from autograd nodes -> parameters
        grad_acc_to_weight: Dict[Node, Tuple[Parameter, int]] = {}
        for param in stage_params:
            node = _get_grad_fn_or_grad_acc(param)
            grad_acc_to_weight[node] = param

        param_groups = self._bw_param_groups[stage_id][mb_idx]

        pgs = param_groups
        with_int = [pg for pg in pgs if pg.get("intermediates")]
        self.logger.info(
            f"stage {stage_id} mb {mb_idx}: groups={len(pgs)} with_intermediates={len(with_int)} "
            f"intermediates_sizes={[len(pg.get('intermediates', [])) for pg in with_int]}"
        )

        seen = set()
        overlap = 0
        for pg in param_groups:
            for n in pg.get("intermediates", []) or []:
                k = id(n)
                overlap += (k in seen)
                seen.add(k)
        self.logger.info(f"stage {stage_id} mb {mb_idx}: intermediate_overlaps={overlap}")

        # Perform the weight updates separately for each param_group, beginning
        # backprop from each the intermediate node(s) of each group
        for pg in param_groups:
            intermediates: List[Node] = pg.get("intermediates", [])
            intermediate_grads = pg.get("grads", None) # List of intermediate node gradients, captured by the hooks

            # Skip groups without intermediate nodes (could happen in weird cases
            # where one node is disconnected from the rest of the autograd graph for some reason)
            if not intermediates or intermediate_grads is None:
                continue

            intermediate_edges: List[GradientEdge] = []
            intermediate_edge_grads: List[torch.Tensor] = []

            for intermediate_node, grad_inputs in zip(intermediates, intermediate_grads):
                if grad_inputs is None:
                    continue

                gs = [x for x in grad_inputs if x is not None]
                if not gs:
                    continue
                
                # Sum all gradients arriving at the current intermediate node
                # in case the node has multiple source of gradients
                summed = sum(gs)

                # Create a GradientEdge for each intermediate node (we can backprop with respect to these)
                # and store the summed gradient for that node
                intermediate_edges.append(GradientEdge(intermediate_node, 0))
                intermediate_edge_grads.append(summed)

            del pg["intermediates"]

            if not intermediate_edges:
                continue

            # Grab params for the param_nodes in this param group using our grad_acc_to_weight map from earlier
            mapped_param_nodes = [p for p in pg["params"] if p in grad_acc_to_weight]
            if not mapped_param_nodes:
                continue

            # Use these parameters to create a GradientEdge that we'll use as our input to autograd.grad
            weight_edges = tuple(GradientEdge(p, 0) for p in mapped_param_nodes)

            gparams = torch.autograd.grad(
                outputs=intermediate_edges,
                inputs=weight_edges,
                grad_outputs=intermediate_edge_grads,
                retain_graph=True,
            )

            del pg["grads"]

            assert len(gparams) == len(mapped_param_nodes), (
                f"Stage {stage_id}: mismatch #param grads {len(gparams)} vs params {len(mapped_param_nodes)}"
            )
            
            # Finally, update gradients for the params in this param_group
            for param_node, dw in zip(pg["params"], gparams):
                if dw is None:
                    continue

                weight = grad_acc_to_weight[param_node]

                if weight.grad is None:
                    weight.grad = dw
                else:
                    weight.grad.add_(dw)

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event)
            self.trace_data[stage_id]['backward_weight']['time'].append(total_time)

        return [None] + ["done"]
