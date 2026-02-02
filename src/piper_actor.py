import ray
import torch
import logging
import os
import time
import gc
from torch._guards import CompileId
from torch.nn import Parameter
from collections import defaultdict
import torch.distributed as dist
import threading
from typing import Callable

from .piper_utils import deserialize_graphmodule, create_logger, RemoteTensor, print_backward_graph, LOG_LEVEL

CLEANUP_MEMORY = False

logger = create_logger("piper_actor", LOG_LEVEL)

def create_actors(num_actors, optim_class, num_mbs, naive_gradient_sync=False):
    dp_rank = int(os.environ['PIPER_DP_RANK'])
    world_size = int(os.environ['PIPER_WORLD_SIZE'])
    dp_degree = int(os.environ['PIPER_DP_DEGREE'])
    pp_degree = int(os.environ['PIPER_PP_DEGREE'])

    from .piper_utils import piper_metadata
    for actor_id in range(num_actors):
        global_rank = dp_rank * dp_degree + actor_id
        actor = PiperActor.options(num_gpus=1, max_concurrency=3).remote(actor_id, optim_class, world_size, num_mbs, naive_gradient_sync, dp_rank=dp_rank, dp_degree=dp_degree, pp_degree=pp_degree)
        piper_metadata.actors[actor_id] = actor

    ray.get([actor.load_actor_handles.remote(piper_metadata.actors) for actor in piper_metadata.actors.values()])

def get_actor(actor_id):
    from .piper_utils import piper_metadata
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

        from .piper_utils import piper_metadata
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

        from .piper_utils import piper_metadata
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

@ray.remote
class PiperActor:
    def __init__(self, actor_id, optim_class, world_size, num_mbs, naive_gradient_sync=False, dp_rank=0, dp_degree=1, pp_degree=1):
        self.logger = create_logger("piper_actor", LOG_LEVEL)

        self.actor_id = actor_id
        self.optim_class = optim_class
        self.naive_gradient_sync = naive_gradient_sync
        
        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.pp_degree = pp_degree
        self.world_size = world_size

        self.num_mbs = num_mbs
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
        # map stage id -> data parallel communication operations
        self.comm_ops = dict()
        # map stage id -> tensor id -> comm op status
        self.comm_op_status = defaultdict(lambda: defaultdict(int))
        # map stage id -> tensor id -> comm op handle
        self.comm_op_handles = defaultdict(dict)
        # Timing infrastructure
        self.tracing = False  # Toggle for timing and memory tracing
        self.trace_data = {
            'update': {
                'total': [], 
                'sync': [], 
                'optim': [], 
                'peak_memory_delta': [], 
                'peak_memory': []
            }}

        from .piper_utils import piper_metadata
        piper_metadata.actor_self = self

    def load_actor_handles(self, actor_handles):
        from .piper_utils import piper_metadata
        piper_metadata.actors = actor_handles
    
    def id(self):
        return self.actor_id

    def send_input(self, tensor):
        self.input = tensor.to(self.device)
        return "done"
    
    def send_truth(self, tensor):
        self.truth = tensor.to(self.device)
        return "done"

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

    def prepare_comm_ops(self, stage_id, comm_ops, graphargs, ids):
        def hook_maker(tensor_id):
            def post_backward_hook(grad):
                self.comm_op_status[stage_id][tensor_id] += 1
                self.logger.debug(f"post_accumulate_grad_hook called: dp_rank: {self.dp_rank}, tensor_id={tensor_id}, shape={grad.shape}, status={self.comm_op_status[stage_id][tensor_id]}")
                return grad
            return post_backward_hook
        
        comm_op_ids = [op.tensor_id for op in comm_ops]
        for (t, i) in zip(graphargs, ids):
            if i in comm_op_ids:
               comm_op = comm_ops[comm_op_ids.index(i)]
               comm_op.tensor = t
               match comm_op.op:
                   case "allreduce":
                        if comm_op.pass_type == "backward" and comm_op.dep == "post":
                            tensor_id = i
                            tensor_shape = t.shape
                            self.logger.debug(f"Registering post_accumulate_grad_hook on dp_rank: {self.dp_rank}, tensor_id={tensor_id}, shape={tensor_shape}")
                            t.register_post_accumulate_grad_hook(hook_maker(tensor_id))
                        else:
                            raise ValueError(f"Unknown comm op type or dependency: {comm_op.op} {comm_op.dep}")
                   case _:
                       raise ValueError(f"Unknown communication operation: {comm_op.op}")

    def comm_loop(self, stage_id):
        comm_stream = torch.cuda.Stream()
        self.logger.info(f"Running comm loop for stage {stage_id} on actor {self.actor_id} global rank {self.global_rank}")
        for comm_op in self.comm_ops[stage_id]:
            done = False
            while not done:
                if comm_op.op == "allreduce" and comm_op.pass_type == "backward" and comm_op.dep == "post" and comm_op.group == "dp":
                    # perform the all reduce when all gradients have been accumulated
                    if self.comm_op_status[stage_id][comm_op.tensor_id] == self.num_mbs:
                        with torch.cuda.stream(comm_stream):
                            handle = dist.all_reduce(comm_op.tensor, op=dist.ReduceOp.AVG, group=self.dp_group, async_op=True)
                        # -1 indicates that the all reduce has been performed
                        self.comm_op_status[stage_id][comm_op.tensor_id] = 0
                        self.comm_op_handles[stage_id][comm_op.tensor_id] = handle
                        done = True
                else:
                    raise ValueError(f"Unknown comm op: {comm_op}")
                # time.sleep(0.001)

    def wait_for_comm_ops(self):
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} waiting for comm ops")
        for stage_id, comm_ops in self.comm_ops.items():
            for comm_op in comm_ops:
                done = False
                while not done:
                    if comm_op.tensor_id in self.comm_op_handles[stage_id]:
                        self.comm_op_handles[stage_id][comm_op.tensor_id].wait()
                        done = True
                    # time.sleep(0.001)

    def load_graph(self, stage_id: int, gm_data, comm_ops, graphargs, ids, input_idxs):
        self.logger.info(f"Loading stage {stage_id} graph on actor {self.actor_id} global rank {self.global_rank}")

        # set up tracing data structure
        if stage_id not in self.trace_data:
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
            }

        # compile the graph with the given graphargs
        gm = deserialize_graphmodule(gm_data)

        # Store GraphModule reference
        self.graph_modules[stage_id] = gm
        self.forward_fns[stage_id] = gm.forward

        # place parameters on the device
        def move_to_device(idx, arg):
            if idx not in input_idxs:
                return arg.to(self.device).detach().requires_grad_(True)
            else:
                return arg.to(self.device)
        params = graphargs
        params = list(map(move_to_device, range(len(params)), params))

        # discard the graphargs that correspond to input tensors
        for i in input_idxs:
            params[i] = None

        # prepare tensors with comm ops
        if not self.naive_gradient_sync:
            self.prepare_comm_ops(stage_id, comm_ops, params, ids)
        self.comm_ops[stage_id] = list(reversed(comm_ops))

        # save parameters
        self.input_idxs[stage_id] = input_idxs
        self.parameters[stage_id] = params

        # add the parameters to the optimizer for this stage
        all_params = [param for param in params if param is not None]
        if stage_id not in self.optims:
            self.optims[stage_id] = self.optim_class(all_params)
        else:
            self.optims[stage_id].add_param_group({'params': all_params})

        del gm_data


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
    
    @ray.method(tensor_transport="nccl")
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

        # save first input as input activation
        if args[0].dtype == torch.float:
            args[0].requires_grad_().retain_grad()
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
            # Calculate total forward time
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
        if args[0].dtype == torch.float:
            args[0].requires_grad_().retain_grad()
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

    @ray.method(tensor_transport="nccl")
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
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} synchronizing gradients")
        # Iterate over all stages on this actor and synchronize their parameters
        for stage_id, parameters in self.parameters.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.dp_group)

    def update(self, *done_mbs):
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            start_sync = torch.cuda.Event(enable_timing=True)
            end_sync = torch.cuda.Event(enable_timing=True)
            start_optim = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.reset_peak_memory_stats()
            update_start_memory = torch.cuda.memory_allocated()

            start_event.record()
        
        if self.tracing:
            start_sync.record()

        # if dp degree > 1, make sure all gradients are synchronized before optimizer step
        # TODO: this does not allow overlapping with the optimizer step
        if self.dp_degree > 1:
            if self.naive_gradient_sync:
                self.synchronize_gradients()
            else:
                self.wait_for_comm_ops()
        
        if self.tracing:
            end_sync.record()
            start_optim.record()
        
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
            sync_time = start_sync.elapsed_time(end_sync)
            optim_time = start_optim.elapsed_time(end_event)
            update_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - update_start_memory) / (1024**3)
            update_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

            self.trace_data['update']['total'].append(total_time)
            self.trace_data['update']['sync'].append(sync_time)
            self.trace_data['update']['optim'].append(optim_time)
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
            }
        self.trace_data['update'] = {
            'total': [],
            'sync': [],
            'optim': [],
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
