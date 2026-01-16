import ray
import torch
import logging
import os
import time
import gc
from torch._guards import CompileId
from torch.nn import Parameter
from collections import defaultdict
from .piper_utils import deserialize_graphmodule, piper_metadata, create_logger
import torch.distributed as dist
from typing import Callable

CLEANUP_MEMORY = False

def create_actor(actor_id, optim_class):
    dp_rank = int(os.environ['PIPER_DP_RANK'])
    world_size = int(os.environ['PIPER_WORLD_SIZE'])
    dp_degree = int(os.environ['PIPER_DP_DEGREE'])
    pp_degree = int(os.environ['PIPER_PP_DEGREE'])
    global_rank = dp_rank * dp_degree + actor_id
    actor = PiperActor.options(num_gpus=1).remote(actor_id, optim_class, world_size, dp_rank=dp_rank, dp_degree=dp_degree, pp_degree=pp_degree)
    piper_metadata.actors[actor_id] = actor

def get_actor(actor_id):
    return piper_metadata.actors[actor_id]

# Function to print the full backward graph of a tensor
def print_backward_graph(tensor, prefix=""):
    seen = set()
    def _print(t, indent=0):
        fn = t.grad_fn if hasattr(t, 'grad_fn') and t.grad_fn is not None else None
        if fn is None:
            print(" " * indent + f"{prefix}Tensor: no grad_fn")
            return
        if fn in seen:
            print(" " * indent + f"{prefix}{type(fn).__name__} (recursive/ref)")
            return
        seen.add(fn)
        print(" " * indent + f"{prefix}{type(fn).__name__}")
        for next_fn, _ in fn.next_functions:
            if next_fn is not None and hasattr(next_fn, 'variable'):
                print(" " * (indent + 2) + f"{prefix}Variable: {type(next_fn.variable).__name__}")
            elif next_fn is not None:
                _print(type('Dummy', (), {'grad_fn': next_fn})(), indent + 2)
            else:
                print(" " * (indent + 2) + f"{prefix}None")
    _print(tensor, 0)

@ray.remote
class PiperActor:
    def __init__(self, actor_id, optim_class, world_size, dp_rank=0, dp_degree=1, pp_degree=1):
        self.logger = create_logger("piper_actor", "DEBUG")

        self.actor_id = actor_id
        self.optim_class = optim_class
        
        # Data parallel attributes
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

        self.logger.info(f"Initializing Ray actor {actor_id} global rank {self.global_rank} with PID: {os.getpid()} and has GPU {os.environ['CUDA_VISIBLE_DEVICES']}")

        self.input = None
        self.truth = None
        self.fwd_objs = {}
        self.bwd_objs = {}

        # map stage id -> subgraph id -> compiled fx.Graph function
        self.compiled_fns = defaultdict(lambda: defaultdict(dict))
        # map stage id -> subgraph id -> original GraphModule (for hook registration)
        self.graph_modules = defaultdict(lambda: defaultdict(dict))
        # map stage id -> list of subgraph ids in the order they are called
        self.subgraph_order = defaultdict(list)
        # map stage id -> subgraph id -> model parameters used by the fx.Graph with holes (None values) for input tensors
        self.parameters = defaultdict(lambda: defaultdict(dict))
        # map stage id -> subgraph id -> indices of the input tensors (as opposed to model parameters) used by the fx.Graph
        self.input_idxs = defaultdict(lambda: defaultdict(dict))
        # map stage id -> optimizer for the fx.Graph
        self.optims = dict()
        # map stage id -> mb_idx -> previous activation (if this stage is not first)
        self.inp_activation = defaultdict(lambda: defaultdict(dict))
        # map stage id -> mb_idx -> current activation
        self.out_activation = defaultdict(lambda: defaultdict(dict))
        # accumuate loss for each microbatch
        self.loss = []
        # map expert id -> expert
        self.experts = dict()

        # Timing infrastructure
        self.tracing = False  # Toggle for timing and memory tracing
        self.trace_data = {'update': {'total': [], 'peak_memory_delta': [], 'peak_memory': []}}

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

    def get_max_subgraph_id(self, stage_id: int) -> int:
        """Get the maximum subgraph_id for a given stage_id."""
        if stage_id not in self.compiled_fns:
            return 0
        subgraph_ids = list(self.compiled_fns[stage_id].keys())
        return max(subgraph_ids) if subgraph_ids else 0

    def compile_graph(self, stage_id: int, subgraph_id: int, gm_data, compiler_fn, graphargs, input_idxs):
        self.logger.debug(f"Compiling graph on actor {self.actor_id} for stage id: {stage_id} subgraph id: {subgraph_id} with inputs: {len(graphargs)} and input indices: {input_idxs}")

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
        self.graph_modules[stage_id][subgraph_id] = gm

        # Store subgraph in list of subgraph ids for this stage
        self.subgraph_order[stage_id].append(subgraph_id)
        
        # Place graphargs on the device
        def move_to_device(idx, arg):
            if idx not in input_idxs:
                return arg.to(self.device).detach().requires_grad_(True)
            else:
                return arg.to(self.device)
        graphargs = list(map(move_to_device, range(len(graphargs)), graphargs))

        compiled_fn = compiler_fn(gm, graphargs)
        assert callable(compiled_fn), "compiler_fn did not return callable"
        self.compiled_fns[stage_id][subgraph_id] = compiled_fn

        # discard the graphargs that correspond to input tensors
        for i in input_idxs:
            graphargs[i] = None
        self.input_idxs[stage_id][subgraph_id] = input_idxs

        # save the graphargs that correspond to model parameters
        self.parameters[stage_id][subgraph_id] = graphargs

        # initialize the optimizer for this stage
        if stage_id not in self.optims:
            self.optims[stage_id] = self.optim_class([param for param in self.parameters[stage_id][subgraph_id] if param is not None])
        else:
            self.optims[stage_id].add_param_group({'params': [param for param in self.parameters[stage_id][subgraph_id] if param is not None]})

        del gm_data

        return "Finished compiling"

    def load_expert(self, expert_class, uid, args, kwargs):
        self.logger.info(f"Loading expert {expert_class} id {uid} on actor {self.actor_id}")
        kwargs = kwargs or {}
        self.experts[uid] = expert_class(*args, **kwargs).to(self.device)

    def forward_expert(self, uid, *args):
        with self.expert_mutexes[uid]:
            return self.experts[uid](*args)

    @ray.method(tensor_transport="nccl")
    def forward(self, stage_id: int, subgraph_id: int, mb_idx: int, *args):
        self.logger.debug(f"Calling forward {stage_id} subgraph {subgraph_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

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
            return arg.to(self.device)
        args = list(map(place, args))

        # place input tensors in the correct indices
        for i, arg in zip(self.input_idxs[stage_id][subgraph_id], args):
            self.parameters[stage_id][subgraph_id][i] = arg

        # save first input to the first subgraph as input activation
        if stage_id != 0 and subgraph_id == self.subgraph_order[stage_id][0]:
            if not args[0].requires_grad:
                args[0].requires_grad_()
            self.inp_activation[stage_id][mb_idx] = args[0]

        # Record start event for forward timing
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            forward_start_memory = torch.cuda.memory_allocated()
            forward_start_event.record()

        # Call compiled function
        out = self.compiled_fns[stage_id][subgraph_id](*self.parameters[stage_id][subgraph_id])

        # Record end event and calculate forward timing
        if self.tracing:
            forward_end_event.record()
            torch.cuda.synchronize()
            # Calculate total forward time
            forward_time = forward_start_event.elapsed_time(forward_end_event)
            forward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - forward_start_memory) / (1024**3)
            forward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # save first output of the last subgraph as output activation
        if subgraph_id == self.subgraph_order[stage_id][-1]:
            activation_tensor = out[0] if isinstance(out, (list, tuple)) else out
            self.out_activation[stage_id][mb_idx] = activation_tensor

        # clear the input tensors
        for i in self.input_idxs[stage_id][subgraph_id]:
            self.parameters[stage_id][subgraph_id][i] = None
        del args

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = forward_start_event.elapsed_time(end_event)
            # Store in trace_data (all microbatches stored sequentially)
            self.trace_data[stage_id]['forward']['forward'].append(forward_time)
            self.trace_data[stage_id]['forward']['total'].append(forward_time)
            self.trace_data[stage_id]['forward']['peak_memory_delta'].append(forward_peak_memory_delta_gb)
            self.trace_data[stage_id]['forward']['peak_memory'].append(forward_peak_memory_gb)

            if isinstance(out, list) or isinstance(out, tuple):
                self.fwd_objs[stage_id] = [torch.ones_like(t) for t in out]
            else:
                self.fwd_objs[stage_id] = torch.ones_like(out)
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return out

    def forward_no_nccl(self, stage_id: int, subgraph_id: int, mb_idx: int, *args):
        self.logger.debug(f"Calling cpu forward {stage_id} subgraph {subgraph_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

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
            return arg.to(self.device)
        args = list(map(place, args))

        # place input tensors in the correct indices
        for i, arg in zip(self.input_idxs[stage_id][subgraph_id], args):
            self.parameters[stage_id][subgraph_id][i] = arg

        # save first input to the first subgraph as input activation
        if stage_id != 0 and subgraph_id == self.subgraph_order[stage_id][0]:
            assert args[0].requires_grad
            self.inp_activation[stage_id][mb_idx] = args[0]

        out = self.compiled_fns[stage_id][subgraph_id](*self.parameters[stage_id][subgraph_id])
        
        # save first output of the last subgraph as output activation
        if subgraph_id == self.subgraph_order[stage_id][-1]:
            activation_tensor = out[0] if isinstance(out, (list, tuple)) else out
            self.out_activation[stage_id][mb_idx] = activation_tensor
        
        # clear the input tensors
        for i in self.input_idxs[stage_id][subgraph_id]:
            self.parameters[stage_id][subgraph_id][i] = None
        del args
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return out

    @ray.method(tensor_transport="nccl")
    def backward(self, stage_id: int, mb_idx: int, inp, loss_fn=None):
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

        # Propagate gradients backwards through the subgraphs by calling backward on the last subgraph, return the gradient of the input to the first subgraph
        subgraph_id = self.subgraph_order[stage_id][-1]

        # get the activation for the current stage and subgraph
        activation = self.out_activation[stage_id][mb_idx]
        
        # print("activation backward graph:")
        # print_backward_graph(activation)

        # compute loss in the last subgraph of the last stage. use the saved activation rather
        # than inp because the saved activation remembers the computation graph
        if loss_fn is not None:
            if self.truth is not None:
                labels = self.truth
            else:
                labels = inp[0]
            loss = loss_fn(activation, labels)
            loss.backward()
            self.loss.append(loss.item())
            # Clean up loss tensor after extracting item
            del loss
        # if not the last stage, backprop on the stored activation given 
        # the input gradient from the subsequent stage
        else:
            assert inp is not None
            assert activation.shape == inp.shape
            activation.backward(gradient=inp)

        del self.out_activation[stage_id][mb_idx]
        del activation

        # propagate the gradient backwards if not the first stage and first subgraph
        if stage_id != 0:
            ret = self.inp_activation[stage_id][mb_idx].grad
        else:
            ret = "done"

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
            # Store in trace_data (all microbatches stored sequentially)
            self.trace_data[stage_id]['backward']['backward'].append(backward_time)
            self.trace_data[stage_id]['backward']['total'].append(total_time)
            self.trace_data[stage_id]['backward']['peak_memory_delta'].append(backward_peak_memory_delta_gb)
            self.trace_data[stage_id]['backward']['peak_memory'].append(backward_peak_memory_gb)
            if stage_id != 0:
                self.bwd_objs[stage_id] = torch.ones_like(ret[0])
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return [ret] * 2

    def synchronize_gradients(self):
        """Synchronize gradients across all DP ranks for this stage using all-reduce."""     
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} synchronizing gradients")
        
        # Iterate over all stages and subgraphs on this actor and synchronize their parameters
        for stage_id, subgraph_params in self.parameters.items():
            for subgraph_id, parameters in subgraph_params.items():
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

    def get_weights(self, stage_id: int):
        return self.parameters[stage_id]

    def verify_weights(self, msg="first parameter"):
        for stage_id, subgraph_params in self.parameters.items():
            for subgraph_id, parameters in subgraph_params.items():
                for param in parameters:
                    if param is not None and param.grad is not None:
                        if len(param.shape) == 2:
                            self.logger.info(f"Actor {self.actor_id} global rank {self.global_rank} stage {stage_id} subgraph {subgraph_id} {msg}: {param.shape, param[0][0], param.grad[0][0]}")
                        elif len(param.shape) == 1:
                            self.logger.info(f"Actor {self.actor_id} global rank {self.global_rank} stage {stage_id} subgraph {subgraph_id} {msg}: {param.shape, param[0], param.grad[0]}")
                        else:
                            assert False, f"Unsupported parameter shape: {param.shape}"
                        return "done"

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
