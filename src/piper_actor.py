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

from .piper_utils import _deserialize_graphmodule, create_logger, LOG_LEVEL

CLEANUP_MEMORY = False

logger = create_logger("piper_actor", LOG_LEVEL)

def _create_actors(num_actors, optim_class, num_mbs, naive_gradient_sync=False):
    dp_rank = int(os.environ['PIPER_DP_RANK'])
    world_size = int(os.environ['PIPER_WORLD_SIZE'])
    dp_degree = int(os.environ['PIPER_DP_DEGREE'])
    pp_degree = int(os.environ['PIPER_PP_DEGREE'])

    from .piper_utils import piper_metadata
    for actor_id in range(num_actors):
        global_rank = dp_rank * dp_degree + actor_id
        actor = PiperActor.options(num_gpus=1, max_concurrency=3).remote(actor_id, optim_class, world_size, num_mbs, naive_gradient_sync, dp_rank=dp_rank, dp_degree=dp_degree, pp_degree=pp_degree)
        piper_metadata.actors[actor_id] = actor
        logger.debug(f"DP rank {dp_rank} created actor {actor} global rank {global_rank}")

def _get_actor(actor_id):
    from .piper_utils import piper_metadata
    return piper_metadata.actors[actor_id]

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
        self.tracing = enabled
        self.logger.info(f"Actor {self.actor_id}: Tracing {'enabled' if enabled else 'disabled'}")

    def start_mem_tracing(self) -> None:
        torch.cuda.memory._record_memory_history()
    
    def stop_mem_tracing(self) -> None:
        torch.cuda.memory._dump_snapshot(f"actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        self.logger.info(f"Saved memory snapshot to actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()

    def get_peak_memory(self):
        return torch.cuda.max_memory_allocated() / (1024**3)

    def load_input(self, tensor):
        self.input = tensor.to(self.device)
    
    def load_labels(self, tensor):
        self.truth = tensor.to(self.device)

    def _join_process_groups(self):
        master_addr = os.environ.get('PIPER_MASTER_ADDR', "127.0.0.1")
        master_port = os.environ.get('PIPER_MASTER_PORT', "10000")
        init_method = f"tcp://{master_addr}:{master_port}"

        dist.init_process_group("nccl", init_method=init_method, rank=self.global_rank, world_size=self.world_size)
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} has GPU {os.environ['CUDA_VISIBLE_DEVICES']}, joined the global process group")

        if self.dp_degree > 1:
            self._join_dp_process_group()
    
    def _join_dp_process_group(self):
        num_dp_groups = self.world_size // self.dp_degree
        for dp_group_id in range(num_dp_groups):
            group_ranks = [(dp_group_id + num_dp_groups * i) for i in range(self.dp_degree)]
            process_group = dist.new_group(ranks=group_ranks, backend='nccl')
            if self.global_rank % num_dp_groups == dp_group_id:
                self.dp_group = process_group
                self.logger.info(f"Global rank {self.global_rank} joined its dp group {dp_group_id} along with ranks {group_ranks}")

    def _prepare_comm_ops(self, stage_id, comm_ops, graphargs, ids):
        def hook_maker(tensor_id):
            def post_backward_hook(grad):
                self.comm_op_status[stage_id][tensor_id] += 1
                self.logger.debug(f"Updating tensor status dp_rank: {self.dp_rank}, tensor={comm_op.name}, shape={grad.shape}, status={self.comm_op_status[stage_id][tensor_id]}")
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
                            self.logger.debug(f"Registering update hook dp_rank: {self.dp_rank}, tensor={comm_op.name}, shape={tensor_shape}")
                            t.register_post_accumulate_grad_hook(hook_maker(tensor_id))
                        else:
                            raise ValueError(f"Unknown comm op type or dependency: {comm_op.op} {comm_op.dep}")
                   case _:
                       raise ValueError(f"Unknown communication operation: {comm_op.op}")

    def _comm_loop(self, stage_id):
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
                            self.logger.debug(f"Allreduce on dp_rank: {self.dp_rank}, tensor={comm_op.name}, shape={comm_op.tensor.shape}")
                        self.comm_op_status[stage_id][comm_op.tensor_id] = 0
                        self.comm_op_handles[stage_id][comm_op.tensor_id] = handle
                        done = True
                else:
                    raise ValueError(f"Unknown comm op: {comm_op}")

    def _wait_for_comm_ops(self):
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} waiting for comm ops")
        for stage_id, comm_ops in self.comm_ops.items():
            for comm_op in comm_ops:
                self.logger.debug(f"Waiting for comm op to be launched dp_rank: {self.dp_rank}, tensor={comm_op.name}, shape={comm_op.tensor.shape}")
                done = False
                while not done:
                    if comm_op.tensor_id in self.comm_op_handles[stage_id]:
                        self.logger.debug(f"Waiting for comm op to be finished dp_rank: {self.dp_rank}, tensor={comm_op.name}, shape={comm_op.tensor.shape}")
                        self.comm_op_handles[stage_id][comm_op.tensor_id].wait()
                        done = True

    def _load_stage(self, stage_id: int, gm_data, comm_ops, params_with_holes, ids, input_idxs):
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
        gm = _deserialize_graphmodule(gm_data)

        # Store GraphModule reference
        self.graph_modules[stage_id] = gm
        self.forward_fns[stage_id] = gm.forward

        # place parameters on the device
        def move_to_device(idx, arg):
            if idx not in input_idxs:
                if arg.requires_grad:
                    return arg.to(self.device).detach().requires_grad_(True)
                else:
                    return arg.to(self.device)
            return arg
        params_with_holes = list(map(move_to_device, range(len(params_with_holes)), params_with_holes))


        # prepare tensors with comm ops
        if not self.naive_gradient_sync and self.dp_degree > 1:
            self._prepare_comm_ops(stage_id, comm_ops, params_with_holes, ids)
        self.comm_ops[stage_id] = list(reversed(comm_ops))

        # save parameters
        self.input_idxs[stage_id] = input_idxs
        self.parameters[stage_id] = params_with_holes

        # add the parameters to the optimizer for this stage
        only_params = [param for param in params_with_holes if param is not None]
        if stage_id not in self.optims:
            self.optims[stage_id] = self.optim_class(only_params)
        else:
            self.optims[stage_id].add_param_group({'params': only_params})

        del gm_data

    # @ray.method(tensor_transport="nccl")
    def _forward(self, stage_id: int, mb_idx: int, args):
        self.logger.debug(f"Calling forward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

        if self.tracing:
            beginning_event = torch.cuda.Event(enable_timing=True)
            forward_start_event = torch.cuda.Event(enable_timing=True)
            forward_end_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            beginning_event.record()
        
        if not (isinstance(args, list) or isinstance(args, tuple)):
            args = [args]

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
        self.logger.debug(f"Stage {stage_id} inputs: {[arg.shape for arg in args]}, input_idxs: {self.input_idxs[stage_id]}, parameters: {[p.shape if p is not None else None for p in self.parameters[stage_id]]}")
        assert len(args) == len([p for p in self.parameters[stage_id] if p is None]), "Number of arguments must match number of holes in the parameters"
        for i, arg in zip(self.input_idxs[stage_id], args):
            self.parameters[stage_id][i] = arg

        # save first input that requires grad as input activation
        if stage_id > 0:
            inp_with_grad = [inp for inp in args if inp.requires_grad]
            assert len(inp_with_grad) == 1, "Piper only supports one input per subgraph with requires_grad"
            if inp_with_grad[0].dtype == torch.float:
                inp_with_grad[0].requires_grad_().retain_grad()
            self.logger.debug(f"Saving input activation {inp_with_grad[0].shape} for stage {stage_id} mb {mb_idx}")
            self.inp_activation[stage_id][mb_idx] = inp_with_grad[0]

        # Record start event for forward timing
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            forward_start_memory = torch.cuda.memory_allocated()
            forward_start_event.record()

        # Call compiled function
        output = self.forward_fns[stage_id](*self.parameters[stage_id])

        # Record end event and calculate forward timing
        if self.tracing:
            forward_end_event.record()
            torch.cuda.synchronize()
            # Calculate total forward time
            forward_time = forward_start_event.elapsed_time(forward_end_event)
            forward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - forward_start_memory) / (1024**3)
            forward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # save first output that requires grad as output activation
        # TODO: support multiple outputs
        out_with_grad = [out for out in output if out.requires_grad]
        assert len(out_with_grad) == 1, "Piper only supports one output per subgraph with requires_grad"
        self.logger.debug(f"Saving output activation {out_with_grad[0].shape} for stage {stage_id} mb {mb_idx}")
        self.out_activation[stage_id][mb_idx] = out_with_grad[0]

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
        
        self.logger.debug(f"Forward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank} returning {[out.shape for out in output]}")
        return output

    # @ray.method(tensor_transport="nccl")
    def _backward(self, stage_id: int, mb_idx: int, inp, truth=None, loss_fn=None):
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
            self.logger.debug(f"Backward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank} receiving input {[inp.shape for inp in inp]}")
            assert len(inp) == 1
            inp = inp[0]

        # Get the activations for this stage and microbatch
        out_activation = self.out_activation[stage_id][mb_idx]
        if stage_id > 0:
            inp_activation = self.inp_activation[stage_id][mb_idx]

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
            self.logger.debug(f"activation: {out_activation.shape}, input: {inp.shape}")
            assert out_activation.shape == inp.shape
            out_activation.backward(gradient=inp)

        del out_activation

        # propagate the gradient backwards if not the first stage
        if stage_id > 0:
            ret = inp_activation.grad
            del inp_activation
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
            self.trace_data[stage_id]['backward']['backward'].append(backward_time)
            self.trace_data[stage_id]['backward']['total'].append(total_time)
            self.trace_data[stage_id]['backward']['peak_memory_delta'].append(backward_peak_memory_delta_gb)
            self.trace_data[stage_id]['backward']['peak_memory'].append(backward_peak_memory_gb)
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return [ret] * 2

    def _synchronize_gradients(self):
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} synchronizing gradients")
        # Iterate over all stages on this actor and synchronize their parameters
        for stage_id, parameters in self.parameters.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.dp_group)

    def _update(self, *done_mbs):
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
                self._synchronize_gradients()
            else:
                self._wait_for_comm_ops()
        
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
