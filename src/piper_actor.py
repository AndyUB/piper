import ray
import torch
import logging
import os
import time
from torch._guards import CompileId
from collections import defaultdict
from .piper_utils import deserialize_graphmodule, piper_metadata
import torch.distributed as dist

from typing import Callable

torch.set_float32_matmul_precision('high')

@ray.remote
class StageActor:
    # def __init__(self, id, compiler_fn, example_inputs, parameters, optim_fn=None):
    def __init__(self, id, optim_fn=None, dp_rank=0, dp_world_size=1, num_stages = 1, stage_id_for_dp=None):
        torch.manual_seed(0)

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        self.log.info(f"Initializing Ray actor {id} with PID: {os.getpid()}")

        start = time.perf_counter()

        self.actor_id = id
        self.optim_fn = optim_fn
        
        # Data parallel attributes
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.num_stages = num_stages
        self.stage_id_for_dp = stage_id_for_dp  if stage_id_for_dp is not None else id# Stage ID without DP offset
        self.dp_process_group = None  # Will be set later
        self.stage_dp_group = None
        self.global_stage = self.num_stages * self.dp_rank + self.actor_id
        self.global_world_size = self.num_stages * self.dp_world_size

        self.input = None
        self.truth = None
        self.fwd_objs = {}
        self.bwd_objs = {}

        # ordered list of frame ids for ordering the fx.Graphs on this actor
        self.frame_ids = []
        # map stage id -> compiled fx.Graph function
        self.compiled_fns = dict()
        # map stage id -> model parameters used by the fx.Graph with holes (None values) for input tensors
        self.parameters = dict()
        # map stage id -> indices of the input tensors (as opposed to model parameters) used by the fx.Graph
        self.input_idxs = dict()
        # map stage id -> optimizer for the fx.Graph
        self.optims = dict()
        # map stage -> mb_idx -> previous activation (if this stage is not first)
        self.prev_activation = defaultdict(dict)
        # map stage id -> mb_idx -> current activation
        self.activation = defaultdict(dict)
        # accumuate loss for each microbatch
        self.loss = []

        # Timing infrastructure
        self.tracing = False  # Toggle for timing and memory tracing
        self.trace_data = {'update': {'total': [], 'peak_memory_delta': [], 'peak_memory': []}}

        end = time.perf_counter()
        self.log.debug(f"__init__ took {(end-start)*1000:.2f}ms")

        self.log.info(f"About to join a distributed process group")
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print(f"[PIPER RANK {self.dp_rank}] CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
       

    def id(self):
        return self.actor_id

    def send_input(self, tensor):
        self.input = tensor.to('cuda')
        return "done"
    
    def send_truth(self, tensor):
        self.truth = tensor.to('cuda')
        return "done"

    # this is for making sure every StageActor process joins the big process group
    # After everyone joins. then each StageActor can create their own Stage Group
    @ray.method
    def join_process_group(self):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print(f"[PIPER RANK {self.dp_rank}] CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"Stage {self.stage_id_for_dp} on dp_rank {self.dp_rank} w/ global stage id {self.global_stage} is joining the large torch process group!")
        master_addr = os.environ.get('PIPER_MASTER_ADDR', "127.0.0.1")
        master_port = os.environ.get('PIPER_MASTER_PORT', "10000")
        #init_method = f"tcp://{master_addr}:{master_port}"
        init_method = f"tcp://{master_addr}:{master_port}"
        dist.init_process_group("nccl", init_method=init_method, rank=self.global_stage, world_size=self.global_world_size)
        self.log.info(f"Global Stage {self.global_stage} joined the DP group! Now about to join the stage_process_groups...")
        self.join_stage_process_group()
    
    # for each Stage i in a dp_rank, create a new process group consisting of only actors of that particular stage
    def join_stage_process_group(self):
        # every process needs to participate in every subgroup creation
        for stage in range(self.num_stages):
            relevant_ranks = [(stage + self.num_stages * i) for i in range(self.dp_world_size)]
            process_group = dist.new_group(ranks=relevant_ranks, backend='nccl')
            if self.global_stage % self.num_stages == stage:
                self.stage_dp_group = process_group
                print(f"{self.global_stage} has joined its own stage dp group!")

    def compile_graph(self, stage_id, gm_data, compiler_fn, graphargs, input_idxs):
        self.log.info(f"Compiling graph on actor {self.actor_id}. stage id: {stage_id}. inputs: {len(graphargs)}")
        start = time.perf_counter()

        self.trace_data[stage_id] = {
            'forward': {
                'pre_forward': [],
                'forward': [],
                'post_forward': [],
                'peak_memory_delta': [],
                'peak_memory': []
            },
            'backward': {
                'pre_backward': [],
                'backward': [],
                'post_backward': [],
                'peak_memory_delta': [],
                'peak_memory': []
            },
        }

        # compile the graph with the given graphargs
        gm = deserialize_graphmodule(gm_data)
        compiled_fn = compiler_fn(gm, graphargs)
        assert callable(compiled_fn), "compiler_fn did not return callable"
        self.compiled_fns[stage_id] = compiled_fn

        # discard the graphargs that correspond to input tensors
        for i in input_idxs:
            graphargs[i] = None
        self.input_idxs[stage_id] = input_idxs

        # save the graphargs that correspond to model parameters
        self.parameters[stage_id] = graphargs

        # initialize the optimizer for this stage
        self.optims[stage_id] = self.optim_fn([param for param in self.parameters[stage_id] if param is not None])

        # MEMORY CLEANUP
        del gm
        del gm_data

        end = time.perf_counter()
        self.log.debug(f"compile_graph took {(end-start)*1000:.2f}ms")
        return "Finished compiling"

    @ray.method(tensor_transport="nccl")
    def call(self, stage_id: int, mb_idx: int, *args):
        self.log.info(f"Calling forward {stage_id} mb {mb_idx} on actor {self.actor_id} with {len(args)} args")
        
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            pre_forward_event = torch.cuda.Event(enable_timing=True)
            forward_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            forward_start_memory = torch.cuda.memory_allocated()
            start_event.record()
        
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

        # place input tensors in the correct indices
        for i, arg in zip(self.input_idxs[stage_id], args):
            self.parameters[stage_id][i] = arg

        # save first input as previous activation
        if stage_id != 0:
            args[0].requires_grad_()
            self.prev_activation[stage_id][mb_idx] = args[0]

        if self.tracing:
            pre_forward_event.record()

        out = self.compiled_fns[stage_id](*self.parameters[stage_id])

        if self.tracing:
            forward_event.record()

        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.parameters[stage_id][i] = None
        del args

        # save first output as activation
        self.activation[stage_id][mb_idx] = out[0]

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            pre_forward_time = start_event.elapsed_time(pre_forward_event)
            forward_time = pre_forward_event.elapsed_time(forward_event)
            post_forward_time = forward_event.elapsed_time(end_event)
            
            forward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - forward_start_memory) / (1024**3)
            forward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

            self.trace_data[stage_id]['forward']['pre_forward'].append(pre_forward_time)
            self.trace_data[stage_id]['forward']['forward'].append(forward_time)
            self.trace_data[stage_id]['forward']['post_forward'].append(post_forward_time)
            self.trace_data[stage_id]['forward']['peak_memory_delta'].append(forward_peak_memory_delta_gb)
            self.trace_data[stage_id]['forward']['peak_memory'].append(forward_peak_memory_gb)

            if isinstance(out, list) or isinstance(out, tuple):
                self.fwd_objs[stage_id] = [torch.ones_like(t) for t in out]
            else:
                self.fwd_objs[stage_id] = torch.ones_like(out)
        return out

    def call_cpu(self, stage_id: int, mb_idx: int, *args):
        self.log.info(f"Calling cpu forward {stage_id} mb {mb_idx} on actor {self.actor_id} with {len(args)} args")

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

        # place input tensors in the correct indices
        for i, arg in zip(self.input_idxs[stage_id], args):
            self.parameters[stage_id][i] = arg

        # save first input as previous activation
        if stage_id != 0:
            assert args[0].requires_grad
            self.prev_activation[stage_id][mb_idx] = args[0]

        out = self.compiled_fns[stage_id](*self.parameters[stage_id])
        
        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.parameters[stage_id][i] = None
        del args

        # save first output as activation
        self.activation[stage_id][mb_idx] = out[0]
        return out

    @ray.method(tensor_transport="nccl")
    def backward(self, stage_id: int, mb_idx: int, inp, loss_fn=None):
        self.log.info(f"Calling backward {stage_id} mb {mb_idx} on actor {self.actor_id}", inp)
        
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            pre_backward_event = torch.cuda.Event(enable_timing=True)
            backward_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
        
        # get the activation for the current stage
        activation = self.activation[stage_id][mb_idx]
        
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            backward_start_memory = torch.cuda.memory_allocated()

        if self.tracing:
            pre_backward_event.record()
        
        # compute loss in the last stage. use the saved activation rather
        # than inp because the saved activation remembers the computation graph
        if loss_fn is not None:
            if self.truth is not None:
                labels = self.truth
            else:
                labels = inp[0]
            loss = loss_fn(activation, labels)
            self.loss.append(loss)
            loss.backward()
        # if not the last stage, backprop on the stored activation given 
        # the input gradient from the subsequent stage
        else:
            assert inp is not None
            assert activation.shape == inp.shape
            activation.backward(gradient=inp)

        if self.tracing:
            backward_event.record()

        del self.activation[stage_id][mb_idx]
        del activation

        if stage_id != 0:
            ret = [self.prev_activation[stage_id][mb_idx].grad.clone()]
            del self.prev_activation[stage_id][mb_idx]
        else:
            ret = ["done"]
        
        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            pre_backward_time = start_event.elapsed_time(pre_backward_event)
            backward_time = pre_backward_event.elapsed_time(backward_event)
            post_backward_time = backward_event.elapsed_time(end_event)
            backward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - backward_start_memory) / (1024**3)
            backward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

            self.trace_data[stage_id]['backward']['pre_backward'].append(pre_backward_time)
            self.trace_data[stage_id]['backward']['backward'].append(backward_time)
            self.trace_data[stage_id]['backward']['post_backward'].append(post_backward_time)
            self.trace_data[stage_id]['backward']['peak_memory_delta'].append(backward_peak_memory_delta_gb)
            self.trace_data[stage_id]['backward']['peak_memory'].append(backward_peak_memory_gb)
            
            if stage_id != 0:
                self.bwd_objs[stage_id] = torch.ones_like(ret[0])
        return ret + ret

    def set_dp_process_group(self, group):
        """Set the torch.distributed process group for this actor's DP synchronization."""
        self.dp_process_group = group
        self.log.info(f"Actor {self.actor_id}: Set DP process group for stage {self.stage_id_for_dp}")
        return "done"

    def synchronize_gradients(self):
        """Synchronize gradients across all DP ranks for this stage using all-reduce."""
        if self.stage_dp_group is None or self.dp_world_size is None or self.dp_world_size <= 1:
            # No synchronization needed for single DP rank
            return
        
        self.log.info(f"Actor {self.actor_id}: Synchronizing gradients for stage {self.stage_id_for_dp}")
        
        # Iterate over all stages on this actor and synchronize their parameters
        for stage_id, parameters in self.parameters.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    # Use AVG to average gradients across DP ranks
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.stage_dp_group)

    def update(self, *done_mbs):
        # first need to synchronize the gradients
        #print("UPDATE IS HAPPENING RIGHT NOW YOU IMBECILE!")
        self.synchronize_gradients()
        self.log.debug(f"Calling update on actor {self.actor_id}")
        
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.reset_peak_memory_stats()
            update_start_memory = torch.cuda.memory_allocated()

            start_event.record()
        
        assert self.optim_fn
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

    def get_trace_data(self) -> dict:
        """
        Retrieve timing data collected during training.
        
        Returns:
            dict: Dictionary containing timing data for call, backward, and update functions.
                 Each function has sub-dictionaries with timing measurements in milliseconds.
        """
        return self.trace_data.copy()
    
    def clear_trace_data(self) -> None:
        """
        Clear all collected timing data.
        """
        for stage_id in self.trace_data:
            self.trace_data[stage_id] = {
                'forward': {
                    'pre_forward': [],
                    'forward': [],
                    'post_forward': [],
                    'peak_memory_delta': [],
                    'peak_memory': []
                },
                'backward': {
                    'pre_backward': [],
                    'backward': [],
                    'post_backward': [],
                    'peak_memory_delta': [],
                    'peak_memory': []
                },
            }
        self.trace_data['update'] = {
            'total': [],
            'peak_memory_delta': [],
            'peak_memory': []
        }

    def set_tracing(self, enabled: bool) -> None:
        """
        Enable or disable timing and memory tracing.
        
        Args:
            enabled (bool): True to enable tracing, False to disable.
        """
        self.tracing = enabled
        self.log.info(f"Actor {self.actor_id}: Tracing {'enabled' if enabled else 'disabled'}")

    def start_mem_tracing(self) -> None:
        torch.cuda.memory._record_memory_history()
        return "done"
    
    def stop_mem_tracing(self) -> None:
        torch.cuda.memory._dump_snapshot(f"actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        print(f"Saved memory snapshot to actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return "done"
    
    @ray.method(tensor_transport="nccl")
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

# TODO: when scheduling the run_dp_rank processes to GPUs, set the CUDA_VISIBLE_DEVICES for these processes to be disjoint i.e rank 0 only sees GPU 0, rank 1 only sees GPU 1, etc...
@ray.remote(num_gpus=0.01)
def run_dp_rank(rank, world_size, master_addr, master_port, num_stages, training_func: Callable, *args, **kwargs):
    print("Inside the run_dp_rank function")
    def wrapped_fn(*args, **kwargs):
            print(f"inside the wrapped_fn that is being run on rank {rank}")
            os.environ['PIPER_RANK'] = str(rank)
            os.environ['PIPER_WORLD_SIZE'] = str(world_size)
            os.environ['PIPER_NUM_STAGES'] = str(num_stages)
            os.environ['PIPER_MASTER_ADDR'] = str(master_addr)
            os.environ['PIPER_MASTER_PORT'] = str(master_port)
            #init_method = f"tcp://{master_addr}:{master_port}"
            #print(f"on rank {rank} about to init the process group")
            #dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
            #print(f"initialized {rank}!")
            res = training_func(*args, **kwargs)
            #dist.destroy_process_group()
            return res
    return wrapped_fn(*args, **kwargs)

def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    return port

@ray.remote
class PiperProgramCoordinator:
    """ Central Actor that Coordinates all the DP replicas of a single pipeline"""
    def __init__(self, dp_size: int = 2, num_stages: int = 1):
        self.dp_size = dp_size
        self.num_stages = num_stages
        print(f"Created the PiperProgramCoordinator with dp_size={self.dp_size}, num_stages={self.num_stages}")
        self.master_port = find_free_port()
    
    def run_program(self, training_func: Callable, *args, **kwargs):
        print("In the run_program method in the ProgramCoordinator")
        if torch.cuda.is_available():
            print("CUDA is available")
        else:
            print("CUDA is NOT available")
        ret_vals_handles, ret_vals = [], []
        for rank in range(self.dp_size):
            print(f"Dispathching the run_dp_rank function for rank {rank}")
            ret_handle = run_dp_rank.remote(rank, self.dp_size, "127.0.0.1", self.master_port, self.num_stages, training_func, *args, **kwargs)
            print(f"Dispatched for rank {rank}")
            ret_vals_handles.append(ret_handle)
        for ret_val_handle in ret_vals_handles:
            ret_vals.append(ray.get(ret_val_handle))
        return ret_vals
    

            


