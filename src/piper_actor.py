import ray
import torch
import logging
import os
import time
from torch._guards import CompileId
from torch.nn import Parameter
from collections import defaultdict
from .piper_utils import deserialize_graphmodule, piper_metadata

# torch._dynamo.config.compiled_autograd = True
def backward_backend(gm, example_inputs, **kwargs):
    print("BACKWARD GRAPH")

    placeholders = gm.graph.find_nodes(op="placeholder")
    graphargs = [node.meta["grapharg"] for node in placeholders]
    for arg in graphargs:
        if isinstance(arg._example, list) or isinstance(arg._example, tuple):
            for ex in arg._example:
                print(f"Arg {ex.shape} | parameter: {isinstance(ex, Parameter)} | requires_grad: {ex.requires_grad}")
    gm.print_readable()

    return gm

@ray.remote(num_gpus=1)
class PiperActor:
    # def __init__(self, id, compiler_fn, example_inputs, parameters, optim_fn=None):
    def __init__(self, id, optim_fn=None):
        torch.manual_seed(0)

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        self.log.info(f"Initializing Ray actor {id} with PID: {os.getpid()}")

        start = time.perf_counter()

        self.actor_id = id
        self.optim_fn = optim_fn

        self.input = None
        self.truth = None
        self.fwd_objs = {}
        self.bwd_objs = {}

        # ordered list of frame ids for ordering the fx.Graphs on this actor
        self.frame_ids = []
        # map stage id -> compiled fx.Graph function
        self.compiled_fns = dict()
        # map stage id -> original GraphModule (for hook registration)
        self.graph_modules = dict()
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

    def id(self):
        return self.actor_id

    def send_input(self, tensor):
        self.input = tensor.to('cuda')
        return "done"
    
    def send_truth(self, tensor):
        self.truth = tensor.to('cuda')
        return "done"
    
    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()
        return "done"

    def get_peak_memory(self):
        return torch.cuda.max_memory_allocated() / (1024**3)

    def compile_graph(self, stage_id, gm_data, compiler_fn, graphargs, input_idxs):
        self.log.info(f"Compiling graph on actor {self.actor_id}. stage id: {stage_id}. inputs: {len(graphargs)}")
        start = time.perf_counter()

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
        if [param for param in self.parameters[stage_id] if param is not None]:
            self.optims[stage_id] = self.optim_fn([param for param in self.parameters[stage_id] if param is not None])

        # MEMORY CLEANUP
        del gm_data

        end = time.perf_counter()
        self.log.debug(f"compile_graph took {(end-start)*1000:.2f}ms")
        return "Finished compiling"

    @ray.method(tensor_transport="nccl")
    def forward(self, stage_id: int, mb_idx: int, *args):
        self.log.debug(f"Calling forward {stage_id} mb {mb_idx} on actor {self.actor_id}")
        # print(f"Calling forward {stage_id} mb {mb_idx} on actor {self.actor_id}")

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

        def unwrap_arg(arg):
            if isinstance(arg, list) or isinstance(arg, tuple):
                assert len(arg) == 1
                return arg[0]
            else:
                return arg
        args = list(map(unwrap_arg, args))

        def detach_arg(arg):
            return arg.detach().clone()
        args = list(map(detach_arg, args))

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
            if not args[0].requires_grad:
                args[0].requires_grad_()
            self.prev_activation[stage_id][mb_idx] = args[0]

        # Record start event for forward timing
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            forward_start_memory = torch.cuda.memory_allocated()
            forward_start_event.record()

        # Call compiled function
        out = self.compiled_fns[stage_id](*self.parameters[stage_id])

        # Record end event and calculate forward timing
        if self.tracing:
            forward_end_event.record()
            torch.cuda.synchronize()
            
            # Calculate total forward time
            forward_time = forward_start_event.elapsed_time(forward_end_event)
            
            forward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - forward_start_memory) / (1024**3)
            forward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # save first output as activation BEFORE clearing input tensors
        # This ensures the activation doesn't share memory with parameters
        # that might be modified when processing other stages on the same actor
        activation_tensor = out[0] if isinstance(out, (list, tuple)) else out
        self.activation[stage_id][mb_idx] = activation_tensor

        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.parameters[stage_id][i] = None
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
        
        return out

    def forward_no_nccl(self, stage_id: int, mb_idx: int, *args):
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

        # print(f"Stage {stage_id} input_idxs: {self.input_idxs[stage_id]}")
        # for arg in args:
        #     print(arg.shape, arg.requires_grad)
        # save first input as previous activation
        if stage_id != 0:
            assert args[0].requires_grad
            self.prev_activation[stage_id][mb_idx] = args[0]

        out = self.compiled_fns[stage_id](*self.parameters[stage_id])
        
        # save first output as activation BEFORE clearing input tensors
        # This ensures the activation doesn't share memory with parameters
        # that might be modified when processing other stages on the same actor
        activation_tensor = out[0] if isinstance(out, (list, tuple)) else out
        self.activation[stage_id][mb_idx] = activation_tensor
        
        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.parameters[stage_id][i] = None
        del args
        return out

    @ray.method(tensor_transport="nccl")
    def backward(self, stage_id: int, mb_idx: int, inp, loss_fn=None):
        self.log.debug(f"Calling backward {stage_id} mb {mb_idx} on actor {self.actor_id}")
        # print(f"Calling backward {stage_id} mb {mb_idx} on actor {self.actor_id}")
        
        if self.tracing:
            beginning_event = torch.cuda.Event(enable_timing=True)
            backward_start_event = torch.cuda.Event(enable_timing=True)
            backward_end_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            beginning_event.record()

        # get the activation for the current stage
        assert mb_idx in self.activation[stage_id], f"mb_idx {mb_idx} not in activation[stage_id {stage_id}]"
        activation = self.activation[stage_id][mb_idx]
        
        # Record start event for backward timing
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            backward_start_memory = torch.cuda.memory_allocated()
            backward_start_event.record()
        
        # compute loss in the last stage. use the saved activation rather
        # than inp because the saved activation remembers the computation graph
        if loss_fn is not None:
            if self.truth is not None:
                labels = self.truth
            else:
                labels = inp[0]
            loss = loss_fn(activation, labels)
            loss.backward()
            self.loss.append(loss.item())
        # if not the last stage, backprop on the stored activation given 
        # the input gradient from the subsequent stage
        else:
            assert inp is not None
            assert activation.shape == inp.shape
            activation.backward(gradient=inp)

        # Record end event and calculate backward timing
        if self.tracing:
            backward_end_event.record()
            torch.cuda.synchronize()
            
            # Calculate total backward time
            backward_time = backward_start_event.elapsed_time(backward_end_event)
            
            backward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - backward_start_memory) / (1024**3)
            backward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        del self.activation[stage_id][mb_idx]
        del activation

        # propagate the gradient backwards if not the first stage
        if stage_id != 0:
            ret = [self.prev_activation[stage_id][mb_idx]]
        else:
            ret = ["done"]
        
        if self.tracing:
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
        
        return ret + ret

    def backward_compile(self, stage_id: int, mb_idx: int, inp, loss_fn=None):
        self.log.info(f"Calling backward {stage_id} mb {mb_idx} on actor {self.actor_id}", inp)
        
        # get the activation for the current stage
        activation = self.activation[stage_id][mb_idx]

        # Compile the backward pass
        @torch.compile(backend=backward_backend)
        def backward_loss(loss_fn, activation, labels):
            loss = loss_fn(activation, labels)
            loss.backward()
            return loss
        @torch.compile(backend=backward_backend)
        def backward_activation(activation, inp):
            activation.backward(gradient=inp)

        # compute loss in the last stage. use the saved activation rather
        # remembers the computation graph, inp should be the labels
        if loss_fn is not None:
            labels = inp[0]
            loss = backward_loss(loss_fn, activation, labels)
            self.loss.append(loss)

        # if not the last stage, backprop on the stored activation given 
        # the input gradient from the subsequent stage
        else:
            grad = inp[0]
            assert activation.shape == grad.shape
            backward_activation(activation, grad)

        # propagate the gradient backwards if not the first stage
        if stage_id != 0:
            ret = [self.prev_activation[stage_id][mb_idx]]
        else:
            ret = ["done"]

        return ret

    def update(self, *done_mbs):
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