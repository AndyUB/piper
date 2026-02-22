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
from pathlib import Path
from typing import Callable

from .piper_utils import (
    _deserialize_graphmodule,
    create_logger,
    LOG_LEVEL,
    piper_metadata,
)

CLEANUP_MEMORY = False

logger = create_logger("piper_actor", LOG_LEVEL)


def _get_rank(pp_rank, dp_rank, pp_degree):
    return pp_rank + dp_rank * pp_degree


def _create_actors(
    num_actors,
    optim_class,
    num_mbs,
    num_stages,
    p2p_schedules,
    naive_gradient_sync=False,
):
    dp_rank = int(os.environ["PIPER_DP_RANK"])
    world_size = int(os.environ["PIPER_WORLD_SIZE"])
    dp_degree = int(os.environ["PIPER_DP_DEGREE"])
    pp_degree = int(os.environ["PIPER_PP_DEGREE"])

    from .piper_utils import piper_metadata

    for pp_rank in range(num_actors):
        global_rank = _get_rank(pp_rank, dp_rank, pp_degree)
        if dp_degree > 1:
            max_concurrency = 2
        else:
            max_concurrency = 1
        p2p_schedule = p2p_schedules[pp_rank]
        actor = PiperActor.options(num_gpus=1, max_concurrency=max_concurrency).remote(
            pp_rank,
            optim_class,
            world_size,
            num_mbs,
            num_stages,
            p2p_schedule,
            naive_gradient_sync,
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            pp_degree=pp_degree,
        )
        piper_metadata.actors[pp_rank] = actor
        logger.debug(
            f"DP rank {dp_rank} created actor {actor} global rank {global_rank}"
        )


def _get_actor(pp_rank):
    from .piper_utils import piper_metadata

    return piper_metadata.actors[pp_rank]


@ray.remote
class PiperActor:
    def __init__(
        self,
        pp_rank,
        optim_class,
        world_size,
        num_mbs,
        num_stages,
        p2p_schedule,
        naive_gradient_sync=False,
        dp_rank=0,
        dp_degree=1,
        pp_degree=1,
    ):
        self.logger = create_logger("piper_actor", LOG_LEVEL)

        self.pp_rank = pp_rank
        self.optim_class = optim_class
        self.naive_gradient_sync = naive_gradient_sync

        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.pp_degree = pp_degree
        self.world_size = world_size

        self.num_mbs = num_mbs
        self.num_stages = num_stages
        # list of (src_stage, dst_stage, mb_idx, is_sender)
        self.p2p_schedule = p2p_schedule
        self.next_p2p_idx = 0
        # set of (src_stage, dst_stage, mb_idx, is_sender) for completed p2p ops
        self.executed_p2ps = set()
        self.dp_group = None
        self.pp_group = None
        self.device = "cuda"

        self.global_rank = _get_rank(pp_rank, dp_rank, pp_degree)

        self.logger.info(
            f"Initializing Ray actor {pp_rank} global rank {self.global_rank} GPU {os.environ['CUDA_VISIBLE_DEVICES']}"
        )

        self.input = None
        self.labels = None

        self.comp_stream = torch.cuda.Stream()
        self.comm_stream = torch.cuda.Stream()
        self.a2a_stream = torch.cuda.Stream()
        self.p2p_stream = torch.cuda.Stream()

        # map stage id -> compiled fx.Graph function
        self.forward_fns = dict()
        # map stage id -> original GraphModule (for hook registration)
        self.graph_modules = dict()
        # map stage id -> model parameters used by the fx.Graph with holes (None values) for input tensors
        self.forward_args = dict()
        # map stage id -> input idx -> input tensor metadata
        self.forward_input_meta = defaultdict(dict)
        # map stage id -> indices of the input tensors (as opposed to model parameters) used by the fx.Graph
        self.input_idxs = dict()
        # map stage id -> optimizer for the fx.Graph
        self.optims = dict()
        # map stage id -> mb_idx -> previous activation (if this stage is not first)
        self.inp_activation = defaultdict(dict)
        # map stage id -> mb_idx -> current activation
        self.out_activation = defaultdict(dict)
        # map (src_stage, dst_stage, mb_idx, is_sender) -> tensor for p2p communication
        self.p2p_cache = dict()
        # accumuate loss for each microbatch
        self.loss = []
        # map stage id -> data parallel communication operations
        self.comm_ops = dict()
        # map stage id -> tensor id -> comm op status
        self.comm_op_status = defaultdict(lambda: defaultdict(int))
        # map stage id -> tensor id -> comm op handle
        self.comm_op_handles = defaultdict(dict)
        # map stage id -> p2p op handle
        self.p2p_op_handles = defaultdict(list)

        self.tracing = False
        self.update_step = 0
        self.trace_events = dict()
        self.trace_data = defaultdict(list)

        from .piper_utils import piper_metadata

        piper_metadata.actor_self = self

    def reset_p2p_states(self):
        self.next_p2p_idx = 0
        self.executed_p2ps = set()
        self.p2p_cache = dict()

    def get_trace_data(self) -> dict:
        return self.global_rank, self.trace_data

    def clear_trace_data(self) -> None:
        self.trace_data.clear()

    def set_tracing(self, enabled: bool) -> None:
        self.tracing = enabled
        self.logger.info(
            f"Actor {self.global_rank}: Tracing {'enabled' if enabled else 'disabled'}"
        )

    def start_mem_tracing(self) -> None:
        torch.cuda.memory._record_memory_history()

    def stop_mem_tracing(self) -> None:
        torch.cuda.memory._dump_snapshot(
            f"actor{self.global_rank}_memory_snapshot_mb4_gpipe.pickle"
        )
        self.logger.info(
            f"Saved memory snapshot to actor{self.global_rank}_memory_snapshot_mb4_gpipe.pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()

    def get_peak_memory(self):
        return torch.cuda.max_memory_allocated() / (1024**3)

    def get_stage_parameter_vectors(self):
        stage_to_vector = {}
        for stage_id in sorted(self.forward_args.keys()):
            stage_to_vector[stage_id] = self._stage_parameter_vector(stage_id)
        return stage_to_vector


    def _stage_parameter_vector(self, stage_id: int):
        params = [
            tensor.detach().cpu().reshape(-1)
            for tensor in self.forward_args[stage_id]
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad
        ]
        return torch.cat(params) if params else torch.empty(0)

    def _maybe_dump_parity_data(self, filename: str, payload: dict):
        dump_dir = os.environ.get("PIPER_PARITY_DUMP_DIR")
        if not dump_dir:
            return
        output_path = Path(dump_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path / filename)

    def load_input(self, inputs):
        self.inputs = [inp.to(self.device) for inp in inputs]
        self.logger.debug(f"Actor {self.global_rank} loaded inputs {len(self.inputs)}")

    def load_labels(self, labels):
        self.labels = labels.to(self.device)
        self.logger.debug(f"Actor {self.global_rank} loaded labels {self.labels.shape}")

    def _start_timing(self, stream, label):
        if self.tracing:
            if label not in self.trace_events:
                self.trace_events[label] = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
            start, _ = self.trace_events[label]
            start.record(stream)

    def _stop_timing(self, stream, label):
        if self.tracing:
            if label in self.trace_events:
                start, stop = self.trace_events[label]
                stop.record(stream)
                stream.synchronize()
                self.trace_data[label].append(start.elapsed_time(stop))

    def _join_process_groups(self):
        master_addr = os.environ.get("PIPER_MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("PIPER_MASTER_PORT", "10000")
        init_method = f"tcp://{master_addr}:{master_port}"

        dist.init_process_group(
            "nccl",
            init_method=init_method,
            rank=self.global_rank,
            world_size=self.world_size,
        )
        self.logger.debug(
            f"Actor {self.global_rank} has GPU {os.environ['CUDA_VISIBLE_DEVICES']}, joined the global process group"
        )

        if self.dp_degree > 1:
            self._join_dp_process_group()
        if self.pp_degree > 1:
            self._join_pp_process_group()

    def _join_dp_process_group(self):
        num_dp_groups = self.world_size // self.dp_degree
        for dp_group_id in range(num_dp_groups):
            group_ranks = [
                (dp_group_id + num_dp_groups * i) for i in range(self.dp_degree)
            ]
            process_group = dist.new_group(ranks=group_ranks, backend="nccl")
            if self.global_rank % num_dp_groups == dp_group_id:
                self.dp_group = process_group
                self.logger.info(
                    f"Global rank {self.global_rank} joined its dp group {dp_group_id} along with ranks {group_ranks}"
                )

    def _join_pp_process_group(self):
        num_pp_groups = self.world_size // self.pp_degree
        for pp_group_id in range(num_pp_groups):
            group_ranks = [
                (pp_group_id * self.pp_degree + i) for i in range(self.pp_degree)
            ]
            process_group = dist.new_group(ranks=group_ranks, backend="nccl")
            if self.global_rank // self.pp_degree == pp_group_id:
                self.pp_group = process_group
                self.logger.info(
                    f"Global rank {self.global_rank} joined its pp group {pp_group_id} along with ranks {group_ranks}"
                )

    def _prepare_comm_ops(self, stage_id, comm_ops, graphargs, ids):
        def hook_maker(tensor_id):
            def post_backward_hook(grad):
                self.comm_op_status[stage_id][tensor_id] += 1
                self.logger.debug(
                    f"Updating tensor status dp_rank: {self.dp_rank}, tensor={comm_op.name}, shape={grad.shape}, status={self.comm_op_status[stage_id][tensor_id]}"
                )
                return grad

            return post_backward_hook

        comm_op_ids = [op.tensor_id for op in comm_ops]
        for t, i in zip(graphargs, ids):
            if i in comm_op_ids:
                comm_op = comm_ops[comm_op_ids.index(i)]
                comm_op.tensor = t
                match comm_op.op:
                    case "allreduce":
                        if comm_op.pass_type == "backward" and comm_op.dep == "post":
                            tensor_id = i
                            tensor_shape = t.shape
                            t.register_post_accumulate_grad_hook(hook_maker(tensor_id))
                        else:
                            raise ValueError(
                                f"Unknown comm op type or dependency: {comm_op.op} {comm_op.dep}"
                            )
                    case _:
                        raise ValueError(
                            f"Unknown communication operation: {comm_op.op}"
                        )

    def _comm_loop(self):
        stage_id = self.stage_id
        # look into: nccl priority for sync vs async ops, stream priority
        self.logger.info(
            f"Running comm loop for stage {stage_id} on actor {self.global_rank}"
        )
        for comm_op in self.comm_ops[stage_id]:
            done = False
            while not done:
                if (
                    comm_op.op == "allreduce"
                    and comm_op.pass_type == "backward"
                    and comm_op.dep == "post"
                    and comm_op.group == "dp"
                ):
                    # perform the all reduce when all gradients have been accumulated
                    if self.comm_op_status[stage_id][comm_op.tensor_id] == self.num_mbs:
                        with torch.cuda.stream(self.comm_stream):
                            handle = dist.all_reduce(
                                comm_op.tensor,
                                op=dist.ReduceOp.AVG,
                                group=self.dp_group,
                                async_op=True,
                            )
                        self.logger.debug(
                            f"Allreduce on dp_rank: {self.dp_rank}, tensor={comm_op.name}, shape={comm_op.tensor.shape}"
                        )
                        self.comm_op_status[stage_id][comm_op.tensor_id] = 0
                        self.comm_op_handles[stage_id][comm_op.tensor_id] = handle
                        done = True
                else:
                    raise ValueError(f"Unknown comm op: {comm_op}")
        self.logger.debug(
            f"Completed comm loop for stage {stage_id} on actor {self.global_rank}"
        )

    def _wait_for_comm_ops(self):
        self.logger.debug(f"Actor {self.global_rank} waiting for comm ops")
        for stage_id, comm_ops in self.comm_ops.items():
            for comm_op in comm_ops:
                self.logger.debug(
                    f"Waiting for comm op to be launched dp_rank: {self.dp_rank}, tensor={comm_op.name}, shape={comm_op.tensor.shape}"
                )
                done = False
                while not done:
                    if comm_op.tensor_id in self.comm_op_handles[stage_id]:
                        self.logger.debug(
                            f"Waiting for comm op to be finished dp_rank: {self.dp_rank}, tensor={comm_op.name}, shape={comm_op.tensor.shape}"
                        )
                        self.comm_op_handles[stage_id][comm_op.tensor_id].wait()
                        done = True

    def _wait_for_p2p_ops(self):
        self.logger.debug(f"Actor {self.global_rank} waiting for p2p ops")
        for stage_id, handles in self.p2p_op_handles.items():
            for handle in handles:
                handle.wait()

    def _load_stage(
        self, stage_id: int, gm_data, comm_ops, forward_args, ids, input_idxs
    ):
        self.logger.info(f"Loading stage {stage_id} graph on actor {self.global_rank}")

        # compile the graph with the given graphargs
        gm = _deserialize_graphmodule(gm_data)

        # Store GraphModule reference
        self.graph_modules[stage_id] = gm
        self.forward_fns[stage_id] = gm.forward

        # place parameters on the device
        def move_to_device(idx, arg):
            if arg.requires_grad:
                return arg.to(self.device).detach().requires_grad_(True)
            else:
                return arg.to(self.device)

        forward_args = list(map(move_to_device, range(len(forward_args)), forward_args))

        # prepare tensors with comm ops
        if not self.naive_gradient_sync and self.dp_degree > 1:
            self._prepare_comm_ops(stage_id, comm_ops, forward_args, ids)
        self.comm_ops[stage_id] = list(reversed(comm_ops))

        # save parameters
        self.input_idxs[stage_id] = input_idxs
        self.forward_args[stage_id] = forward_args
        self.stage_id = stage_id

        self._maybe_dump_parity_data(
            f"rank{self.global_rank}_stage{stage_id}_init.pt",
            {"global_rank": self.global_rank, "stage_id": stage_id, "vector": self._stage_parameter_vector(stage_id)},
        )

        for i in self.input_idxs[stage_id]:
            self.forward_input_meta[stage_id][i] = (
                forward_args[i].shape,
                forward_args[i].dtype,
                forward_args[i].requires_grad,
            )

        # add the parameters to the optimizer for this stage
        params = [param for param in forward_args if param.requires_grad]
        if stage_id not in self.optims:
            self.optims[stage_id] = self.optim_class(params)
        else:
            self.optims[stage_id].add_param_group({"params": params})

        for i in self.input_idxs[stage_id]:
            self.forward_args[stage_id][i] = None

        del gm_data

    def _exec_p2p_op(
        self, src_stage: int, dst_stage: int, mb_idx: int, is_sender: bool
    ):
        logger.debug(
            f"[DEBUG] Actor {self.global_rank} executing p2p op: {src_stage} -> {dst_stage}, mb {mb_idx}, is_sender={is_sender}"
        )
        p2p_op = (src_stage, dst_stage, mb_idx, is_sender)
        if p2p_op in self.executed_p2ps:
            logger.debug(
                f"P2P op {p2p_op} already executed on actor {self.global_rank}, skipping"
            )
            return

        op_idx = self.next_p2p_idx
        while op_idx < len(self.p2p_schedule):
            op = self.p2p_schedule[op_idx]
            if op == p2p_op:
                break
            op_idx += 1
        assert op_idx < len(
            self.p2p_schedule
        ), f"P2P op {p2p_op} not found in schedule for actor {self.global_rank}"

        # Execute everything before and including the given p2p op
        for idx in range(self.next_p2p_idx, op_idx + 1):
            op = self.p2p_schedule[idx]
            src_stage, dst_stage, mb_idx, is_sender = op
            is_fwd = src_stage == dst_stage - 1
            is_bwd = src_stage == dst_stage + 1
            assert is_fwd or is_bwd
            is_recver = not is_sender

            op_name = "P2P(unknown)"
            if is_fwd and is_sender:
                self._exec_fwd_send(src_stage, mb_idx)
                op_name = (
                    f"P2P(fwd_send, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            elif is_fwd and is_recver:
                self._exec_fwd_recv(dst_stage, mb_idx)
                op_name = (
                    f"P2P(fwd_recv, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            elif is_bwd and is_sender:
                self._exec_bwd_send(src_stage, mb_idx)
                op_name = (
                    f"P2P(bwd_send, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            elif is_bwd and is_recver:
                self._exec_bwd_recv(dst_stage, mb_idx)
                op_name = (
                    f"P2P(bwd_recv, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            else:
                raise ValueError(f"Invalid p2p op: {op}")

            self.executed_p2ps.add(op)
            logger.debug(f"Executed {op_name} on actor {self.global_rank}")

        self.next_p2p_idx = op_idx + 1

    def _exec_fwd_recv(self, stage_id: int, mb_idx: int):
        if stage_id == 0:
            return

        p2p_op = (stage_id - 1, stage_id, mb_idx, False)
        assert p2p_op not in self.p2p_cache

        inputs_to_recv = []
        for i in self.input_idxs[stage_id]:
            shape, dtype, requires_grad = self.forward_input_meta[stage_id][i]
            inputs_to_recv.append(
                torch.empty(
                    shape, dtype=dtype, requires_grad=requires_grad, device=self.device
                )
            )

        # For non-first stages, receive input tensors from the previous stage
        pp_rank = piper_metadata.stage_to_device[stage_id - 1]
        global_src_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        self.logger.debug(
            f"Dispatch fwd p2p recv on {self.global_rank} from {global_src_rank}, op: ({stage_id-1} -> {stage_id}, mb {mb_idx})"
        )
        self._start_timing(self.p2p_stream, "fwd_p2p_recv")
        with torch.cuda.stream(self.p2p_stream):
            for i in self.input_idxs[stage_id]:
                dist.recv(
                    inputs_to_recv[i],
                    src=global_src_rank,
                    group=self.pp_group,
                )
        self._stop_timing(self.p2p_stream, "fwd_p2p_recv")
        self.logger.debug(
            f"Completed fwd p2p recv on {self.global_rank} from {global_src_rank}, op: ({stage_id-1} -> {stage_id}, mb {mb_idx})"
        )

        self.p2p_cache[p2p_op] = inputs_to_recv

    def _exec_fwd_send(self, stage_id: int, mb_idx: int):
        if stage_id == self.num_stages - 1:
            return

        p2p_op = (stage_id, stage_id + 1, mb_idx, True)
        output = self.p2p_cache.pop(p2p_op)
        # For non-final stages, send output tensors to the next stage
        pp_rank = piper_metadata.stage_to_device[stage_id + 1]
        global_dst_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        self.logger.debug(
            f"Dispatch fwd p2p send on {self.global_rank} to {global_dst_rank}, op: ({stage_id} -> {stage_id+1}, mb {mb_idx})"
        )
        self._start_timing(self.p2p_stream, "fwd_p2p_send")
        with torch.cuda.stream(self.p2p_stream):
            for i in range(len(output)):
                dist.send(output[i], dst=global_dst_rank, group=self.pp_group)
        self._stop_timing(self.p2p_stream, "fwd_p2p_send")
        self.logger.debug(
            f"Completed fwd p2p send on {self.global_rank} to {global_dst_rank}, op: ({stage_id} -> {stage_id+1}, mb {mb_idx})"
        )

    def _exec_bwd_recv(self, stage_id: int, mb_idx: int):
        if stage_id >= self.num_stages - 1:
            return

        out_activation = self.out_activation[stage_id][mb_idx]
        # For non-final stages, recieve input gradients from the subsequent backward pass
        input_grad = torch.empty_like(out_activation)
        pp_rank = piper_metadata.stage_to_device[stage_id + 1]
        global_src_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        self.logger.debug(
            f"Dispatch bwd p2p recv on {self.global_rank} from {global_src_rank}, op: ({stage_id+1} -> {stage_id}, mb {mb_idx})"
        )
        self._start_timing(self.p2p_stream, "bwd_p2p_recv")
        with torch.cuda.stream(self.p2p_stream):
            dist.recv(input_grad, src=global_src_rank, group=self.pp_group)
        self._stop_timing(self.p2p_stream, "bwd_p2p_recv")
        self.logger.debug(
            f"Completed bwd p2p recv on {self.global_rank} from {global_src_rank}, op: ({stage_id+1} -> {stage_id}, mb {mb_idx})"
        )

        out_activation.backward(gradient=input_grad)

    def _exec_bwd_send(self, stage_id: int, mb_idx: int):
        if stage_id <= 0:
            return

        # For non-first stages, send output gradients to the previous backward stage
        output_grad = self.inp_activation[stage_id][mb_idx].grad
        assert output_grad is not None
        pp_rank = piper_metadata.stage_to_device[stage_id - 1]
        global_src_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        self.logger.debug(
            f"Dispatch bwd p2p send on {self.global_rank} to {global_src_rank}, op: ({stage_id} -> {stage_id-1}, mb {mb_idx})"
        )
        self._start_timing(self.p2p_stream, "bwd_p2p_send")
        with torch.cuda.stream(self.p2p_stream):
            dist.send(output_grad, dst=global_src_rank, group=self.pp_group)
        self._stop_timing(self.p2p_stream, "bwd_p2p_send")
        self.logger.debug(
            f"Completed bwd p2p send on {self.global_rank} to {global_src_rank}, op: ({stage_id} -> {stage_id-1}, mb {mb_idx})"
        )

        self.inp_activation[stage_id][mb_idx] = None

    def _forward(self, stage_id: int, mb_idx: int):

        self.logger.debug(
            f"Calling forward {stage_id} mb {mb_idx} on actor {self.global_rank}"
        )

        if stage_id == 0:
            # For the first stage, load input tensors from self.inputs
            for i, inp in zip(self.input_idxs[stage_id], self.inputs):
                self.forward_args[stage_id][i] = inp
        else:
            self._exec_p2p_op(stage_id - 1, stage_id, mb_idx, False)
            inputs_from_prev_stage = self.p2p_cache.pop(
                (stage_id - 1, stage_id, mb_idx, False)
            )
            for i, tensor in zip(self.input_idxs[stage_id], inputs_from_prev_stage):
                self.forward_args[stage_id][i] = tensor

            # save first input that requires grad as input activation
            inp_with_grad = [
                self.forward_args[stage_id][i]
                for i in self.input_idxs[stage_id]
                if self.forward_args[stage_id][i].requires_grad
            ]
            assert (
                len(inp_with_grad) == 1
            ), "Exactly one input per stage should require a gradient"
            self.logger.debug(
                f"Saving input activation {inp_with_grad[0].shape} for stage {stage_id} mb {mb_idx}"
            )
            self.inp_activation[stage_id][mb_idx] = inp_with_grad[0]

        # Run the forward pass
        output = self.forward_fns[stage_id](*self.forward_args[stage_id])

        # Save first output that requires grad as output activation
        # TODO: support multiple outputs
        out_with_grad = [out for out in output if out.requires_grad]
        assert (
            len(out_with_grad) == 1
        ), "Piper only supports one output per subgraph with requires_grad"
        self.logger.debug(
            f"Saving output activation {out_with_grad[0].shape} for stage {stage_id} mb {mb_idx}"
        )
        self.out_activation[stage_id][mb_idx] = out_with_grad[0]

        # clear the input tensors
        # for i in self.input_idxs[stage_id]:
        #     self.forward_args[stage_id][i] = torch.empty_like(self.forward_args[stage_id][i])

        for i in self.input_idxs[stage_id]:
            self.forward_args[stage_id][i] = None

        if stage_id < self.num_stages - 1:
            send_p2p_op = (stage_id, stage_id + 1, mb_idx, True)
            assert send_p2p_op not in self.p2p_cache
            self.p2p_cache[send_p2p_op] = output
            self._exec_p2p_op(stage_id, stage_id + 1, mb_idx, True)

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.debug(
            f"Forward {stage_id} mb {mb_idx} on actor {self.global_rank} returning {[out.shape for out in output]}"
        )
        torch.cuda.synchronize()

    # @ray.method(tensor_transport="nccl")
    def _backward(self, stage_id: int, mb_idx: int, loss_fn=None):

        self.logger.debug(
            f"Calling backward {stage_id} mb {mb_idx} on actor {self.global_rank}"
        )

        out_activation = self.out_activation[stage_id][mb_idx]

        if stage_id < self.num_stages - 1:
            self._exec_p2p_op(stage_id + 1, stage_id, mb_idx, False)
        else:
            # For the final stage, wait for the forward pass to complete
            assert loss_fn is not None
            labels = self.labels
            assert out_activation.shape == labels.shape
            loss = loss_fn(out_activation, labels)
            loss.backward()
            self.loss.append(loss.item())

        del out_activation

        if stage_id > 0:
            self._exec_p2p_op(stage_id, stage_id - 1, mb_idx, True)

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.cuda.synchronize()

    def _synchronize_gradients(self):
        self.logger.debug(f"Actor {self.global_rank} synchronizing gradients")
        # Iterate over all stages on this actor and synchronize their parameters
        for stage_id, parameters in self.forward_args.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    dist.all_reduce(
                        param.grad, op=dist.ReduceOp.AVG, group=self.dp_group
                    )

    def _update(self):
        self.logger.debug(f"Actor {self.global_rank} waiting for backward sync events")

        # if dp degree > 1, make sure all gradients are synchronized before optimizer step
        # TODO: this does not allow overlapping with the optimizer step
        if self.pp_degree > 1:
            self._wait_for_p2p_ops()

        if self.dp_degree > 1:
            if self.naive_gradient_sync:
                self._synchronize_gradients()
            else:
                self._wait_for_comm_ops()

        # step the optimizer for each stage
        for _, optim in self.optims.items():
            optim.step()
            optim.zero_grad()
        losses = self.loss
        self.loss.clear()

        self._maybe_dump_parity_data(
            f"rank{self.global_rank}_step{self.update_step}.pt",
            {
                "global_rank": self.global_rank,
                "step": self.update_step,
                "losses": losses,
                "stage_to_vector": self.get_stage_parameter_vectors(),
            },
        )
        self.update_step += 1

        torch.cuda.synchronize()

        return losses
