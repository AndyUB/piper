"""
ZeRO-1 with bucketed reduce-scatter for Piper.

Each trainable parameter is divided into dp_degree contiguous shards; each DP rank
owns and optimises one shard. Parameters are grouped into 25 MB gradient-sync buckets
ordered by reverse forward-use (approximating backward arrival order), enabling
overlap between communication and backward computation.

Per-step communication pipeline (per bucket, pipelined across buckets):
  1. reduce_scatter gradient flat buffer → local grad shard   (async, on comm_stream)
  2. Optimizer step on local param shard                      (comp_stream)
  3. all_gather local param shard → flat param buffer         (async, on comm_stream)
  4. Scatter flat param buffer back to p.data

See docs/zero1_design.md for the full design rationale.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Type

import torch
import torch.distributed as dist
import torch.fx
from torch.nn import Parameter
from torch.optim import Optimizer

from .piper_utils import create_logger, LOG_LEVEL

logger = create_logger("piper_zero1", LOG_LEVEL)

# Default bucket size in bytes — matches PyTorch DDP's default of 25 MB.
BUCKET_SIZE_BYTES_DEFAULT: int = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# FX graph analysis
# ---------------------------------------------------------------------------

def _get_param_forward_order(
    gm: torch.fx.GraphModule,
    param_idxs: list[int],
    params: list[Parameter],
) -> list[Parameter]:
    """
    Return ``params`` sorted by the first position in ``gm``'s FX graph where
    each parameter placeholder is used as an input to a computation node.

    Parameters used earlier in the forward graph (lower node index) come first.
    Parameters never referenced as inputs are appended last in original order.

    Args:
        gm: The FX GraphModule for this pipeline stage.
        param_idxs: Indices of parameter placeholders within the graph's full
            placeholder list (matches the positional order of ``forward_args``).
        params: Trainable parameter tensors, one per entry in ``param_idxs``.

    Returns:
        ``params`` in forward-use order.
    """
    placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    param_idx_set = set(param_idxs)

    # Build a set of placeholder nodes that correspond to parameters.
    param_ph_nodes: set[torch.fx.Node] = {
        placeholder_nodes[i]
        for i in param_idxs
        if i < len(placeholder_nodes)
    }

    # Walk graph nodes to find the first computation node that uses each param.
    first_use_pos: dict[torch.fx.Node, int] = {}
    for node_pos, node in enumerate(gm.graph.nodes):
        if node.op == "placeholder":
            continue
        for inp in node.all_input_nodes:
            if inp in param_ph_nodes and inp not in first_use_pos:
                first_use_pos[inp] = node_pos

    # Sort params by first-use position.
    param_ph_by_idx = {
        i: placeholder_nodes[i]
        for i in param_idxs
        if i < len(placeholder_nodes)
    }
    pairs: list[tuple[Parameter, float]] = []
    for idx, p in zip(param_idxs, params):
        ph = param_ph_by_idx.get(idx)
        pos = first_use_pos.get(ph, math.inf) if ph is not None else math.inf
        pairs.append((p, pos))

    pairs.sort(key=lambda x: x[1])
    return [p for p, _ in pairs]


# ---------------------------------------------------------------------------
# Bucket
# ---------------------------------------------------------------------------

class ZeROOneBucket:
    """
    One gradient-synchronisation bucket for ZeRO-1.

    Holds a flat parameter buffer and a flat gradient buffer covering all params
    in the bucket. The local shard (owned by this DP rank) is a contiguous slice
    of those flat buffers. One optimizer instance manages the local param shard.

    Lifecycle per training step:
      ``on_grad_accumulated`` → (when bucket full) ``launch_reduce_scatter``
      → (in finalize_step) ``optimizer_step`` → ``launch_all_gather``
      → ``copy_flat_to_params`` → ``reset_step_state``
    """

    def __init__(
        self,
        bucket_id: int,
        params: list[Parameter],
        dp_rank: int,
        dp_degree: int,
        dp_group: dist.ProcessGroup,
        device: str,
        dtype: torch.dtype,
        optim_class: Callable[[list[Parameter]], Optimizer],
        num_mbs: int,
    ) -> None:
        self.bucket_id = bucket_id
        self.params = params
        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.dp_group = dp_group
        self.num_mbs = num_mbs

        # ------------------------------------------------------------------
        # Flat buffer layout: [p0 elems | p1 elems | ... | padding]
        # ------------------------------------------------------------------
        self.param_offsets: list[int] = []
        total_numel = 0
        for p in params:
            self.param_offsets.append(total_numel)
            total_numel += p.numel()

        # Pad so that padded_numel is divisible by dp_degree (for equal shards).
        self.shard_numel: int = math.ceil(total_numel / dp_degree)
        self.padded_numel: int = self.shard_numel * dp_degree
        self.shard_start: int = dp_rank * self.shard_numel
        self.shard_end: int = self.shard_start + self.shard_numel

        # Persistent flat buffers (allocated once, reused each step).
        self.flat_param_buf: torch.Tensor = torch.empty(
            self.padded_numel, device=device, dtype=dtype
        )
        self.flat_grad_buf: torch.Tensor = torch.zeros(
            self.padded_numel, device=device, dtype=dtype
        )

        # Initialise flat_param_buf from current parameter values.
        for p, off in zip(params, self.param_offsets):
            self.flat_param_buf[off : off + p.numel()].copy_(p.data.view(-1))

        # Local shard: a separate tensor (not a view of flat_param_buf) to
        # avoid aliasing when all_gather writes into flat_param_buf.
        self.local_param_shard: Parameter = Parameter(
            self.flat_param_buf[self.shard_start : self.shard_end].clone(),
            requires_grad=True,
        )
        # Receives the reduce-scattered gradient for the local shard.
        self.local_grad_shard: torch.Tensor = torch.zeros(
            self.shard_numel, device=device, dtype=dtype
        )

        # One optimizer per bucket, managing only the local param shard.
        self.optimizer: Optimizer = optim_class([self.local_param_shard])

        # ------------------------------------------------------------------
        # Per-step state (reset by reset_step_state)
        # ------------------------------------------------------------------
        # Counts how many microbatch backward passes have touched each param.
        self._mb_accumulated: dict[int, int] = {id(p): 0 for p in params}
        # Number of params that have fully accumulated all num_mbs microbatches.
        self._ready_param_count: int = 0

        # Async collective handles (None until launched).
        self.reduce_scatter_handle: Optional[dist.Work] = None
        self.all_gather_handle: Optional[dist.Work] = None

        logger.debug(
            f"ZeROOneBucket {bucket_id}: {len(params)} params, "
            f"{total_numel} elems, shard [{self.shard_start}:{self.shard_end}]"
        )

    # ------------------------------------------------------------------
    # Per-step trigger (called from grad hook)
    # ------------------------------------------------------------------

    def on_grad_accumulated(self, param: Parameter) -> bool:
        """
        Record one microbatch gradient accumulation for *param*.

        Returns ``True`` when every parameter in this bucket has accumulated
        gradients from all ``num_mbs`` microbatches, signalling that the bucket
        is ready for reduce_scatter.
        """
        pid = id(param)
        self._mb_accumulated[pid] += 1
        if self._mb_accumulated[pid] == self.num_mbs:
            self._ready_param_count += 1
        return self._ready_param_count == len(self.params)

    # ------------------------------------------------------------------
    # Communication and computation
    # ------------------------------------------------------------------

    def launch_reduce_scatter(
        self,
        comm_stream: torch.cuda.Stream,
        comp_stream: torch.cuda.Stream,
    ) -> dist.Work:
        """
        Copy accumulated per-param gradients into the flat gradient buffer, then
        launch an async ``reduce_scatter_tensor`` on *comm_stream*.

        After completion, ``self.local_grad_shard`` holds the averaged gradient
        for this rank's shard of the bucket.

        Returns the ``dist.Work`` handle for the caller to wait on.
        """
        comm_stream.wait_stream(comp_stream)
        with torch.cuda.stream(comm_stream):
            for p, off in zip(self.params, self.param_offsets):
                if p.grad is not None:
                    self.flat_grad_buf[off : off + p.numel()].copy_(p.grad.view(-1))
                else:
                    self.flat_grad_buf[off : off + p.numel()].zero_()
            handle = dist.reduce_scatter_tensor(
                self.local_grad_shard,
                self.flat_grad_buf,
                op=dist.ReduceOp.AVG,
                group=self.dp_group,
                async_op=True,
            )
        return handle

    def optimizer_step(self) -> None:
        """
        Attach the reduce-scattered gradient to ``local_param_shard`` and run
        the optimizer.

        Must be called after ``reduce_scatter_handle.wait()``.
        """
        if self.local_param_shard.grad is None:
            self.local_param_shard.grad = self.local_grad_shard.clone()
        else:
            self.local_param_shard.grad.copy_(self.local_grad_shard)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def launch_all_gather(
        self,
        comm_stream: torch.cuda.Stream,
        comp_stream: torch.cuda.Stream,
    ) -> dist.Work:
        """
        Launch an async ``all_gather_into_tensor`` that reconstructs
        ``flat_param_buf`` from all ranks' updated ``local_param_shard``.

        Must be called after ``optimizer_step()``.
        Returns the ``dist.Work`` handle.
        """
        comp_stream.wait_stream(comm_stream)
        with torch.cuda.stream(comm_stream):
            handle = dist.all_gather_into_tensor(
                self.flat_param_buf,
                self.local_param_shard.data,
                group=self.dp_group,
                async_op=True,
            )
        return handle

    def copy_flat_to_params(self) -> None:
        """
        Scatter ``flat_param_buf`` back into each parameter's ``.data``.

        Must be called after ``all_gather_handle.wait()``.
        """
        for p, off in zip(self.params, self.param_offsets):
            p.data.copy_(
                self.flat_param_buf[off : off + p.numel()].view(p.shape)
            )

    def reset_step_state(self) -> None:
        """Reset per-step counters and handles for the next training step."""
        for pid in self._mb_accumulated:
            self._mb_accumulated[pid] = 0
        self._ready_param_count = 0
        self.reduce_scatter_handle = None
        self.all_gather_handle = None
        self.flat_grad_buf.zero_()

    @property
    def size_bytes(self) -> int:
        """Total byte size of parameters in this bucket."""
        return sum(p.numel() * p.element_size() for p in self.params)


# ---------------------------------------------------------------------------
# State manager
# ---------------------------------------------------------------------------

class ZeROOneState:
    """
    Manages ZeRO-1 optimizer state partitioning for one pipeline stage.

    At construction time, parameters are grouped into gradient-sync buckets in
    reverse forward-use order. Post-accumulate-grad hooks are registered on every
    parameter; each hook decrements a per-bucket counter and launches an async
    reduce_scatter when the bucket's counter reaches zero.

    At the end of each training step, call ``finalize_step`` to complete the
    pipelined reduce_scatter → optimizer step → all_gather sequence.

    Args:
        all_params: All trainable parameters for this stage.
        params_in_forward_order: Same parameters sorted by first use in the
            FX graph forward pass (from ``_get_param_forward_order``).
        dp_rank: Data-parallel rank of this actor.
        dp_degree: Total number of data-parallel replicas.
        dp_group: ``torch.distributed`` process group for this DP replica set.
        device: CUDA device string for the actor.
        num_mbs: Number of microbatches per training step.
        optim_class: Callable ``(params) -> Optimizer`` used per bucket.
        comm_stream: CUDA stream for collective communication ops.
        comp_stream: CUDA stream for computation (backward / optimizer).
        bucket_size_bytes: Maximum byte size per bucket (default 25 MB).
    """

    def __init__(
        self,
        all_params: list[Parameter],
        params_in_forward_order: list[Parameter],
        dp_rank: int,
        dp_degree: int,
        dp_group: dist.ProcessGroup,
        device: str,
        num_mbs: int,
        optim_class: Callable[[list[Parameter]], Optimizer],
        comm_stream: torch.cuda.Stream,
        comp_stream: torch.cuda.Stream,
        bucket_size_bytes: int = BUCKET_SIZE_BYTES_DEFAULT,
    ) -> None:
        self.all_params = all_params
        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.dp_group = dp_group
        self._comm_stream = comm_stream
        self._comp_stream = comp_stream

        # Build buckets in reverse forward order ≈ backward arrival order.
        self._buckets: list[ZeROOneBucket] = _build_buckets(
            params_in_bwd_order=list(reversed(params_in_forward_order)),
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            dp_group=dp_group,
            device=device,
            num_mbs=num_mbs,
            optim_class=optim_class,
            bucket_size_bytes=bucket_size_bytes,
        )

        # Fast lookup: parameter id → owning bucket.
        self._param_to_bucket: dict[int, ZeROOneBucket] = {
            id(p): bucket
            for bucket in self._buckets
            for p in bucket.params
        }

        self._setup_grad_hooks()

        logger.debug(
            f"ZeROOneState: {len(self._buckets)} buckets, "
            f"{len(all_params)} params, dp_rank={dp_rank}/{dp_degree}"
        )

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _setup_grad_hooks(self) -> None:
        """Register a post-accumulate-grad hook on every tracked parameter."""
        for p in self.all_params:
            if id(p) in self._param_to_bucket:
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, param: Parameter) -> Callable[[Parameter], None]:
        """Return a closure that fires when *param*'s gradient is accumulated."""
        bucket = self._param_to_bucket[id(param)]

        def _hook(_param: Parameter) -> None:
            if bucket.on_grad_accumulated(_param):
                # All params in bucket have full gradients: launch reduce_scatter.
                bucket.reduce_scatter_handle = bucket.launch_reduce_scatter(
                    self._comm_stream, self._comp_stream
                )
                logger.debug(
                    f"Launched reduce_scatter for bucket {bucket.bucket_id}"
                )

        return _hook

    # ------------------------------------------------------------------
    # End-of-step pipeline
    # ------------------------------------------------------------------

    def finalize_step(
        self,
        comm_stream: torch.cuda.Stream,
        comp_stream: torch.cuda.Stream,
    ) -> None:
        """
        Complete the pipelined reduce_scatter → optimizer → all_gather sequence
        for all buckets, then reset state for the next step.

        Should be called in ``_update`` after all backward passes are complete.

        Pipeline structure:
          For bucket k: wait rs_k → optim_k → launch ag_k
          The all_gather of bucket k-1 overlaps with optim_k and launch_ag_k.
        """
        # Ensure every bucket has a reduce_scatter in flight (handles buckets
        # whose hooks were never triggered, e.g. params with no gradient).
        for bucket in self._buckets:
            if bucket.reduce_scatter_handle is None:
                bucket.reduce_scatter_handle = bucket.launch_reduce_scatter(
                    comm_stream, comp_stream
                )

        # Pipeline: per-bucket wait → optimizer → async all_gather.
        for bucket in self._buckets:
            bucket.reduce_scatter_handle.wait()
            bucket.optimizer_step()
            bucket.all_gather_handle = bucket.launch_all_gather(
                comm_stream, comp_stream
            )

        # Wait for all all_gathers and write results back into p.data.
        for bucket in self._buckets:
            bucket.all_gather_handle.wait()
            bucket.copy_flat_to_params()

        # Zero all parameter gradients and reset per-step state.
        for p in self.all_params:
            p.grad = None
        for bucket in self._buckets:
            bucket.reset_step_state()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def num_buckets(self) -> int:
        return len(self._buckets)

    def bucket_sizes(self) -> list[int]:
        return [b.size_bytes for b in self._buckets]

    def __repr__(self) -> str:
        return (
            f"ZeROOneState(dp_rank={self.dp_rank}/{self.dp_degree}, "
            f"buckets={self.num_buckets()}, "
            f"params={len(self.all_params)})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_buckets(
    params_in_bwd_order: list[Parameter],
    dp_rank: int,
    dp_degree: int,
    dp_group: dist.ProcessGroup,
    device: str,
    num_mbs: int,
    optim_class: Callable[[list[Parameter]], Optimizer],
    bucket_size_bytes: int,
) -> list[ZeROOneBucket]:
    """
    Fill buckets greedily: add parameters in *params_in_bwd_order* until a
    bucket would exceed *bucket_size_bytes*, then start a new bucket.

    The first parameter always starts a new bucket regardless of size (no
    parameter is skipped).
    """
    buckets: list[ZeROOneBucket] = []
    current: list[Parameter] = []
    current_bytes = 0

    for p in params_in_bwd_order:
        p_bytes = p.numel() * p.element_size()
        if current and current_bytes + p_bytes > bucket_size_bytes:
            dtype = _bucket_dtype(current)
            buckets.append(
                ZeROOneBucket(
                    len(buckets), current,
                    dp_rank, dp_degree, dp_group,
                    device, dtype, optim_class, num_mbs,
                )
            )
            current = []
            current_bytes = 0
        current.append(p)
        current_bytes += p_bytes

    if current:
        dtype = _bucket_dtype(current)
        buckets.append(
            ZeROOneBucket(
                len(buckets), current,
                dp_rank, dp_degree, dp_group,
                device, dtype, optim_class, num_mbs,
            )
        )

    return buckets


def _bucket_dtype(params: list[Parameter]) -> torch.dtype:
    """Return the dtype of the first parameter (all params in a bucket share dtype)."""
    return params[0].dtype
