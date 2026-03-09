"""
ZeRO-1 (Zero Redundancy Optimizer Stage 1) for Piper.

ZeRO-1 partitions optimizer states across data-parallel ranks so that each rank
stores and updates optimizer state for only 1/dp_degree of the parameters.
Parameters and gradients remain fully replicated during forward/backward passes.

Communication per training step:
  1. Backward pass computes full gradients on every DP rank (as in DDP).
  2. All-reduce gradients across DP ranks so every rank has the correct averaged
     gradient for every parameter.
  3. Each rank runs the optimizer step only for its own parameter partition.
  4. Each rank broadcasts its updated parameters to all other DP ranks so that
     every rank has a complete, up-to-date copy before the next forward pass.

Memory savings vs. DDP (per DP rank):
  Optimizer states are reduced by a factor of dp_degree.
  For Adam, this saves ~(1 - 1/dp_degree) * 12 bytes per parameter element
  (fp32 master weight + momentum + variance).

See docs/zero1_design.md for the full design rationale.
"""

from __future__ import annotations

from typing import Optional, Type

import torch
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import Optimizer

from .piper_utils import create_logger, LOG_LEVEL

logger = create_logger("piper_zero1", LOG_LEVEL)


class ZeROOneState:
    """
    Manages ZeRO-1 optimizer state partitioning for a single pipeline stage.

    Each DP rank owns a disjoint subset of the stage's parameters for the purpose
    of optimizer state storage and parameter updates. Parameters are assigned
    round-robin by their index in the full parameter list.

    After the backward pass, this class provides helpers to:
      - all-reduce gradients across the DP group (so all ranks have correct grads),
      - run the optimizer step only on the local parameter partition,
      - all-gather parameters so every rank has the fully updated model weights.

    Args:
        params: All trainable parameters for this pipeline stage, in a fixed order.
        dp_rank: Data-parallel rank of the owning actor (0-indexed).
        dp_degree: Total number of data-parallel replicas.
        dp_group: The ``torch.distributed`` process group for this DP replica set.
        device: CUDA device string (e.g. ``"cuda:0"``).
    """

    def __init__(
        self,
        params: list[Parameter],
        dp_rank: int,
        dp_degree: int,
        dp_group: dist.ProcessGroup,
        device: str,
    ) -> None:
        if dp_degree < 1:
            raise ValueError(f"dp_degree must be >= 1, got {dp_degree}")
        if not (0 <= dp_rank < dp_degree):
            raise ValueError(f"dp_rank {dp_rank} out of range for dp_degree {dp_degree}")

        self.all_params: list[Parameter] = params
        self.dp_rank: int = dp_rank
        self.dp_degree: int = dp_degree
        self.dp_group: dist.ProcessGroup = dp_group
        self.device: str = device

        # Assign each param to a DP rank via round-robin on param index.
        self.param_owners: list[int] = [i % dp_degree for i in range(len(params))]

        # Parameters owned by this DP rank.
        self.local_params: list[Parameter] = [
            p for i, p in enumerate(params) if self.param_owners[i] == dp_rank
        ]

        logger.debug(
            f"ZeROOneState dp_rank={dp_rank}: owns {len(self.local_params)} / "
            f"{len(params)} params "
            f"({sum(p.numel() for p in self.local_params)} / "
            f"{sum(p.numel() for p in params)} elements)"
        )

    # ------------------------------------------------------------------
    # Optimizer creation
    # ------------------------------------------------------------------

    def create_optimizer(
        self,
        optim_class: Type[Optimizer],
        **optim_kwargs,
    ) -> Optimizer:
        """
        Create an optimizer that manages only this rank's parameter partition.

        Args:
            optim_class: The optimizer class (e.g. ``torch.optim.AdamW``).
            **optim_kwargs: Keyword arguments forwarded to the optimizer constructor
                (e.g. ``lr``, ``weight_decay``).

        Returns:
            An optimizer instance managing ``self.local_params``.
        """
        return optim_class(self.local_params, **optim_kwargs)

    # ------------------------------------------------------------------
    # Gradient synchronization
    # ------------------------------------------------------------------

    def all_reduce_gradients(
        self,
        comm_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        All-reduce gradients of all stage parameters across the DP group.

        After this call every DP rank holds the correctly averaged gradient for
        every parameter.  Non-local parameters still have gradients (so their
        values are correct), but only the owning rank will use them in the
        optimizer step.

        Args:
            comm_stream: Optional CUDA stream to run the collectives on.
                If ``None``, the current default stream is used.
        """
        ctx = torch.cuda.stream(comm_stream) if comm_stream is not None else _noop_ctx()
        with ctx:
            for p in self.all_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.dp_group)

    # ------------------------------------------------------------------
    # Parameter synchronization (post optimizer step)
    # ------------------------------------------------------------------

    def all_gather_params(
        self,
        comm_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Broadcast each parameter from its owning DP rank to all other ranks.

        Must be called **after** the optimizer step so that every rank ends up
        with the fully updated parameters before the next forward pass.

        Args:
            comm_stream: Optional CUDA stream to run the collectives on.
                If ``None``, the current default stream is used.
        """
        ctx = torch.cuda.stream(comm_stream) if comm_stream is not None else _noop_ctx()
        with ctx:
            for param_idx, p in enumerate(self.all_params):
                owner_dp_rank = self.param_owners[param_idx]
                # Map DP-group-local rank to global rank for dist.broadcast.
                owner_global_rank = dist.get_global_rank(self.dp_group, owner_dp_rank)
                dist.broadcast(p.data, src=owner_global_rank, group=self.dp_group)

    # ------------------------------------------------------------------
    # Combined convenience method (used in _update)
    # ------------------------------------------------------------------

    def optimizer_step_and_sync(
        self,
        optimizer: Optimizer,
        comm_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Run the optimizer step on local params, then all-gather updated params.

        The caller is responsible for running ``all_reduce_gradients`` before
        calling this method.

        Args:
            optimizer: The optimizer returned by ``create_optimizer``.
            comm_stream: Optional CUDA stream for the all-gather collective.
        """
        optimizer.step()
        optimizer.zero_grad()
        self.all_gather_params(comm_stream=comm_stream)

    def zero_all_gradients(self) -> None:
        """
        Zero gradients of **all** stage parameters (local and non-local).

        Must be called after the optimizer step so that non-local parameters'
        gradient buffers (which were populated by the all-reduce but not used
        by the local optimizer) are cleared before the next backward pass.
        The local optimizer's ``zero_grad()`` only clears local params, so
        this method is needed to handle the non-local ones.
        """
        for p in self.all_params:
            if p.grad is not None:
                p.grad = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def local_param_count(self) -> int:
        """Return the number of parameters owned by this DP rank."""
        return len(self.local_params)

    def local_element_count(self) -> int:
        """Return the total number of parameter elements owned by this DP rank."""
        return sum(p.numel() for p in self.local_params)

    def __repr__(self) -> str:
        return (
            f"ZeROOneState(dp_rank={self.dp_rank}, dp_degree={self.dp_degree}, "
            f"local_params={self.local_param_count()}, "
            f"local_elements={self.local_element_count()})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _noop_ctx:
    """A no-op context manager used when no CUDA stream is provided."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
