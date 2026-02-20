import os
import ray
import torch
import torch.distributed as dist
from typing import NamedTuple
from enum import Enum
from dataclasses import dataclass, field
import threading
import time
from collections import defaultdict

from .piper_utils import piper_metadata, create_logger, LOG_LEVEL

logger = create_logger("piper_exec", LOG_LEVEL)

class CompType(Enum):
    FWD = "forward"
    BWD = "backward"
    UPD = "update"
    FWD_BWD = "forward_backward"


class MicrobatchMetadata(NamedTuple):
    """Metadata for one microbatch executed as part of a task."""
    stage_id: int
    mb_idx: int
    comp_type: CompType


class ScheduleTask(NamedTuple):
    """A node in the schedule DAG: runs on pp_rank with one or more microbatches."""
    pp_rank: int
    microbatches: tuple[MicrobatchMetadata, ...]


class ScheduleEdge(NamedTuple):
    """Data dependency: to_task consumes output of from_task (source and destination only)."""
    from_task_idx: int
    to_task_idx: int


class TemporalEdge(NamedTuple):
    """Temporal dependency: to_task must run after from_task (ordering only)."""
    from_task_idx: int
    to_task_idx: int


@dataclass
class Schedule:
    """DAG schedule: tasks (nodes), data edges, and temporal edges."""
    tasks: list[ScheduleTask]
    edges: list[ScheduleEdge]
    temporal_edges: list[TemporalEdge] = field(default_factory=list)

    def num_stages(self) -> int:
        stages = set()
        for task in self.tasks:
            for mb in task.microbatches:
                stages.add(mb.stage_id)
        return max(stages) + 1 if stages else 0

    def num_mbs(self) -> int:
        mbs = set()
        for task in self.tasks:
            for mb in task.microbatches:
                mbs.add(mb.mb_idx)
        return max(mbs) + 1 if mbs else 0


# Legacy single-microbatch task (for backward compatibility / conversion).
class Task(NamedTuple):
    pp_rank: int
    stage_id: int
    mb_idx: int
    type: CompType
    second_stage_id: int | None = None
    second_mb_idx: int | None = None

class DAGEdge(NamedTuple):
    from_stage: int
    to_stage: int


def _get_backward_targets(stage_id: int, dag_edges: list[DAGEdge]):
    return [edge for edge in dag_edges if edge.to_stage == stage_id]


def _validate_schedule_dag(
    schedule: Schedule,
    dag_edges: list[DAGEdge],
    num_stages: int,
    num_mbs: int,
) -> None:
    """
    Validate that the schedule DAG has valid data dependencies for all tasks.
    For each microbatch: FWD/BWD have correct input/output edges; UPD has none.
    """
    tasks = schedule.tasks
    edges = schedule.edges
    edges_into: dict[int, list[int]] = defaultdict(list)
    edges_out_of: dict[int, list[int]] = defaultdict(list)
    for e in edges:
        edges_into[e.to_task_idx].append(e.from_task_idx)
        edges_out_of[e.from_task_idx].append(e.to_task_idx)

    all_required_stages = set()
    for edge in dag_edges:
        all_required_stages.add(edge.from_stage)
        all_required_stages.add(edge.to_stage)

    task_produces: dict[int, set[tuple[int, int, CompType]]] = defaultdict(set)
    for ti, task in enumerate(tasks):
        for mb in task.microbatches:
            task_produces[ti].add((mb.stage_id, mb.mb_idx, mb.comp_type))

    for ti, task in enumerate(tasks):
        predecessors = edges_into[ti]
        successors = edges_out_of[ti]
        for mb in task.microbatches:
            s, m, c = mb.stage_id, mb.mb_idx, mb.comp_type
            if c == CompType.UPD:
                continue
            if c == CompType.FWD:
                if s > 0:
                    required = (s - 1, m, c)
                    if not any(required in task_produces[u] for u in predecessors):
                        raise ValueError(
                            f"Task {ti} microbatch (stage_id={s}, mb_idx={m}, FWD) missing input: "
                            f"no predecessor produces {required}"
                        )
                if s < num_stages - 1:
                    required = (s + 1, m, c)
                    if not any(required in task_produces[v] for v in successors):
                        raise ValueError(
                            f"Task {ti} microbatch (stage_id={s}, mb_idx={m}, FWD) missing output: "
                            f"no successor consumes {required}"
                        )
            else:  # BWD
                if s == num_stages - 1:
                    required = (s, m, CompType.FWD)
                    if not any(required in task_produces[u] for u in predecessors):
                        raise ValueError(
                            f"Task {ti} microbatch (stage_id={s}, mb_idx={m}, BWD) missing FWD_BWD input: "
                            f"no predecessor produces {required}"
                        )
                if s < num_stages - 1:
                    required = (s + 1, m, c)
                    if not any(required in task_produces[u] for u in predecessors):
                        raise ValueError(
                            f"Task {ti} microbatch (stage_id={s}, mb_idx={m}, BWD) missing BWD input: "
                            f"no predecessor produces {required}"
                        )
                if s > 0:
                    required = (s - 1, m, c)
                    if not any(required in task_produces[v] for v in successors):
                        raise ValueError(
                            f"Task {ti} microbatch (stage_id={s}, mb_idx={m}, BWD) missing output: "
                            f"no successor consumes {required}"
                        )

    mb_fwd_stages: dict[int, set[int]] = defaultdict(set)
    mb_bwd_stages: dict[int, set[int]] = defaultdict(set)
    for task in tasks:
        for mb in task.microbatches:
            if mb.comp_type == CompType.FWD:
                mb_fwd_stages[mb.mb_idx].add(mb.stage_id)
            elif mb.comp_type == CompType.BWD:
                mb_bwd_stages[mb.mb_idx].add(mb.stage_id)
    for mb_idx in range(num_mbs):
        missing_fwd = all_required_stages - mb_fwd_stages.get(mb_idx, set())
        if missing_fwd:
            raise ValueError(f"Microbatch {mb_idx} missing forward stages: {missing_fwd}")
        missing_bwd = all_required_stages - mb_bwd_stages.get(mb_idx, set())
        if missing_bwd:
            raise ValueError(f"Microbatch {mb_idx} missing backward stages: {missing_bwd}")


def _topological_order(schedule: Schedule) -> list[int]:
    """Return task indices in topological order (data + temporal predecessors)."""
    tasks = schedule.tasks
    edges = schedule.edges
    temporal = schedule.temporal_edges or []
    n = len(tasks)
    in_degree = [0] * n
    for e in edges:
        in_degree[e.to_task_idx] += 1
    for e in temporal:
        in_degree[e.to_task_idx] += 1
    queue = [i for i in range(n) if in_degree[i] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for e in edges:
            if e.from_task_idx == u:
                in_degree[e.to_task_idx] -= 1
                if in_degree[e.to_task_idx] == 0:
                    queue.append(e.to_task_idx)
        for e in temporal:
            if e.from_task_idx == u:
                in_degree[e.to_task_idx] -= 1
                if in_degree[e.to_task_idx] == 0:
                    queue.append(e.to_task_idx)
    if len(order) != n:
        raise ValueError("Schedule DAG has a cycle")
    return order


def piper_exec(schedule: Schedule, inputs, truth, loss_fn, dp_degree=1, naive_gradient_sync=False):
    """
    Execute one step of the pipeline schedule on the distributed model.

    Args:
        schedule: Schedule DAG (tasks and edges).
        inputs: Inputs to the model.
        truth: Ground-truth labels or targets.
        loss_fn: Loss function to be used for training.

    Returns:
        List of losses per microbatch (from UPD tasks).
    """
    dag_edges = [DAGEdge(e[0], e[1]) for e in piper_metadata.dag]
    num_stages = schedule.num_stages()
    num_mbs = schedule.num_mbs()
    _validate_schedule_dag(schedule, dag_edges, num_stages, num_mbs)

    order = _topological_order(schedule)
    actors = piper_metadata.actors
    ret = []
    for ti in order:
        task = schedule.tasks[ti]
        actor = actors[task.pp_rank]
        for mb in task.microbatches:
            if mb.comp_type == CompType.UPD:
                ret.append(actor._update.remote())
            elif mb.comp_type == CompType.FWD:
                actor._forward.remote(mb.stage_id, mb.mb_idx)
            elif mb.comp_type == CompType.BWD:
                actor._backward.remote(mb.stage_id, mb.mb_idx, loss_fn=loss_fn)
    return ray.get(ret)