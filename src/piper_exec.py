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
import itertools

from .piper_utils import piper_metadata, create_logger, LOG_LEVEL

logger = create_logger("piper_exec", LOG_LEVEL)

class CompType(Enum):
    FWD = "forward"
    BWD = "backward"
    UPD = "update"
    FWD_BWD = "forward_backward"


class BatchMeta(NamedTuple):
    """Metadata for one microbatch executed as part of a task."""
    stage_id: int
    mb_idx: int


class ScheduleTask(NamedTuple):
    """A node in the schedule DAG: runs on pp_rank with one microbatch."""

    pp_rank: int
    microbatch: BatchMeta

    def __repr__(self) -> str:
        mb = self.microbatch
        if mb.comp_type is None:
            c = "?"
        elif mb.comp_type == CompType.FWD:
            c = "f"
        elif mb.comp_type == CompType.BWD:
            c = "b"
        else:
            c = "u"
        return f"rank{self.pp_rank}_stage{mb.stage_id}_mb{mb.mb_idx}_{c}"


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
        stages = {task.microbatch.stage_id for task in self.tasks}
        return max(stages) + 1 if stages else 0

    def num_mbs(self) -> int:
        mbs = {task.microbatch.mb_idx for task in self.tasks}
        return max(mbs) + 1 if mbs else 0


# Task for 2D schedule grid: one cell (rank, time_step).
class Task(NamedTuple):
    pp_rank: int
    batches: list[BatchMeta]
    type: CompType


@dataclass
class Schedule2D:
    """2D schedule grid: grid[rank][time_step] = Task | None. Execution order: by time_step, then rank descending."""
    grid: list[list["Task | None"]]

    def stage_to_device(self) -> dict[int, int]:
        stage_to_device = {}
        for row in self.grid:
            for task in row:
                if task is not None:
                    for batch in task.batches:
                        stage_to_device[batch.stage_id] = task.pp_rank
        return stage_to_device
    
    def num_mbs(self) -> int:
        mbs = set()
        for row in self.grid:
            for task in row:
                if task is not None:
                    for batch in task.batches:
                        mbs.add(batch.mb_idx)
        return len(mbs)
    
    def num_stages(self) -> int:
        stages = set()
        for row in self.grid:
            for task in row:
                if task is not None:
                    for batch in task.batches:
                        stages.add(batch.stage_id)
        return len(stages)
    
    def num_ranks(self) -> int:
        return len(self.grid)

class DAGEdge(NamedTuple):
    from_stage: int
    to_stage: int


def _get_backward_targets(stage_id: int, dag_edges: list[DAGEdge]):
    return [edge for edge in dag_edges if edge.to_stage == stage_id]

def _validate_schedule(schedule: list[list[Task | None]], dag_edges: list[DAGEdge], num_mbs: int) -> None:
    """
    Validate that the schedule respects well-formedness rules and DAG dependencies.
    
    Args:
        schedule: 2D array with one row per device and one column per time step
        dag_edges: List of DAG edges defining stage dependencies
        num_mbs: Number of microbatches in the schedule
        
    Raises:
        ValueError: If the schedule violates any validation rules
    """
    num_devices, num_steps = len(schedule), len(schedule[0]) if schedule else 0
    
    for row in schedule:
        assert len(row) == num_steps, "Each row must have the same number of time steps"
    
    # Check well-formedness: no duplicates, pp_rank matches row, and all stages present
    all_tasks = set()
    microbatch_tasks = {}  # mb_idx -> set of (stage_id, type)
    
    for stage_id in range(num_devices):
        for time_step in range(num_steps):
            task = schedule[stage_id][time_step]
            if task is not None:
                # Check pp_rank matches row
                if task.pp_rank != stage_id:
                    raise ValueError(
                        f"Task pp_rank {task.pp_rank} does not match row {stage_id} "
                        f"at time step {time_step}"
                    )
                
                # Check for duplicates and track by (stage_id, mb_idx, type) per batch
                for batch in task.batches:
                    comp_type = (
                        task.type
                        if task.type != CompType.FWD_BWD
                        else (CompType.FWD if batch is task.batches[0] else CompType.BWD)
                    )
                    task_key = (batch.stage_id, batch.mb_idx, comp_type)
                    if task_key in all_tasks:
                        raise ValueError(
                            f"Duplicate task found: stage_id={batch.stage_id}, "
                            f"mb_idx={batch.mb_idx}, type={comp_type}"
                        )
                    all_tasks.add(task_key)
                    if batch.mb_idx not in microbatch_tasks:
                        microbatch_tasks[batch.mb_idx] = set()
                    microbatch_tasks[batch.mb_idx].add((batch.stage_id, comp_type))
    
    # Get all required stages from DAG edges
    all_required_stages = set()
    for edge in dag_edges:
        all_required_stages.add(edge.from_stage)
        all_required_stages.add(edge.to_stage)
    
    # Check that each microbatch has all required forward and backward stages
    for mb_idx, tasks in microbatch_tasks.items():
        # Find all stages that have forward/backward tasks for this microbatch
        fwd_stages = {stage_id for stage_id, task_type in tasks if task_type == CompType.FWD}
        bwd_stages = {stage_id for stage_id, task_type in tasks if task_type == CompType.BWD}
        
        # Check that all required stages have forward tasks
        missing_fwd = all_required_stages - fwd_stages
        if missing_fwd:
            raise ValueError(
                f"Microbatch {mb_idx} missing forward stages: {missing_fwd}"
            )
        
        # Check that all required stages have backward tasks
        missing_bwd = all_required_stages - bwd_stages
        if missing_bwd:
            raise ValueError(
                f"Microbatch {mb_idx} missing backward stages: {missing_bwd}"
            )
    
    # Check pipeline stage dependencies
    for mb_idx in range(num_mbs):
        # Find all tasks for this microbatch
        fwd_times = {}  # stage_id -> time_step
        bwd_times = {}  # stage_id -> time_step
        
        for pp_rank in range(num_devices):
            for time_step in range(num_steps):
                task = schedule[pp_rank][time_step]
                if task is None:
                    continue
                for batch in task.batches:
                    if batch.mb_idx != mb_idx:
                        continue
                    comp_type = (
                        task.type
                        if task.type != CompType.FWD_BWD
                        else (CompType.FWD if batch is task.batches[0] else CompType.BWD)
                    )
                    if comp_type == CompType.FWD:
                        fwd_times[batch.stage_id] = time_step
                    elif comp_type == CompType.BWD:
                        bwd_times[batch.stage_id] = time_step
        
        # Check forward stage ordering: if A -> B, then fwd(A) < fwd(B)
        for edge in dag_edges:
            from_stage, to_stage = edge.from_stage, edge.to_stage
            if from_stage in fwd_times and to_stage in fwd_times:
                if fwd_times[from_stage] >= fwd_times[to_stage]:
                    raise ValueError(
                        f"Forward stage ordering violation for microbatch {mb_idx}: "
                        f"forward stage {from_stage} (time {fwd_times[from_stage]}) must come "
                        f"before forward stage {to_stage} (time {fwd_times[to_stage]})"
                    )
        
        # Check forward-backward ordering: fwd(A) < bwd(A)
        for stage_id in fwd_times:
            if stage_id in bwd_times:
                if fwd_times[stage_id] >= bwd_times[stage_id]:
                    raise ValueError(
                        f"Forward-backward ordering violation for microbatch {mb_idx}, "
                        f"stage {stage_id}: forward (time {fwd_times[stage_id]}) must come "
                        f"before backward (time {bwd_times[stage_id]})"
                    )
        
        # Check backward stage ordering: if A -> B, then bwd(B) < bwd(A)
        for edge in dag_edges:
            from_stage, to_stage = edge.from_stage, edge.to_stage
            if from_stage in bwd_times and to_stage in bwd_times:
                if bwd_times[to_stage] >= bwd_times[from_stage]:
                    raise ValueError(
                        f"Backward stage ordering violation for microbatch {mb_idx}: "
                        f"backward stage {to_stage} (time {bwd_times[to_stage]}) must come "
                        f"before backward stage {from_stage} (time {bwd_times[from_stage]})"
                    )


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
        u_rank = tasks[u].pp_rank
        if queue:
            for v in queue:
                if tasks[v].pp_rank == u_rank:
                    logger.warning(
                        "Topological order: ordering between task %s and task %s (same pp_rank %d) cannot be determined (no directed path in DAG)",
                        tasks[u],
                        tasks[v],
                        u_rank,
                    )
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


def piper_exec(
    schedule: Schedule2D,
    loss_fn,
    dp_degree=1,
    naive_gradient_sync=False,
):
    """
    Execute one step of the pipeline schedule on the distributed model.

    Args:
        schedule: 2D schedule grid (rank x time_step).
        inputs: Inputs to the model.
        truth: Ground-truth labels or targets.
        loss_fn: Loss function to be used for training.

    Returns:
        List of losses per microbatch (from UPD tasks).
    """
    num_mbs = schedule.num_mbs()
    num_devices = schedule.num_ranks()
    num_steps = len(schedule.grid[0])

    schedule = schedule.grid

    dag_edges = piper_metadata.dag
    dag_edges = list(map(lambda e: (DAGEdge(e[0], e[1])), list(piper_metadata.dag)))
    _validate_schedule(schedule, dag_edges, num_mbs)

    actors = piper_metadata.actors

    ret = []
    # map pp_rank -> latest dependency
    deps = defaultdict(dict)
    for i in range(num_steps):
        for j in range(num_devices-1, -1, -1):
            if not i < len(schedule[j]):
                continue
            task = schedule[j][i]
            if task:
                pp_rank, batches, task_type = task
                actor_id = j
                actor = actors[actor_id]
                match task_type:
                    case CompType.UPD:
                        loss = actor._update.remote(deps[pp_rank])
                        ret.append(loss)
                    case CompType.FWD:
                        stage_id, mb_idx = batches[0]
                        # logger.info(f"Executing forward {stage_id} mb {mb_idx} on actor {actor_id}")
                        if pp_rank not in deps:
                            dep = actor._forward.remote(stage_id, mb_idx, None)
                        else:
                            dep = actor._forward.remote(stage_id, mb_idx, deps[pp_rank])
                        deps[pp_rank] = dep
                    case CompType.BWD:
                        stage_id, mb_idx = batches[0]
                        # logger.info(f"Executing backward {stage_id} mb {mb_idx} on actor {actor_id}")
                        dep = actor._backward.remote(stage_id, mb_idx, deps[pp_rank], loss_fn=loss_fn)
                        deps[pp_rank] = dep
                    case CompType.FWD_BWD:
                        fwd_stage_id, fwd_mb_idx = batches[0]
                        bwd_stage_id, bwd_mb_idx = batches[1]
                        # logger.info(f"Executing forward-backward {fwd_stage_id} mb {fwd_mb_idx} -> {bwd_stage_id} mb {bwd_mb_idx} on actor {actor_id}")
                        dep = actor._forward_backward.remote(fwd_stage_id, fwd_mb_idx, bwd_stage_id, bwd_mb_idx, deps[pp_rank], loss_fn=loss_fn)
                        deps[pp_rank] = dep
    return ray.get(ret)