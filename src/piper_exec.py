import os
import ray
import torch
import torch.distributed as dist
from typing import NamedTuple
import threading
import time

from .piper_utils import piper_metadata, create_logger, LOG_LEVEL

logger = create_logger("piper_exec", LOG_LEVEL)

class Task(NamedTuple):
    pp_rank: int
    stage_id: int
    mb_idx: int
    is_fwd: bool
    upd: bool

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
    
    # Check well-formedness: no duplicates, pp_rank matches row, and all stages present
    all_tasks = set()
    microbatch_tasks = {}  # mb_idx -> set of (stage_id, is_fwd, upd)
    
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
                
                # Check for duplicates
                task_key = (task.stage_id, task.mb_idx, task.is_fwd, task.upd)
                if task_key in all_tasks:
                    raise ValueError(
                        f"Duplicate task found: stage_id={task.stage_id}, "
                        f"mb_idx={task.mb_idx}, is_fwd={task.is_fwd}, upd={task.upd}"
                    )
                all_tasks.add(task_key)
                
                # Track tasks by microbatch
                if task.mb_idx not in microbatch_tasks:
                    microbatch_tasks[task.mb_idx] = set()
                microbatch_tasks[task.mb_idx].add((task.stage_id, task.is_fwd, task.upd))
    
    # Get all required stages from DAG edges
    all_required_stages = set()
    for edge in dag_edges:
        all_required_stages.add(edge.from_stage)
        all_required_stages.add(edge.to_stage)
    
    # Check that each microbatch has all required forward and backward stages
    for mb_idx, tasks in microbatch_tasks.items():
        # Find all stages that have forward/backward tasks for this microbatch
        fwd_stages = {stage_id for stage_id, is_fwd, upd in tasks if is_fwd and not upd}
        bwd_stages = {stage_id for stage_id, is_fwd, upd in tasks if not is_fwd and not upd}
        
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
                if task is not None and task.mb_idx == mb_idx:
                    if task.is_fwd and not task.upd:
                        fwd_times[task.stage_id] = time_step
                    elif not task.is_fwd and not task.upd:
                        bwd_times[task.stage_id] = time_step
        
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

def piper_exec(model, schedule, inputs, truth, loss_fn, dp_degree=1):
    """
    Execute one step of the pipeline schedule on the distributed model.

    Args:
        model: A model that has been compiled with the piper backend.
        schedule: A 2D list (device x time_step) of Tasks specifying execution order.
        inputs: Inputs to the model.
        truth: Ground-truth labels or targets.
        loss_fn: Loss function to be used for training.
        num_mbs: Number of microbatches in the schedule.

    Returns:
        List of losses per microbatch.)
    """
    num_steps, num_devices = len(schedule[0]), len(schedule)

    actors = piper_metadata.actors
    if dp_degree > 1:
        [actor._comm_loop.remote() for actor in actors.values()]
    
    num_mbs = len(set([task.mb_idx for row in schedule for task in row if task is not None]))
    dag_edges = piper_metadata.dag
    dag_edges = list(map(lambda e: (DAGEdge(e[0], e[1])), list(piper_metadata.dag)))
    
    _validate_schedule(schedule, dag_edges, num_mbs)

    ray.get([actor.reset_p2p_states.remote() for actor in actors.values()])
    ret = []
    for i in range(num_steps):
        for j in range(num_devices-1, -1, -1):
            task = schedule[j][i]
            if task:
                _, stage_id, mb_idx, is_fwd, upd = task
                actor_id = j
                actor = actors[actor_id]
                if upd:
                    ret.append(actor._update.remote())
                elif is_fwd:
                    actor._forward.remote(stage_id, mb_idx)
                else:
                    actor._backward.remote(stage_id, mb_idx, loss_fn=loss_fn)
    return ray.get(ret)