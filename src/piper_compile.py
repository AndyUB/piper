import ray
import torch
import threading
import os
import gc
import copy

from torch._dynamo.backends.debugging import eager

from .piper_actor import _create_actors
from .piper_utils import piper_metadata, create_logger, LOG_LEVEL
from .piper import piper

logger = create_logger("piper_compile", LOG_LEVEL)


def order_p2p_comms(schedule, num_devices, num_stages, stage_to_device):
    is_fwd_bwd = lambda task: task is not None and not task.upd
    schedule = [
        [task for task in rank_tasks if is_fwd_bwd(task)] for rank_tasks in schedule
    ]
    logger.debug(f"Fwd/bwd schedule: {schedule}")
    assert num_devices == len(schedule), "Some rank has no fwd/bwd tasks"
    # Tasks are guaranteed to be fwd or bwd
    num_fwd_bwd_tasks = sum(len(rank_tasks) for rank_tasks in schedule)

    num_executed_tasks = 0
    # (src_stage, dst_stage, mb_idx) for each p2p comm in execution order
    # is_fwd if src_stage < dst_stage else bwd
    p2p_comms = []
    rank_to_next_task = {rank: 0 for rank in range(len(schedule))}
    # P2P comm is enqueued when sender is ready
    # Pick the next executable task in a round-robin fashion
    while True:
        executable_tasks = []
        for rank, time_step in rank_to_next_task.items():
            if time_step < len(schedule[rank]):
                task = schedule[rank][time_step]
                # Check the task's dependency is satisfied
                # i.e., the sender has enqueued the comm
                dependee = None
                if task.is_fwd and task.stage_id > 0:
                    dependee = (task.stage_id - 1, task.stage_id, task.mb_idx)
                elif not task.is_fwd and task.stage_id < num_stages - 1:
                    dependee = (task.stage_id + 1, task.stage_id, task.mb_idx)

                can_exec = dependee is None or dependee in p2p_comms
                if can_exec:
                    executable_tasks.append((rank, task))

        if not executable_tasks:
            break

        for rank, task in executable_tasks:
            if task.is_fwd and task.stage_id < num_stages - 1:
                p2p_comms.append((task.stage_id, task.stage_id + 1, task.mb_idx))
            elif not task.is_fwd and task.stage_id > 0:
                p2p_comms.append((task.stage_id, task.stage_id - 1, task.mb_idx))
            rank_to_next_task[rank] += 1

        num_executed_tasks += len(executable_tasks)

    if num_executed_tasks != num_fwd_bwd_tasks:
        raise ValueError(
            f"Only {num_executed_tasks} out of {num_fwd_bwd_tasks} "
            f"fwd/bwd tasks can be executed"
        )

    logger.debug(f"P2P comms: {p2p_comms}")
    # (src_stage, dst_stage, mb_idx, is_sender)
    p2p_schedules = {rank: [] for rank in range(num_devices)}
    for src_stage, dst_stage, mb_idx in p2p_comms:
        src_rank = stage_to_device[src_stage]
        dst_rank = stage_to_device[dst_stage]
        p2p_schedules[src_rank].append((src_stage, dst_stage, mb_idx, True))
        p2p_schedules[dst_rank].append((src_stage, dst_stage, mb_idx, False))
    logger.debug(f"P2P schedules: {p2p_schedules}")

    return p2p_schedules


def piper_setup(
    model_class,
    model_args,
    optim_fn,
    example_inputs,
    schedule,
    naive_gradient_sync=False,
):
    """
    Compile a model with the piper backend.
    """
    piper_metadata.naive_gradient_sync = naive_gradient_sync

    num_devices = len(schedule)
    num_mbs = len(
        set([task.mb_idx for row in schedule for task in row if task is not None])
    )
    num_stages = len(
        set([task.stage_id for row in schedule for task in row if task is not None])
    )
    piper_metadata.stage_to_device = {
        task.stage_id: task.pp_rank
        for row in schedule
        for task in row
        if task is not None
    }

    logger.debug(f"mbs: {num_mbs}, stages: {num_stages}, devices: {num_devices}")

    p2p_schedules = order_p2p_comms(
        schedule, num_devices, num_stages, piper_metadata.stage_to_device
    )
    _create_actors(
        num_devices, optim_fn, num_mbs, num_stages, p2p_schedules, naive_gradient_sync
    )
    ray.get(
        [
            actor._join_process_groups.remote()
            for actor in piper_metadata.actors.values()
        ]
    )

    model = model_class(*model_args)

    compiled = torch.compile(model, backend=piper)

    dp_rank = int(os.environ["PIPER_DP_RANK"])
    logger.info(f"DP rank {dp_rank+1} compiling...")

    output = compiled(*example_inputs)

    logger.info(f"DP rank {dp_rank+1} done.")

    return compiled
