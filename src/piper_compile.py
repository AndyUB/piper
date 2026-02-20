import ray
import torch
import threading
import os
import gc
import copy

from torch._dynamo.backends.debugging import eager

from .piper_actor import _create_actors
from .piper_utils import piper_metadata, create_logger, LOG_LEVEL
from .piper_exec import CompType, Schedule, _topological_order
from .piper import piper

logger = create_logger("piper_compile", LOG_LEVEL)


def order_p2p_comms_from_dag(
    schedule: Schedule,
    num_devices: int,
    num_stages: int,
    stage_to_device: dict[int, int],
) -> dict[int, list[tuple[int, int, int, bool]]]:
    """Build P2P communication schedule from DAG in topological order."""
    order = _topological_order(schedule)
    p2p_comms: list[tuple[int, int, int]] = []
    for ti in order:
        task = schedule.tasks[ti]
        for mb in task.microbatches:
            if mb.comp_type == CompType.FWD and mb.stage_id < num_stages - 1:
                p2p_comms.append((mb.stage_id, mb.stage_id + 1, mb.mb_idx))
            elif mb.comp_type == CompType.BWD and mb.stage_id > 0:
                p2p_comms.append((mb.stage_id, mb.stage_id - 1, mb.mb_idx))
    logger.debug(f"P2P comms: {p2p_comms}")
    p2p_schedules = {rank: [] for rank in range(num_devices)}
    for src_stage, dst_stage, mb_idx in p2p_comms:
        src_rank = stage_to_device[src_stage]
        dst_rank = stage_to_device[dst_stage]
        p2p_schedules[src_rank].append((src_stage, dst_stage, mb_idx, True))
        p2p_schedules[dst_rank].append((src_stage, dst_stage, mb_idx, False))
    return p2p_schedules


def piper_setup(
    model,
    optim_fn,
    example_inputs,
    schedule: Schedule,
    naive_gradient_sync=False,
):
    """
    Compile a model with the piper backend.
    schedule: Schedule DAG (tasks and edges).
    """
    num_mbs = schedule.num_mbs()
    num_stages = schedule.num_stages()
    stage_to_device = {}
    for task in schedule.tasks:
        for mb in task.microbatches:
            stage_to_device[mb.stage_id] = task.pp_rank
    num_devices = max(stage_to_device.values()) + 1 if stage_to_device else 0
    piper_metadata.stage_to_device = stage_to_device
    p2p_schedules = order_p2p_comms_from_dag(
        schedule, num_devices, num_stages, stage_to_device
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

    compiled = torch.compile(model, backend=piper)

    dp_rank = int(os.environ["PIPER_DP_RANK"])
    logger.info(f"DP rank {dp_rank+1} compiling...")

    output = compiled(*example_inputs)

    logger.info(f"DP rank {dp_rank+1} done.")

    del compiled
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
