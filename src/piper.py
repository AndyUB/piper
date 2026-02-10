from . import piper_patches

import ray
import torch
import os
from torch._dynamo.backends.registry import register_backend

from .piper_utils import _serialize_graphmodule, piper_metadata, create_logger, LOG_LEVEL
from .piper_graph_transform import _get_dp_comm_ops, _split_gm_by_stages, _insert_comm_ops
from .piper_actor import _get_actor

logger = create_logger("piper_backend", LOG_LEVEL)


@register_backend
def piper(gm, example_inputs, **kwargs):

    original_gm = gm
    top_level_gm, submodules = _split_gm_by_stages(gm)

    refs = []
    actor_stages = []
    for (stage_id, stage_gm, input_idxs, params_with_holes, placeholders) in submodules:
        stage_gm = _insert_comm_ops(stage_gm)
        actor_id = piper_metadata.stage_to_device[stage_id]
        comm_ops, tids = _get_dp_comm_ops(params_with_holes, placeholders)
        actor = _get_actor(actor_id)
        actor_stages.append((actor, stage_id))
        stage_gm_data = _serialize_graphmodule(stage_gm)
        logger.debug(f"Loading stage {stage_id} on actor {actor_id}")
        refs.append(actor._load_stage.remote(
            stage_id, 
            stage_gm_data, 
            comm_ops,
            params_with_holes,
            tids,
            input_idxs,
        ))
    ray.get(refs)

    [actor._comm_loop.remote(stage_id) for (actor, stage_id) in actor_stages]

    # TODO: build dag by analyzing stage dependencies
    # this only supports sequential pipelines
    for stage_id in piper_metadata.stage_to_device.keys():
        if stage_id < len(piper_metadata.stage_to_device.keys()) - 1:
            piper_metadata.dag.add((stage_id, stage_id+1))

    def callback(*args):
        logger.warning("Should not directly call compiled module, running non-distributed execution")
        return original_gm(*args)

    return callback