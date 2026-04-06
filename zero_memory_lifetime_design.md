# ZeRO memory-lifetime redesign (Piper)

## Prompt

Currently we have conceptual bugs in our piper zero implementation. From our results, we saw that Zero-1/2/3 have the same memory consumption, which is not correct.
1. For Zero-2 and Zero-3, the full gradients are not freed properly
2. For Zero-3, the full parameters are not freed properly
3. (And a question), for Zero-1 and Zero-2, do we need to explicitly keep sharded params? Can the sharded param just be a view into the full parameters (since we always keep full params for Zero-1 and Zero-2)?

Below is my proposal to fix these:
1. When building the intermediate representation -- the DAG, we should have separate insert_zero1_ops, insert_zero2_ops, and insert_zero3_ops functions. Currently, we have just one insert_zero_ops thing. These 3 separate new functions can use common helpers such as insert_all_gather, insert_reduce_scatter if possible? (Keep them separate if the common helpers are not possible)
2. Let us add new task types (if this is a good idea). We can have the task types: ALLOC_FULL_GRADS and FREE_FULL_GRADS (for Zero-2 and 3), ALLOC_FULL_PARAMS and FREE_FULL_PARAMS (for Zero-3). As the names suggest, ALLOC_FULL_GRADS allocates full gradients and sets individual parameters' gradients as views into the full flat gradients, FREE_FULL_GRADS frees the allocated full gradients and clears any stale views, ALLOC_FULL_PARAMS allocates full parameters and sets individual parameters as views into the full flat parameters, and FREE_FULL_PARAMS frees the full parameters and clears any stale views.
Importantly, all these operations should operate at the granularity of stages! That is, even if a stage contains multiple buckets, we should allocate/free parameters/gradients for all buckets in the same stage at a time. (It is left up to the user to decide stage boundaries, so if a stage OOMS, it is left to the user to fix it.)
For efficiency, we want to reduce the number of collectives needed when possible. For a consecutive chain of tasks on the same actor/GPU that use the same stage's parameters, the full params allocation should happen just once at the start of the chain and the freeing happens at the end of the chain. That is, for Zero-3, ALLOC_FULL_PARAMS followed by the ALL_GATHER at the start of the chain, and FREE_FULL_PARAMS at the end.
For a consecutive chain of tasks that contribute to the same stage's gradients, we should ALLOC_FULL_GRADS once at the start of the chain and FREE_FULL_GRADS at the end. This is for Zero-2 and 3. However, the purpose of bucketing is to overlap backward gradient reduction with computation, so we should launch one reduce-scatter once gradients for a bucket have been produced. (Note we're reduce-scattering a bucket's gradients, but not freeing the full grads until the end of the backward chain.) The assumption here is that the generated DAG will compute the backward pass for the buckets of the same stage "contiguously," meaning these span a consecutive chunk of the generated pipeline.
(Discuss if this automatically supports or can be easily extended to the backward input/weight split in Zero-bubble.)
For implementation details, if we follow this design, for Zero-3 it is necessary to store the sharded params. From this we can easily allocate and all gather full params. Likewise, for zero-2 and zero-3, it is necessary to have sharded grads for a stage. (For zero-3 this corresponds to the sharded params.) This gets a bit complicated when combined with bucketing. For each bucket, the params should be flattened, padded, and evenly split across ranks. For a stage containing multiple buckets, each rank's stage-wise sharded params can either be 1) a flat tensor, with the first chunk being the sharded params for the first bucket; 2) a list of tensors, with the first item in the list being the sharded params for the first bucket. The second option should be more flexible in case we want to do overlapped per-bucket all-gathers in the future. (Likewise, for option 2 there will be a list of per-bucket flat sharded gradients for Zero-2 and 3. The allocated full grads and full params will also be per-bucket then. Hmm actually decide if it's better to launch separate all-gathers for different buckets, compared against launching 1 all-gather for all buckets, assuming the bucket size is at a good granularity.)
3. For the last overall point I mentioned in the beginning, I wonder if the following is possible: keep all parameters for a bucket as a single flat tensor (like in option 2 just described above), the model's parameters will be views into this flat tensor, and the optimizer will optimize over a view that is a shard of this flat tensor.
Please think carefully. Feel free to challenge my design. And also accept the good parts in my design.
Implement code changes to support this. Carefully handle any CUDA synchronizations needed. If PyTorch handles some syncs already, don't add unnecessary redundant syncs. If PyTorch does not handle certain syncs, we need to handle that.
Write good comments, but avoid excessive comments.
Document code changes in a design doc.
No need to run the actual program for testing. But for your reference, I'll be using the scripts titled something like run_zero*_llama.sh

## Problem
Our prior DAG transformation used one generic `insert_zero_ops` path and kept full parameter/gradient buffers alive longer than required. In practice this made ZeRO-1/2/3 peak memory much closer than expected.

## Goals
- Keep the ZeRO transformation explicit per stage (`insert_zero1_ops`, `insert_zero2_ops`, `insert_zero3_ops`).
- Represent explicit full-buffer lifetime operations in the DAG.
- Free full grads in ZeRO-2/3 once reduce-scatter work for the stage is done.
- Free full params in ZeRO-3 after the stage's compute chain completes.
- Perform alloc/free at stage granularity (all buckets in a stage together).

## DAG changes
Added task types:
- `ALLOC_FULL_GRADS` / `FREE_FULL_GRADS`
- `ALLOC_FULL_PARAMS` / `FREE_FULL_PARAMS`

Insertion strategy:
- **ZeRO-1**
  - Keep `ALL_REDUCE`.
  - Insert stage-level `ALL_GATHER` chain after `UPD`.
- **ZeRO-2**
  - Replace `ALL_REDUCE` with `REDUCE_SCATTER`.
  - Insert `ALLOC_FULL_GRADS` before first backward task for each stage.
  - Insert `FREE_FULL_GRADS` after the stage's last reduce-scatter (or last backward task if none).
  - Insert stage-level post-`UPD` `ALL_GATHER`.
- **ZeRO-3**
  - Replace `ALL_REDUCE` with `REDUCE_SCATTER`.
  - Insert `ALLOC_FULL_PARAMS -> ALL_GATHER` before the first forward task for each stage.
  - Insert `ALLOC_FULL_GRADS` before backward for each stage.
  - Insert `FREE_FULL_GRADS` after the stage's last reduce-scatter.
  - Insert `FREE_FULL_PARAMS` after the stage's last backward-like task.

Temporal links are chained between same-kind ZeRO helper nodes to keep deterministic NCCL ordering.

## Actor/runtime changes
- Added runtime handlers for the new task types.
- Added stage-level helpers:
  - `_alloc_full_params_for_stage`, `_free_full_params_for_stage`
  - `_alloc_full_grads_for_stage`, `_free_full_grads_for_stage`
- For ZeRO-3, `bucket_flat_params` and `bucket_flat_grads` are now transient and task-controlled.
- Parameter view metadata is cached (`bucket_param_view_specs`) so views can be rebound when full buffers are allocated.
- `ALL_GATHER` now supports stage-level execution (`bucket_id < 0`) to gather all buckets in a stage in one task.
- `_forward_dag` accepts stage-level all-gather events (`(stage_id, None)`) in addition to per-bucket keys.

## Synchronization notes
- No extra global synchronizations were added.
- Collectives still run on `comm_stream`; compute waits via CUDA events where required.
- `FREE_FULL_GRADS` waits for per-stage reduce-scatter events before releasing full gradient buffers to avoid racing in-flight communication.
- Existing `UPD` synchronization contract is unchanged.

## On sharded params for ZeRO-1/2
For this implementation, we keep explicit shard tensors (`bucket_shard_params`) across ZeRO stages. For ZeRO-1/2, these shards are views into full flat params (no extra param storage). For ZeRO-3, shards are owned tensors because full params are transient.

A view-based optimizer shard over always-resident full params can work for ZeRO-1/2, but it does not generalize to ZeRO-3 and weakens lifetime control, so we keep explicit shard tensors as the common representation.
