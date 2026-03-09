# ZeRO-1 with Bucketed Reduce-Scatter for Piper

## Overview

ZeRO-1 (Zero Redundancy Optimizer Stage 1) shards optimizer states across data-parallel
(DP) ranks. This design uses **per-parameter sharding** and **gradient-sync buckets**
to overlap communication with backward computation, matching PyTorch DDP's bucketing
approach but adapted for ZeRO's reduce-scatter + all-gather pattern.

---

## Memory Savings

For Adam with fp32 states, optimizer state costs 12 bytes/element (fp32 master weight +
momentum + variance). ZeRO-1 reduces this by `dp_degree`:

| DP degree | Optimizer states per rank |
|-----------|--------------------------|
| 1 (DDP)   | 100%                     |
| 2         | 50%                      |
| 4         | 25%                      |
| 8         | 12.5%                    |

---

## Per-Parameter Sharding

Each trainable parameter `p` (numel = N) is divided into `dp_degree` contiguous shards
of size `ceil(N / dp_degree)`. DP rank `r` owns the shard covering flat indices
`[r * shard_size, (r+1) * shard_size)` (last rank may be padded with zeros).

**Optimizer states** are stored only for the local shard — giving the memory savings.
The **full parameter** remains on every rank (needed for forward/backward), but the
optimizer (Adam's `m`, `v`, etc.) operates only on the local slice.

---

## Bucketing

Parameters are grouped into fixed-size buckets so that a single collective covers
multiple parameters. This amortises per-collective latency and improves bandwidth
utilisation.

### Construction Order

Buckets are built in **reverse forward-use order** to approximate backward gradient
arrival order:

1. Extract the first-use position of each parameter in the FX graph (exact, no runtime
   overhead — determined statically from graph node order).
2. Reverse the list: parameters used last in forward produce gradients first in backward.
3. Fill buckets by walking this reversed list, starting a new bucket when the running
   size exceeds the 25 MB threshold (matching PyTorch DDP's default).

Each bucket gets one flat gradient buffer, one flat parameter buffer, and one local
shard buffer — all allocated once at setup and reused across steps.

---

## Communication Pattern per Training Step

```
Forward pass  ─── all ranks have full params (as in DDP) ───────────────────►

Backward pass:
  grad(p_k) ready ──► bucket B_i full? ──► async reduce_scatter(B_i)
                                            ▲ overlaps with backward for B_{i+1},...

_update():
  for each bucket (in backward order):
    wait reduce_scatter
    optimizer step on local shard          ◄── overlaps with all_gather(B_{i-1})
    async all_gather → flat_param_buf

  for each bucket:
    wait all_gather
    copy flat_param_buf → p.data
```

### Collectives Used

| Operation | API | Direction | Result |
|-----------|-----|-----------|--------|
| Gradient averaging | `dist.reduce_scatter_tensor` | N params → local shard | Rank r gets averaged grad for shard r |
| Parameter restore  | `dist.all_gather_into_tensor` | local shard → full params | Every rank gets complete updated params |

Both are launched with `async_op=True` on a dedicated `comm_stream`, overlapping with
backward computation on `comp_stream`.

---

## Flat Buffer Layout

For a bucket containing params `[p0, p1, p2]`:

```
flat_grad_buf / flat_param_buf (padded_numel = ceil(T / dp_degree) * dp_degree):
┌─────────────┬─────────────┬─────────────┬──────────┐
│  p0 elems   │  p1 elems   │  p2 elems   │ (padding)│
└─────────────┴─────────────┴─────────────┴──────────┘
               ◄──── shard for rank r ───►
                 [r*shard_size, (r+1)*shard_size)
```

After `reduce_scatter`, rank r holds the averaged gradient for its shard.
After the optimizer step, rank r holds the updated parameters for its shard.
After `all_gather`, every rank has `flat_param_buf` fully reconstructed.

---

## Implementation Structure

### `src/piper_zero1.py`

| Symbol | Role |
|--------|------|
| `BUCKET_SIZE_BYTES_DEFAULT` | 25 MB threshold (same as DDP) |
| `_get_param_forward_order(gm, param_idxs, params)` | Static FX graph analysis to sort params by first use |
| `ZeROOneBucket` | One bucket: flat buffers, local shard, per-bucket optimizer, grad-ready counting |
| `ZeROOneState` | Manages all buckets, registers hooks, runs pipelined `finalize_step` |

### `src/piper_actor.py` changes

| Location | Change |
|----------|--------|
| `_create_actors` | Accept `zero_stage: int = 0` |
| `PiperActor.__init__` | Accept `zero_stage`, store `self.zero1_states: dict` |
| `_load_stage` | When `zero_stage >= 1`: extract param order from FX graph, build `ZeROOneState` (which registers hooks); skip `_prepare_dp_comm_ops` |
| `_update` | When `zero_stage >= 1`: call `zero1_state.finalize_step(comm_stream, comp_stream)` |

### `src/piper_compile.py` changes

`piper_setup` accepts and forwards `zero_stage: int = 0`.

---

## Hook Counting with Multiple Microbatches

Piper supports `num_mbs` microbatches that accumulate gradients before a single
optimizer step. `register_post_accumulate_grad_hook` fires once per microbatch per
parameter. A bucket's reduce_scatter is launched only when every parameter in the
bucket has accumulated gradients from **all** `num_mbs` microbatches.

Per-step accumulation counts are tracked in `ZeROOneBucket._mb_accumulated` and reset
in `reset_step_state()` at the end of `finalize_step`.

---

## Interaction with BWD_I / BWD_W (Split Backward)

Split backward (`BWD_I` + `BWD_W`) writes gradients to `p.grad` via
`torch.autograd.grad` and manual accumulation, identical to full backward.
`register_post_accumulate_grad_hook` fires whenever `p.grad` is updated, so bucket
counting works transparently across both backward modes.

---

## Comparison: Old vs. New Design

| Aspect | Old (round-robin, all-reduce) | New (per-param shard, reduce_scatter + bucket) |
|--------|-------------------------------|------------------------------------------------|
| Param ownership | Whole params round-robin | Every rank co-owns every param (as a shard) |
| Gradient sync | all-reduce (each rank gets full grad) | reduce_scatter (each rank gets its shard's grad) |
| Gradient memory | Full gradient on every rank | Can free non-local gradient after reduce_scatter |
| Optimizer memory | Saves 1 − 1/dp for locally-owned params only | Saves 1 − 1/dp for **all** params uniformly |
| Load balance | Depends on per-param size distribution | Perfect: every rank has exactly `total/dp` elements |
| Overlap | None | reduce_scatter overlaps with backward; all_gather with optimizer |

---

## Future Extensions

- **ZeRO-2**: After reduce_scatter, free `p.grad` for non-owned elements to recover
  gradient memory. Requires splitting grad buffers.
- **ZeRO-3**: Shard the parameters themselves; fetch on-demand during forward/backward.
- **Larger / adaptive buckets**: Tune bucket size based on observed communication costs.
- **Optimizer overlap with all_gather**: Prefetch parameters for the next microbatch
  during the all_gather of the current step.
