# ZeRO-1 Support in Piper

## Overview

ZeRO-1 (Zero Redundancy Optimizer Stage 1) is a memory-optimization technique that
partitions **optimizer states** across data-parallel (DP) ranks. In a standard
data-parallel setup (DDP), every rank maintains a full replica of optimizer states
(e.g., Adam's momentum and variance buffers), which is redundant. ZeRO-1 eliminates
this redundancy by assigning each parameter to exactly one DP rank for optimizer state
storage and updates.

### Memory Savings

For Adam optimizer with mixed-precision training, optimizer state per parameter includes:
- `fp32` master weight copy: 4 bytes/element
- momentum (`m`): 4 bytes/element
- variance (`v`): 4 bytes/element

Total: **12 bytes/element** (vs. 2 bytes/element for the `bf16` parameter itself).

ZeRO-1 reduces optimizer state memory by a factor of `dp_degree`:

| DP degree | Optimizer state memory per rank | Reduction |
|-----------|--------------------------------|-----------|
| 1         | 100%                           | —         |
| 2         | 50%                            | 2×        |
| 4         | 25%                            | 4×        |
| 8         | 12.5%                          | 8×        |

---

## Integration with Piper's PP + DP Setup

Piper uses **pipeline parallelism** (PP) with multiple stages distributed across
pipeline ranks, optionally combined with **data parallelism** (DP) by replicating
the entire pipeline. Each actor (`PiperActor`) corresponds to one pipeline rank within
one DP replica.

ZeRO-1 operates across the DP dimension: actors at the same PP rank but different DP
ranks form a DP group and split optimizer states among themselves.

```
           DP rank 0          DP rank 1
           (replica 0)        (replica 1)
              │                   │
PP rank 0: Actor(0,0)         Actor(0,1)   ← same stage, different DP replica
              │                   │            owns first half of optimizer states  owns second half
PP rank 1: Actor(1,0)         Actor(1,1)
              │                   │
PP rank 2: Actor(2,0)         Actor(2,1)
```

Each actor participates independently in ZeRO-1 for its own stage's parameters.

---

## Communication Pattern per Training Step

ZeRO-1 introduces two additional collective communication operations per optimizer step,
replacing the standard DDP all-reduce:

```
Standard DDP:
  backward → all-reduce(grads) → optimizer.step() → next forward

ZeRO-1:
  backward → all-reduce(grads) → optimizer.step(local partition) → broadcast(updated params) → next forward
```

### Step-by-Step

1. **Forward pass** — identical to DDP; all ranks have full parameters.

2. **Backward pass** — identical to DDP; all ranks compute gradients for all parameters.

3. **Gradient all-reduce** — all-reduce gradients across DP ranks (same as DDP).
   Every rank ends up with the correctly averaged gradient for every parameter.
   *(Future optimization: replace with reduce-scatter to also partition gradient buffers,
   saving an additional `1 - 1/dp_degree` of gradient memory.)*

4. **Optimizer step (local partition only)** — each rank steps the optimizer only for
   its own parameter partition. Non-local parameters are not touched.

5. **Parameter all-gather** — each rank broadcasts its updated parameter partition
   to all other DP ranks so that every rank has a complete, up-to-date copy of all
   parameters before the next forward pass.

---

## Parameter Partitioning Strategy

Parameters are assigned to DP ranks using **round-robin by parameter index** within
each pipeline stage. For a stage with parameters `[p0, p1, ..., pN]` and `dp_degree`
DP ranks:

- DP rank 0 owns: `{p0, p_dp_degree, p_{2*dp_degree}, ...}`
- DP rank 1 owns: `{p1, p_{dp_degree+1}, ...}`
- ...

This guarantees:
- Each parameter is assigned to exactly **one** DP rank.
- No parameter is split across ranks (important for optimizer correctness).
- Balanced assignment when parameters have similar sizes; easy to extend with
  size-aware greedy assignment for better load balancing.

The owning rank is simply `param_index % dp_degree`.

---

## Implementation Architecture

### New file: `src/piper_zero1.py`

Contains `ZeROOneState`, a self-contained class that encapsulates all ZeRO-1 logic
for a single pipeline stage:

```
ZeROOneState
├── __init__(params, dp_rank, dp_degree, dp_group, device, dtype)
│     └── _partition_params()  — assign params to DP ranks round-robin
├── create_optimizer(optim_class, **kwargs) -> Optimizer
│     └── returns optimizer managing only local_params
├── all_reduce_gradients(comm_stream)
│     └── all-reduce all params' gradients across dp_group
├── all_gather_params(comm_stream)
│     └── broadcast each param from its owner to all dp_group members
└── local_params: list[Parameter]   — params owned by this DP rank
    all_params:   list[Parameter]   — all params for this stage
```

### Modifications to `src/piper_actor.py`

| Location | Change |
|----------|--------|
| `__init__` | Accept `zero_stage: int = 0`; store per-stage `ZeROOneState` in `self.zero1_states` |
| `_load_stage` | If `zero_stage >= 1`: create `ZeROOneState`, use it to create the optimizer |
| `_prepare_dp_comm_ops` | Skip if using ZeRO (no per-param post-accumulate hooks needed) |
| `_update` | If `zero_stage >= 1`: call `zero1_state.all_reduce_gradients()` then `all_gather_params()` instead of DDP sync |

### Modifications to `src/piper_compile.py`

`piper_setup` and `_create_actors` accept a new `zero_stage: int = 0` parameter
and forward it to `PiperActor`.

---

## Design Decisions and Trade-offs

### Why all-reduce instead of reduce-scatter for gradients?

A true ZeRO-1 implementation uses **reduce-scatter** for gradient synchronization
(each rank receives summed gradients only for its owned partition), which also saves
`1 - 1/dp_degree` of gradient memory. However, reduce-scatter over unevenly-sized
parameter partitions requires explicit padding and flat-buffer bookkeeping that
complicates the implementation.

The current implementation uses **all-reduce** (like DDP) so gradients are available
on all ranks. Only the optimizer step and subsequent all-gather differ from DDP.
This still achieves the primary goal of ZeRO-1: **optimizer state memory savings**.

Gradient memory savings via reduce-scatter can be added as a follow-up (ZeRO-1.5).

### Why round-robin parameter assignment?

Round-robin is simple, deterministic, and avoids communication overhead at partition
setup time. Size-aware (greedy) assignment would minimize load imbalance but requires
knowledge of all params' sizes upfront.

Round-robin can be imbalanced when parameter sizes vary significantly (e.g., a large
embedding followed by small linear layers). This is acceptable for an initial
implementation and can be improved later.

### Interaction with `naive_gradient_sync` and smart-mode hooks

- **`naive_gradient_sync=True`**: Calls `_synchronize_gradients()` which all-reduces
  all gradients. ZeRO-1 replaces this path entirely.
- **`naive_gradient_sync=False`** (smart mode): Uses `register_post_accumulate_grad_hook`
  to trigger all-reduce after all microbatches have contributed. ZeRO-1 disables these
  hooks and instead synchronizes in `_update()` (simpler, no overlap).

  Overlapping gradient sync with computation remains possible as a future optimization.

### Interaction with BWD_I / BWD_W split backward

The split backward (`BWD_I` + `BWD_W`) accumulates gradients onto `param.grad` tensors
via `torch.autograd.grad` and manual accumulation, identical to the non-split case.
ZeRO-1 operates after all backward passes have completed (in `_update()`), so it is
transparent to the split backward mechanism.

---

## Example Usage

```python
from src.piper_compile import piper_setup
from src.piper_exec import piper_exec

# Use ZeRO-1 by passing zero_stage=1
piper_setup(
    model=model,
    optim_fn=lambda params: torch.optim.AdamW(params, lr=1e-3),
    example_inputs=example_inputs,
    example_outputs=example_outputs,
    schedule=schedule,
    zero_stage=1,          # <-- enable ZeRO-1
)

# Execution is unchanged
losses = piper_exec(schedule, loss_fn=loss_fn, dp_degree=dp_degree)
```

---

## Future Extensions

- **ZeRO-2**: Partition gradient buffers in addition to optimizer states.
  Implementation: replace all-reduce with reduce-scatter; free non-local gradient
  buffers immediately after reduce.

- **ZeRO-3**: Also partition parameters, fetching them on-demand during forward/backward.
  Requires significant changes to the forward pass and P2P communication logic.

- **Bucket-based communication**: Group small parameters into buckets for fewer
  collective ops, amortizing launch overhead.

- **Overlap with computation**: Trigger all-gather during the forward pass
  of the next microbatch to hide latency.
