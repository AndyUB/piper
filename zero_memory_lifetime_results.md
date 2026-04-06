# ZeRO Memory Lifetime Redesign — Run Results

Corresponds to the code changes described in `zero_memory_lifetime_design.md`.  
All runs: Llama 3B, dp=2 (nopp: pp=1, pp: pp=2), warmup=2, iters=5.

## Run batches

| Batch | Scripts | GPUs |
|-------|---------|------|
| 1 | zero1_nopp, zero2_nopp, zero3_nopp | 0,1 / 2,3 / 4,5 |
| 2 | zero1 (pp), zero2 (pp) | 0–3 / 4–7 |
| 3 | zero3 (pp) | 0–3 |
| 4 | zero1_bucketed, zero2_bucketed | 0–3 / 4–7 |
| 5 | zero3_bucketed | 0–3 |

---

## Throughput (samples/sec)

> Higher is better. Extracted from log lines: `throughput` or `iter time`.

| Config | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|--------|--------|--------|--------|
| no-pp (dp=2, pp=1, mbs=1) | | | |
| interleaved-1f1b (dp=2, pp=2, mbs=4) | | | |
| interleaved-1f1b bucketed (dp=2, pp=2, mbs=4) | | | |

---

## Peak Memory (GB per GPU)

> Lower is better. Extracted from log lines: `peak_memory` or `torch.cuda.max_memory_allocated`.

| Config | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|--------|--------|--------|--------|
| no-pp (dp=2, pp=1, mbs=1) | | | |
| interleaved-1f1b (dp=2, pp=2, mbs=4) | | | |
| interleaved-1f1b bucketed (dp=2, pp=2, mbs=4) | | | |

---

## Iteration Time (ms)

> Lower is better.

| Config | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|--------|--------|--------|--------|
| no-pp (dp=2, pp=1, mbs=1) | | | |
| interleaved-1f1b (dp=2, pp=2, mbs=4) | | | |
| interleaved-1f1b bucketed (dp=2, pp=2, mbs=4) | | | |

---

## Notes / Observations

- (Populated as results come in)

---

## Raw Log Summaries

### Batch 1: no-pp runs

#### zero1_nopp

```
(paste relevant log lines here)
```

#### zero2_nopp

```
(paste relevant log lines here)
```

#### zero3_nopp

```
(paste relevant log lines here)
```

### Batch 2: interleaved-1f1b pp runs

#### zero1 (pp)

```
(paste relevant log lines here)
```

#### zero2 (pp)

```
(paste relevant log lines here)
```

### Batch 3

#### zero3 (pp)

```
(paste relevant log lines here)
```

### Batch 4: bucketed pp runs

#### zero1_bucketed

```
(paste relevant log lines here)
```

#### zero2_bucketed

```
(paste relevant log lines here)
```

### Batch 5

#### zero3_bucketed

```
(paste relevant log lines here)
```
