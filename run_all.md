# Run scripts — post-merge sanity check

Run from the repo root (`piper_ref/`). Each command waits for the previous to finish.

## Baseline

```bash
bash run_baseline_llama.sh
```

## No-ZeRO (plain all-reduce)

```bash
bash run_dp_nozero_llama_nopp.sh
bash run_dp_nozero_llama.sh
bash run_dp_nozero_llama_bucketed.sh
```

## ZeRO-1

```bash
bash run_zero1_llama_nopp.sh
bash run_zero1_llama.sh
bash run_zero1_llama_bucketed.sh
```

## ZeRO-2

```bash
bash run_zero2_llama_nopp.sh
bash run_zero2_llama.sh
bash run_zero2_llama_bucketed.sh
```

## ZeRO-3

```bash
bash run_zero3_llama_nopp.sh
bash run_zero3_llama.sh
bash run_zero3_llama_bucketed.sh
```

## All at once (sequential)

```bash
bash run_baseline_llama.sh && \
bash run_dp_nozero_llama_nopp.sh && \
bash run_dp_nozero_llama.sh && \
bash run_dp_nozero_llama_bucketed.sh && \
bash run_zero1_llama_nopp.sh && \
bash run_zero1_llama.sh && \
bash run_zero1_llama_bucketed.sh && \
bash run_zero2_llama_nopp.sh && \
bash run_zero2_llama.sh && \
bash run_zero2_llama_bucketed.sh && \
bash run_zero3_llama_nopp.sh && \
bash run_zero3_llama.sh && \
bash run_zero3_llama_bucketed.sh
```
