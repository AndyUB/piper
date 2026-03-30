#!/bin/bash
# Single-GPU baseline: no pipeline parallelism, no data parallelism.
# Total tokens per iter = batch_size * mbs * seq_len = 16 * 1 * 256 = 4096
# (same seq_len as the parallelism runs; batch is smaller but single-device)
set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MODEL="${1:-debug}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/piper_baseline_${MODEL}_${TIMESTAMP}.log"

RAY_TMP="${RAY_TMP:-/tmp/ray}"
mkdir -p "$RAY_TMP"

export RAY_TMPDIR="$RAY_TMP"
export CUDA_VISIBLE_DEVICES="4"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

echo "Logging to $LOG_FILE"
python3 -m test.test_llama \
  --dp 1 \
  --pp 1 \
  --warmup 2 \
  --iters 5 \
  --schedule no-pp-4s \
  --mbs 1 \
  --batch-size 16 \
  --model "$MODEL" \
  --zero-stage 0 \
  >"$LOG_FILE" 2>&1

echo "Baseline run complete. Log: $LOG_FILE"
