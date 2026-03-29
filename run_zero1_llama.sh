#!/bin/bash
set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/piper_zero1_${TIMESTAMP}.log"

RAY_TMP="${RAY_TMP:-/tmp/ray}"
mkdir -p "$RAY_TMP"

export RAY_TMPDIR="$RAY_TMP"
export CUDA_VISIBLE_DEVICES="5,6,2,0"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

echo "Logging to $LOG_FILE"
python3 -m test.test_llama \
  --dp 2 \
  --pp 2 \
  --warmup 5 \
  --iters 10 \
  --schedule interleaved-1f1b \
  --mbs 4 \
  --model debug \
  --zero-stage 1 \
  >"$LOG_FILE" 2>&1
