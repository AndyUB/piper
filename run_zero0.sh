#!/bin/bash
set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/piper_zero1_${TIMESTAMP}.log"

RAY_TMP="${RAY_TMP:-/tmp/ray}"
mkdir -p "$RAY_TMP"

export RAY_TMPDIR="$RAY_TMP"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export PIPER_MASTER_PORT="${PIPER_MASTER_PORT:-12388}"

echo "Logging to $LOG_FILE"
python3 -m test.test_llama \
  --dp 1 \
  --pp 2 \
  --warmup 5 \
  --iters 10 \
  --schedule interleaved-1f1b \
  --mbs 4 \
  --model 3b \
  --zero_stage 0 \
  >"$LOG_FILE" 2>&1
