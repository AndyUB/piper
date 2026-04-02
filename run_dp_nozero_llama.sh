#!/bin/bash
# DP=2, PP=2, ZeRO=0 baseline: same topology as ZeRO runs but with plain all-reduce
# (no parameter sharding). Use this to isolate ZeRO communication overhead.
# Total tokens per DP-rank per iter = batch_size * mbs * seq_len = 16 * 4 * 256 = 16384
set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MODEL="${1:-3b}"
NSIGHT="${NSIGHT:-1}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/piper_dp_nozero_${MODEL}_${TIMESTAMP}.log"

RAY_TMP="${RAY_TMP:-/m-coriander/coriander/wxdeng/ray_tmp}"
mkdir -p "$RAY_TMP"

export RAY_TMPDIR="$RAY_TMP"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

echo "Logging to $LOG_FILE"
python3 -m test.test_llama \
  --dp 2 \
  --pp 2 \
  --warmup 2 \
  --iters 5 \
  --schedule interleaved-1f1b \
  --mbs 4 \
  --model "$MODEL" \
  --zero-stage 0 \
  ${NSIGHT:+--nsight} \
  >"$LOG_FILE" 2>&1

echo "Run complete. Log: $LOG_FILE"

NSYS_OUT_DIR="nsys_traces/dp_nozero_${MODEL}_${TIMESTAMP}"
mkdir -p "$NSYS_OUT_DIR"

SESSION_DIR=$(ls -td "${RAY_TMP}/ray"/session_* 2>/dev/null | head -1)
if [ -d "${SESSION_DIR:-}" ]; then
    echo "Ray session: $SESSION_DIR"
    NSIGHT_DIR="$SESSION_DIR/logs/nsight"
    if [ -d "$NSIGHT_DIR" ]; then
        echo "Waiting for nsys to finish flushing..."
        for i in $(seq 1 60); do
            if find "$NSIGHT_DIR" -name "*.nsys-rep" -size +0c 2>/dev/null | grep -q .; then
                break
            fi
            sleep 5
        done
        mapfile -t NSYS_FILES < <(find "$NSIGHT_DIR" -name "*.nsys-rep" -o -name "*.sqlite" 2>/dev/null)
        if [ "${#NSYS_FILES[@]}" -gt 0 ]; then
            for f in "${NSYS_FILES[@]}"; do
                cp "$f" "$NSYS_OUT_DIR/"
                echo "  Copied: $(basename "$f")"
            done
            echo "Nsys traces copied to $NSYS_OUT_DIR"
        else
            echo "No nsys trace files found in $NSIGHT_DIR"
        fi
    else
        echo "No nsight directory found at $NSIGHT_DIR"
    fi
else
    echo "No Ray session directory found under ${RAY_TMP}/ray"
fi
