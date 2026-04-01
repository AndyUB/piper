#!/bin/bash
# DP=2, PP=1, ZeRO=0: no pipeline parallelism, plain all-reduce.
# Use this to isolate the PP overhead from ZeRO overhead.
# Total tokens per DP-rank per iter = batch_size * seq_len = 16 * 256 = 4096
set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MODEL="${1:-3b}"
NSIGHT="${NSIGHT:-0}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/piper_dp_nozero_nopp_${MODEL}_${TIMESTAMP}.log"

RAY_TMP="${RAY_TMP:-/tmp/ray}"
mkdir -p "$RAY_TMP"

export RAY_TMPDIR="$RAY_TMP"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

echo "Logging to $LOG_FILE"
python3 -m test.test_llama \
  --dp 2 \
  --pp 1 \
  --warmup 2 \
  --iters 5 \
  --schedule no-pp-4s \
  --mbs 1 \
  --model "$MODEL" \
  --zero-stage 0 \
  ${NSIGHT:+--nsight} \
  >"$LOG_FILE" 2>&1

echo "Run complete. Log: $LOG_FILE"

if [[ "${NSIGHT:-0}" == "1" ]]; then
    sleep 30
    NSYS_OUT_DIR="nsys_traces/dp_nozero_nopp_${MODEL}_${TIMESTAMP}"
    mkdir -p "$NSYS_OUT_DIR"
    SESSION_DIR=$(ls -td "${RAY_TMP}/ray"/session_* 2>/dev/null | head -1)
    if [ -d "${SESSION_DIR:-}" ]; then
        NSIGHT_DIR="$SESSION_DIR/logs/nsight"
        if [ -d "$NSIGHT_DIR" ]; then
            mapfile -t NSYS_FILES < <(find "$NSIGHT_DIR" -name "*.nsys-rep" -o -name "*.sqlite" 2>/dev/null)
            if [ "${#NSYS_FILES[@]}" -gt 0 ]; then
                for f in "${NSYS_FILES[@]}"; do
                    cp "$f" "$NSYS_OUT_DIR/"
                    echo "  Copied: $(basename "$f")"
                done
                echo "Nsys traces copied to $NSYS_OUT_DIR"
            fi
        fi
    fi
fi
