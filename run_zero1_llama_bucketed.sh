#!/bin/bash
set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MODEL="${1:-3b}"
NSIGHT="${NSIGHT:-1}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/piper_zero1_bucketed_${MODEL}_${TIMESTAMP}.log"

RAY_TMP="${RAY_TMP:-/tmp/ray}"
mkdir -p "$RAY_TMP"

export RAY_TMPDIR="$RAY_TMP"
export CUDA_VISIBLE_DEVICES="0,2,3,4"
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
  --zero-stage 1 \
  --bucketing \
  --tracing \
  ${NSIGHT:+--nsight} \
  >"$LOG_FILE" 2>&1

echo "Run complete. Log: $LOG_FILE"

# Wait for nsys to finish flushing trace data before copying
sleep 30

# Copy nsys traces out of Ray's temp directory
# Ray places nsys-rep files in $RAY_TMPDIR/ray/session_*/logs/nsight/
NSYS_OUT_DIR="nsys_traces/zero1_bucketed_${MODEL}_${TIMESTAMP}"
mkdir -p "$NSYS_OUT_DIR"

SESSION_DIR=$(ls -td "${RAY_TMP}/ray"/session_* 2>/dev/null | head -1)
if [ -d "${SESSION_DIR:-}" ]; then
    echo "Ray session: $SESSION_DIR"
    NSIGHT_DIR="$SESSION_DIR/logs/nsight"
    if [ -d "$NSIGHT_DIR" ]; then
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
