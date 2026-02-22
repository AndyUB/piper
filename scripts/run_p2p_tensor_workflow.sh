#!/usr/bin/env bash
set -euo pipefail

WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-1}"
MBS="${MBS:-4}"
SCHEDULE="${SCHEDULE:-interleaved-1f1b}"
MODEL="${MODEL:-debug}"
LOG_DIR="${LOG_DIR:-./logs}"
P2P_TENSOR_LOG_DIR="${P2P_TENSOR_LOG_DIR:-$LOG_DIR/p2p_tensors}"

mkdir -p "$LOG_DIR" "$P2P_TENSOR_LOG_DIR"

export PIPER_P2P_TENSOR_LOG_DIR="$P2P_TENSOR_LOG_DIR"

RUN_LOG="$LOG_DIR/${SCHEDULE}_${MODEL}-memopt.log"

echo "Running Piper with warmup=$WARMUP iters=$ITERS mbs=$MBS schedule=$SCHEDULE model=$MODEL"
python3 -m test.test_llama \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --schedule "$SCHEDULE" \
    --mbs "$MBS" \
    --model "$MODEL" > "$RUN_LOG" 2>&1

echo "Comparing send/recv P2P tensors in $P2P_TENSOR_LOG_DIR"
python3 scripts/compare_p2p_tensors.py --log-dir "$P2P_TENSOR_LOG_DIR"

echo "Workflow completed successfully"
