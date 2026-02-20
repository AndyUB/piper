#!/bin/bash
LOG_DIR=logs
mkdir -p $LOG_DIR
# RAY_TMP=$HOME/ray_tmp
RAY_TMP=/m-coriander/coriander/tmp/ray
mkdir -p $RAY_TMP
export RAY_TMPDIR="$RAY_TMP"
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=0,1,4
export RAY_DEDUP_LOGS=0
export PIPER_MASTER_PORT=12389

# python3 -m test.test_llama --iters 5 --schedule interleaved-1f1b --mbs 4 > $LOG_DIR/interleaved.log 2>&1
python3 -m test.test_llama \
    --warmup 1 \
    --iters 1 \
    --schedule interleaved-1f1b \
    --mbs 4 \
    --model debug > $LOG_DIR/interleaved_debug-memopt.log 2>&1
    # --tracing \