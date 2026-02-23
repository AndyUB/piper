#!/bin/bash
LOG_DIR=logs
mkdir -p $LOG_DIR
# RAY_TMP=$HOME/ray_tmp
RAY_TMP=/m-coriander/coriander/tmp/ray
mkdir -p $RAY_TMP
export RAY_TMPDIR="$RAY_TMP"
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=5,7
export RAY_DEDUP_LOGS=0
export PIPER_MASTER_PORT=12389

python3 -m test.test_llama_piper_parity > $LOG_DIR/piper_parity.log 2>&1