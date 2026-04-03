#!/bin/bash

LOG_DIR=logs
mkdir -p $LOG_DIR

export CUDA_VISIBLE_DEVICES=4,5
export PYTHONUNBUFFERED=1

python3 -m test.test_llama \
    --iters 5 \
    --devices 2 \
    --num_stages 4 \
    --schedule interleaved-1f1b \
    --tracing \
    --model LLAMA_DEBUG > $LOG_DIR/interleaved_1f1b.log 2>&1
