#!/bin/bash

LOG_DIR=logs
mkdir -p $LOG_DIR
RAY_TMP=$HOME/ray_tmp
mkdir -p $RAY_TMP
export RAY_TMPDIR="$RAY_TMP"
# export CUDA_VISIBLE_DEVICES=0,1,2 # worked for LLAMA_DEBUG
# export CUDA_VISIBLE_DEVICES=5,6,7 # fails for LLAMA_1B and LLAMA_8B
# export CUDA_VISIBLE_DEVICES=4,5,6 # fails for LLAMA_1B, LLAMA_DEBUG
export PYTHONUNBUFFERED=1

# python3 -m test.test_llama \
#     --iters 5 \
#     --num_stages 4 \
#     --schedule interleaved-1f1b \
#     --model LLAMA_DEBUG > $LOG_DIR/interleaved_1f1b_debug.log 2>&1

# python3 -m test.test_llama \
#     --iters 5 \
#     --num_stages 4 \
#     --schedule interleaved-1f1b \
#     --model LLAMA_1B > $LOG_DIR/interleaved_1f1b_1b_gpu012.log 2>&1

# python3 -m test.test_llama \
#     --iters 5 \
#     --num_stages 4 \
#     --schedule interleaved-1f1b \
#     --model LLAMA_3B > $LOG_DIR/interleaved_1f1b_3b_gpu012.log 2>&1

# python3 -m test.test_llama \
#     --iters 5 \
#     --num_stages 4 \
#     --schedule interleaved-1f1b \
#     --tracing \
#     --model LLAMA_3B > $LOG_DIR/interleaved_1f1b_3b_gpu012-nsys.trace 2>&1

# python3 -m test.test_llama \
#     --iters 5 \
#     --num_stages 4 \
#     --schedule interleaved-1f1b \
#     --model LLAMA_3B > $LOG_DIR/interleaved_1f1b_3b_gpu012-nsys.log 2>&1

python3 -m test.test_llama \
    --warmup_iters 1 \
    --iters 1 \
    --num_stages 4 \
    --schedule interleaved-1f1b \
    --model LLAMA_3B > $LOG_DIR/interleaved_1f1b_3b_gpu012-nsys.log 2>&1

# python3 -m test.test_llama \
#     --iters 5 \
#     --num_stages 4 \
#     --schedule interleaved-1f1b \
#     --model LLAMA_8B > $LOG_DIR/interleaved_1f1b_8b_gpu012.log 2>&1

# python3 -m test.test_llama \
#     --iters 5 \
#     --devices 2 \
#     --num_stages 4 \
#     --schedule interleaved-1f1b \
#     --model LLAMA_DEBUG > $LOG_DIR/interleaved_1f1b.log 2>&1
