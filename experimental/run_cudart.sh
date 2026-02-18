export CUDA_VISIBLE_DEVICES=6,7
export PYTHONUNBUFFERED=1
python nccl_priority_cudart.py --nproc 2 \
    --bg-mb 512 --bg-per-iter 4 --prefill-bg 16 --drain-every 1 \
    --sync-kb 256 --sync-iters 500 --warmup 50 > nccl_priority_cudart.log 2>&1
