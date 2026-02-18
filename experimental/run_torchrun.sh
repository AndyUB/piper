# Strongly recommended for this experiment (multi-communicator concurrency):
export NCCL_LAUNCH_ORDER_IMPLICIT=1
export CUDA_VISIBLE_DEVICES=6,7
export NCCL_DEBUG=WARN

torchrun --standalone --nproc_per_node=2 nccl_stream_priority_exp.py \
  --bg-mb 1 --sync-kb 524288 --sync-iters 200 --warmup 20 > nccl_stream_priority_exp.log 2>&1
