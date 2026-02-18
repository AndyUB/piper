export CUDA_VISIBLE_DEVICES=6,7
export PYTHONUNBUFFERED=1
python nccl_priority_mp.py --nproc 2 > nccl_priority_mp.log 2>&1
