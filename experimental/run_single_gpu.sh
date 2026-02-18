export CUDA_VISIBLE_DEVICES=6,7
export PYTHONUNBUFFERED=1
python priority_sanity_single_gpu.py > priority_single.log 2>&1