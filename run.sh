#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-piper-container}"
GPU_DEVICES="${GPU_DEVICES:-0}"

docker run --rm -it \
  --gpus "\"device=${GPU_DEVICES}\"" \
  -e CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
  --network host \
  --shm-size=32g \
  -v "$PWD:/workspace/piper" \
  -v "$PWD/.venv/lib/python3.10/site-packages:/opt/venv/lib/python3.10/site-packages" \
  -w /workspace/piper \
  "${IMAGE}" \
  "${@:-bash}"

# Example of using this script
# GPU_DEVICES=0,1,2 ./run.sh python -m test.test_llama