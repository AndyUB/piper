
# Resync code

```bash
rsync -av --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  --exclude='venv' --exclude='out' --exclude='ray_tmp' --exclude='aws' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  . ubuntu@$HEAD_PUBLIC_IP:/tmp/piper/

for WORKER_IP in $WORKER_PRIVATE_IP $WORKER2_PRIVATE_IP $WORKER3_PRIVATE_IP; do
  rsync -av --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
    --exclude='venv' --exclude='out' --exclude='ray_tmp' --exclude='aws' \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o 'ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP'" \
    . ubuntu@$WORKER_IP:/tmp/piper/
done
```

# Restart ray session after resync

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "
  docker pull $IMAGE
  docker stop piper_ray 2>/dev/null || true
  docker rm   piper_ray 2>/dev/null || true
  docker run -d --name piper_ray \
    --gpus all --ipc=host --shm-size=64g \
    --ulimit nofile=65536:65536 --ulimit memlock=-1:-1 --privileged \
    --device /dev/infiniband --network host \
    -v /opt/amazon/efa:/opt/amazon/efa:ro \
    -v /opt/amazon/ofi-nccl:/opt/aws-ofi-nccl:ro \
    -v /usr/lib/x86_64-linux-gnu:/opt/host-lib:ro \
    -e LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:/opt/host-lib \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    -e RDMAV_FORK_SAFE=1 \
    -e FI_EFA_FORK_SAFE=1 \
    -e NCCL_SOCKET_IFNAME=ens32 \
    -e GLOO_SOCKET_IFNAME=ens32 \
    -e NCCL_PROTO=simple \
    $IMAGE sleep infinity
  docker cp /tmp/piper piper_ray:/tmp/
  docker exec piper_ray bash -c 'ulimit -n 65536; ray stop || true; ray start --head --port=6379 --object-manager-port=8076 --dashboard-host=0.0.0.0 --num-gpus=8 --temp-dir=/tmp/piper/ray_tmp'
"

for WORKER_IP in $WORKER_PRIVATE_IP $WORKER2_PRIVATE_IP $WORKER3_PRIVATE_IP; do
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@$WORKER_IP "
    docker pull $IMAGE
    docker stop piper_ray 2>/dev/null || true
    docker rm   piper_ray 2>/dev/null || true
    docker run -d --name piper_ray \
      --gpus all --ipc=host --shm-size=64g \
      --ulimit nofile=65536:65536 --ulimit memlock=-1:-1 --privileged \
      --device /dev/infiniband --network host \
      -v /opt/amazon/efa:/opt/amazon/efa:ro \
      -v /opt/amazon/ofi-nccl:/opt/aws-ofi-nccl:ro \
      -v /usr/lib/x86_64-linux-gnu:/opt/host-lib:ro \
      -e LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:/opt/host-lib \
      -e FI_PROVIDER=efa \
      -e FI_EFA_USE_DEVICE_RDMA=1 \
      -e RDMAV_FORK_SAFE=1 \
      -e FI_EFA_FORK_SAFE=1 \
      -e NCCL_SOCKET_IFNAME=ens32 \
      -e GLOO_SOCKET_IFNAME=ens32 \
      -e NCCL_PROTO=simple \
      $IMAGE sleep infinity
    docker cp /tmp/piper piper_ray:/tmp/
    docker exec piper_ray bash -c 'ulimit -n 65536; ray stop || true; ray start --address=$HEAD_PRIVATE_IP:6379 --object-manager-port=8076 --num-gpus=8 --temp-dir=/tmp/piper/ray_tmp'
  "
done
```

# Qwen run command

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -e RAY_DEDUP_LOGS=0 -w /tmp/piper -e PYTHONPATH=/tmp/piper piper_ray \
   bash -c 'mkdir -p /tmp/piper/out && python3 -m test.test_qwen \
     --model 9B \
     --pp 8 \
     --dp 2 \
     --zero-stage 3 \
     --batch-size 8 \
     --seq-len 512 \
     --mbs 16 \
     --warmup 2 \
     --iters 2 \
     --address $HEAD_PRIVATE_IP \
     --port 6379 \
     --output-dir /tmp/piper/out \
     --nsight \
     --temp-dir /tmp/piper/ray_tmp > /tmp/piper/out/log 2>&1'"
```

# Memory breakdown debugging (ZeRO-2 vs ZeRO-3)

Run with `--memory-breakdown` flag to get detailed memory breakdown at each step.
Compare ZeRO-2 and ZeRO-3 runs to identify what's causing the memory increase.

```bash
# ZeRO-2 baseline
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -e RAY_DEDUP_LOGS=0 -w /tmp/piper -e PYTHONPATH=/tmp/piper piper_ray \
   bash -c 'mkdir -p /tmp/piper/out && python3 -m test.test_qwen \
     --model 9B \
     --pp 8 \
     --dp 2 \
     --zero-stage 2 \
     --batch-size 8 \
     --seq-len 512 \
     --mbs 16 \
     --warmup 1 \
     --iters 1 \
     --memory-breakdown \
     --address $HEAD_PRIVATE_IP \
     --port 6379 \
     --output-dir /tmp/piper/out \
     --temp-dir /tmp/piper/ray_tmp 2>&1 | tee /tmp/piper/out/zero2_memory_breakdown.log'"

# ZeRO-3 comparison
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -e RAY_DEDUP_LOGS=0 -w /tmp/piper -e PYTHONPATH=/tmp/piper piper_ray \
   bash -c 'mkdir -p /tmp/piper/out && python3 -m test.test_qwen \
     --model 9B \
     --pp 8 \
     --dp 2 \
     --zero-stage 3 \
     --batch-size 8 \
     --seq-len 512 \
     --mbs 16 \
     --warmup 1 \
     --iters 1 \
     --memory-breakdown \
     --address $HEAD_PRIVATE_IP \
     --port 6379 \
     --output-dir /tmp/piper/out \
     --temp-dir /tmp/piper/ray_tmp 2>&1 | tee /tmp/piper/out/zero3_memory_breakdown.log'"
```

# Copy outputs to coriander

```bash
mkdir -p out/ec2 out/ec2/nsight-head out/ec2/nsight-worker

# Copy log files (full log and training results log)
ssh -i $SSH_KEY ubuntu@$HEAD_PUBLIC_IP "docker cp piper_ray:/tmp/piper/out/. /tmp/out/"
scp -r -i $SSH_KEY ubuntu@$HEAD_PUBLIC_IP:/tmp/out/ out/ec2/

# Copy Nsight Systems profiles from the head node.
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "rm -rf /tmp/nsight-head && mkdir -p /tmp/nsight-head && docker cp piper_ray:/tmp/piper/ray_tmp/session_latest/logs/nsight/. /tmp/nsight-head/"
scp -r -i $SSH_KEY ubuntu@$HEAD_PUBLIC_IP:/tmp/nsight-head/. out/ec2/nsight-head/
```
