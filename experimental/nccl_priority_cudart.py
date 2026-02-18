# nccl_priority_mp_cudart.py
import os
import socket
import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import cupy as cp
from cupy.cuda import nccl

import cuda.bindings.runtime as cudart


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _check(err, msg=""):
    # cuda-python returns (err, ...) where err is a cudaError_t integer
    if err != 0:
        raise RuntimeError(f"CUDA Runtime error {err} {msg}")


def create_nccl_comm(world_size, rank):
    uid_list = [nccl.get_unique_id()] if rank == 0 else [None]
    dist.broadcast_object_list(uid_list, src=0)
    return nccl.NcclCommunicator(world_size, uid_list[0], rank)


def nccl_allreduce_inplace(comm, buf: cp.ndarray, torch_stream: torch.cuda.Stream):
    comm.allReduce(
        buf.data.ptr,
        buf.data.ptr,
        buf.size,
        nccl.NCCL_FLOAT32,
        nccl.NCCL_SUM,
        torch_stream.cuda_stream,  # cudaStream_t
    )


def make_external_torch_stream(priority: int):
    # Create CUDA stream with priority using cuda.bindings.runtime (cudart),
    # then wrap as a Torch stream.
    err, stream_ptr = cudart.cudaStreamCreateWithPriority(
        cudart.cudaStreamNonBlocking, priority
    )
    _check(err, "cudaStreamCreateWithPriority")

    # Wrap external stream ptr as torch stream.
    # NOTE: PyTorch does NOT manage the stream lifetime.
    tstream = torch.cuda.get_stream_from_external(stream_ptr)
    return stream_ptr, tstream


def run_phase(
    rank,
    world_size,
    *,
    use_priority: bool,
    bg_mb: int,
    sync_kb: int,
    sync_iters: int,
    warmup: int,
    prefill_bg: int,
    bg_per_iter: int,
    drain_every: int,
):

    # Query stream priority range: (leastPriority, greatestPriority)
    err, least_pri, greatest_pri = cudart.cudaDeviceGetStreamPriorityRange()
    _check(err, "cudaDeviceGetStreamPriorityRange")

    # lower number => higher priority
    if use_priority:
        hi_pri = greatest_pri
        lo_pri = least_pri
    else:
        hi_pri = least_pri
        lo_pri = least_pri

    hi_ptr, hi_stream = make_external_torch_stream(hi_pri)
    lo_ptr, lo_stream = make_external_torch_stream(lo_pri)

    comm = create_nccl_comm(world_size, rank)

    bg_elems = (bg_mb * 1024 * 1024) // 4  # float32
    sync_elems = (sync_kb * 1024) // 4  # float32
    bg_buf = cp.ones(bg_elems, dtype=cp.float32)
    sync_buf = cp.ones(sync_elems, dtype=cp.float32)

    # Prime background queue (enqueue only; no sync)
    for _ in range(prefill_bg):
        nccl_allreduce_inplace(comm, bg_buf, lo_stream)

    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []

    for i in range(sync_iters + warmup):
        # enqueue background work (no immediate sync)
        for _ in range(bg_per_iter):
            nccl_allreduce_inplace(comm, bg_buf, lo_stream)

        # time "sync" op on hi stream
        with torch.cuda.stream(hi_stream):
            start.record()
        nccl_allreduce_inplace(comm, sync_buf, hi_stream)
        with torch.cuda.stream(hi_stream):
            end.record()

        end.synchronize()
        ms = start.elapsed_time(end)
        if i >= warmup:
            times_ms.append(ms)

        # periodically drain background queue
        if drain_every > 0 and (i + 1) % drain_every == 0:
            lo_stream.synchronize()

    # final drain
    lo_stream.synchronize()
    hi_stream.synchronize()
    dist.barrier()

    # destroy CUDA streams we created (torch wrapper doesn't own them)
    (err,) = cudart.cudaStreamDestroy(hi_ptr)
    _check(err, "cudaStreamDestroy(hi)")
    (err,) = cudart.cudaStreamDestroy(lo_ptr)
    _check(err, "cudaStreamDestroy(lo)")

    arr = np.array(times_ms, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def worker(local_rank: int, world_size: int, master_addr: str, master_port: int, args):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=local_rank,
        world_size=world_size,
    )

    torch.cuda.set_device(local_rank)
    cp.cuda.Device(local_rank).use()

    if local_rank == 0:
        err, least_pri, greatest_pri = cudart.cudaDeviceGetStreamPriorityRange()
        _check(err, "cudaDeviceGetStreamPriorityRange")
        print(f"[info] world_size={world_size}")
        print(
            f"[info] cudart priority range: least={least_pri}, greatest={greatest_pri} (lower=more priority)"
        )

    dist.barrier()

    stats_no = run_phase(
        local_rank,
        world_size,
        use_priority=False,
        bg_mb=args.bg_mb,
        sync_kb=args.sync_kb,
        sync_iters=args.sync_iters,
        warmup=args.warmup,
        prefill_bg=args.prefill_bg,
        bg_per_iter=args.bg_per_iter,
        drain_every=args.drain_every,
    )

    stats_pri = run_phase(
        local_rank,
        world_size,
        use_priority=True,
        bg_mb=args.bg_mb,
        sync_kb=args.sync_kb,
        sync_iters=args.sync_iters,
        warmup=args.warmup,
        prefill_bg=args.prefill_bg,
        bg_per_iter=args.bg_per_iter,
        drain_every=args.drain_every,
    )

    # average across ranks
    def avg(x: float) -> float:
        t = torch.tensor([x], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float((t / world_size).item())

    agg_no = {k: avg(v) for k, v in stats_no.items()}
    agg_pri = {k: avg(v) for k, v in stats_pri.items()}

    if local_rank == 0:

        def fmt(s):
            return f"mean={s['mean']:.3f} ms  p50={s['p50']:.3f}  p95={s['p95']:.3f}  p99={s['p99']:.3f}"

        print("\n=== Sync-op latency under queued background NCCL traffic ===")
        print(f"[no priority] {fmt(agg_no)}")
        print(f"[priority]    {fmt(agg_pri)}")
        print(
            f"[speedup]     mean x{(agg_no['mean']/agg_pri['mean']):.2f}, p95 x{(agg_no['p95']/agg_pri['p95']):.2f}"
        )

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--bg-mb", type=int, default=64)
    ap.add_argument("--sync-kb", type=int, default=256)
    ap.add_argument("--sync-iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--prefill-bg", type=int, default=4)
    ap.add_argument("--bg-per-iter", type=int, default=1)
    ap.add_argument("--drain-every", type=int, default=8)
    args = ap.parse_args()

    if torch.cuda.device_count() < args.nproc:
        raise RuntimeError(
            f"Need >= {args.nproc} GPUs, found {torch.cuda.device_count()}"
        )

    master_addr = "127.0.0.1"
    master_port = _free_port()

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(args.nproc, master_addr, master_port, args),
        nprocs=args.nproc,
        join=True,
    )


if __name__ == "__main__":
    main()
