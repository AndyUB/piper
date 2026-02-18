# nccl_stream_priority_exp.py
import os
import time
import threading
import argparse
import numpy as np

import torch
import torch.distributed as dist

import cupy as cp
from cupy.cuda import nccl, runtime


def make_priority_stream(priority: int):
    """
    Create a CUDA stream with a given priority and wrap it as a CuPy ExternalStream.
    Returns (stream_handle, cupy_stream).
    """
    # flags = 0  # default stream flags
    # s_handle = runtime.streamCreateWithPriority(flags, priority)
    # s = cp.cuda.ExternalStream(s_handle)
    # return s_handle, s
    tstream = torch.cuda.Stream(priority=priority)
    cstream = cp.cuda.ExternalStream(tstream.cuda_stream)  # cudaStream_t pointer
    return tstream, cstream


def create_nccl_comm(world_size, rank):
    uid_list = [nccl.get_unique_id()] if rank == 0 else [None]
    dist.broadcast_object_list(uid_list, src=0)
    uid = uid_list[0]
    return nccl.NcclCommunicator(world_size, uid, rank)


def nccl_allreduce_inplace(comm, buf: cp.ndarray, stream: cp.cuda.Stream):
    # In-place allreduce: sendbuff == recvbuff
    # dtype/op constants in CuPy map to NCCL datatypes/ops
    dt = getattr(nccl, "NCCL_FLOAT32", None)
    op = getattr(nccl, "NCCL_SUM", None)
    if dt is None or op is None:
        raise RuntimeError(
            "Expected CuPy NCCL constants NCCL_FLOAT32 and NCCL_SUM to exist."
        )

    comm.allReduce(
        buf.data.ptr,  # sendbuff
        buf.data.ptr,  # recvbuff
        buf.size,  # count
        dt,  # datatype
        op,  # op
        stream.ptr,  # cudaStream_t
    )


def run_phase(
    rank,
    world_size,
    *,
    use_priority: bool,
    bg_mb: int,
    sync_kb: int,
    sync_iters: int,
    warmup: int,
    sleep_ms: float,
):
    # Choose priorities
    # least_pri, greatest_pri = runtime.deviceGetStreamPriorityRange()
    # # Note: numerically "smaller" is higher priority; CUDA returns (least, greatest)
    # # where "greatest" is the highest priority stream value.
    # hi_pri = greatest_pri
    # lo_pri = least_pri
    # if not use_priority:
    #     # Make both streams same priority (use least_pri for both)
    #     hi_pri = least_pri
    #     lo_pri = least_pri

    lo_pri = 0
    hi_pri = -1

    if not use_priority:
        hi_pri = 0
        lo_pri = 0

    # Streams
    # hi_handle, hi_stream = make_priority_stream(hi_pri)
    # lo_handle, lo_stream = make_priority_stream(lo_pri)
    hi_tstream, hi_stream = make_priority_stream(hi_pri)
    lo_tstream, lo_stream = make_priority_stream(lo_pri)

    # Two communicators (background + sync)
    comm_bg = create_nccl_comm(world_size, rank)
    comm_sync = create_nccl_comm(world_size, rank)

    # Buffers
    # Background: large enough to create sustained bandwidth pressure
    bg_elems = (bg_mb * 1024 * 1024) // 4  # float32
    bg_buf = cp.ones(bg_elems, dtype=cp.float32)

    # Sync: small latency-sensitive message
    sync_elems = (sync_kb * 1024) // 4
    sync_buf = cp.ones(sync_elems, dtype=cp.float32)

    # Background loop thread
    stop_evt = threading.Event()
    bg_err = {"exc": None}

    def bg_loop():
        try:
            while not stop_evt.is_set():
                # Launch a background collective and wait for it to finish,
                # so we keep ~one background op "in flight" continuously.
                nccl_allreduce_inplace(comm_bg, bg_buf, lo_stream)
                lo_stream.synchronize()
        except Exception as e:
            bg_err["exc"] = e
            stop_evt.set()

    t = threading.Thread(target=bg_loop, daemon=True)
    t.start()

    # Sync timing
    dist.barrier()
    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()

    times_ms = []
    for i in range(sync_iters + warmup):
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        # Measure elapsed GPU time on hi_stream for the sync collective.
        start_evt.record(hi_stream)
        nccl_allreduce_inplace(comm_sync, sync_buf, hi_stream)
        end_evt.record(hi_stream)
        end_evt.synchronize()
        ms = cp.cuda.get_elapsed_time(start_evt, end_evt)

        if i >= warmup:
            times_ms.append(ms)

    # Stop background
    stop_evt.set()
    t.join(timeout=5)

    dist.barrier()

    # Cleanup streams
    # runtime.streamDestroy(hi_handle)
    # runtime.streamDestroy(lo_handle)
    del hi_tstream, lo_tstream

    if bg_err["exc"] is not None:
        raise bg_err["exc"]

    times_ms = np.array(times_ms, dtype=np.float64)
    stats = {
        "mean": float(times_ms.mean()),
        "p50": float(np.percentile(times_ms, 50)),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99)),
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bg-mb", type=int, default=64, help="Background allreduce size (MB) per op"
    )
    parser.add_argument(
        "--sync-kb", type=int, default=256, help="Sync allreduce size (KB) per op"
    )
    parser.add_argument(
        "--sync-iters", type=int, default=200, help="Number of measured sync ops"
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Warmup sync ops (not measured)"
    )
    parser.add_argument(
        "--sleep-ms",
        type=float,
        default=0.0,
        help="Sleep between sync ops to simulate spacing",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    cp.cuda.Device(local_rank).use()

    if rank == 0:
        print(f"[info] world_size={world_size}")
        print(
            f"[info] Recommended: export NCCL_LAUNCH_ORDER_IMPLICIT=1 (NCCL>=2.26) for multi-communicator safety."
        )

    dist.barrier()

    # Phase A: no priority
    stats_no = run_phase(
        rank,
        world_size,
        use_priority=False,
        bg_mb=args.bg_mb,
        sync_kb=args.sync_kb,
        sync_iters=args.sync_iters,
        warmup=args.warmup,
        sleep_ms=args.sleep_ms,
    )

    # Phase B: with priority
    stats_pri = run_phase(
        rank,
        world_size,
        use_priority=True,
        bg_mb=args.bg_mb,
        sync_kb=args.sync_kb,
        sync_iters=args.sync_iters,
        warmup=args.warmup,
        sleep_ms=args.sleep_ms,
    )

    # Aggregate (average of per-rank means/p50/p95/p99)
    def allreduce_avg(x: float) -> float:
        t = torch.tensor([x], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float((t / world_size).item())

    agg_no = {k: allreduce_avg(v) for k, v in stats_no.items()}
    agg_pri = {k: allreduce_avg(v) for k, v in stats_pri.items()}

    if rank == 0:

        def fmt(s):
            return f"mean={s['mean']:.3f} ms  p50={s['p50']:.3f}  p95={s['p95']:.3f}  p99={s['p99']:.3f}"

        print("\n=== Sync-op latency under background NCCL traffic ===")
        print(f"[no priority] {fmt(agg_no)}")
        print(f"[priority]    {fmt(agg_pri)}")
        print(
            f"[speedup]     mean x{(agg_no['mean']/agg_pri['mean']):.2f}, p95 x{(agg_no['p95']/agg_pri['p95']):.2f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
