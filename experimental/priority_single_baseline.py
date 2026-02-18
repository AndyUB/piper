# priority_sanity_single_gpu_3cases.py
import numpy as np
import torch
import cupy as cp
import cuda.bindings.runtime as cudart


def check(err, where):
    if err != 0:
        raise RuntimeError(f"CUDA error {err} at {where}")


def dev_attr(attr_name: str, device: int = 0) -> int:
    if hasattr(cudart, attr_name):
        attr = getattr(cudart, attr_name)
    else:
        attr = getattr(cudart.cudaDeviceAttr, attr_name)
    err, val = cudart.cudaDeviceGetAttribute(attr, device)
    check(err, f"cudaDeviceGetAttribute({attr_name})")
    return int(val)


SPIN_SRC = r"""
extern "C" __global__
void spin(unsigned long long cycles) {
    unsigned long long start = clock64();
    while ((clock64() - start) < cycles) { }
}
"""


def make_stream(priority: int):
    err, ptr = cudart.cudaStreamCreateWithPriority(
        cudart.cudaStreamNonBlocking, priority
    )
    check(err, "cudaStreamCreateWithPriority")
    s = torch.cuda.get_stream_from_external(ptr)
    return ptr, s


def destroy_stream(ptr: int):
    (err,) = cudart.cudaStreamDestroy(ptr)
    check(err, "cudaStreamDestroy")


def get_priority_range():
    err, least_pri, greatest_pri = cudart.cudaDeviceGetStreamPriorityRange()
    check(err, "cudaDeviceGetStreamPriorityRange")
    return least_pri, greatest_pri


def time_hi_only(hi_ms: float):
    """High-priority tiny work running alone (no interference)."""
    least_pri, greatest_pri = get_priority_range()
    hi_ptr, hi_stream = make_stream(greatest_pri)

    cp_hi = cp.cuda.ExternalStream(hi_stream.cuda_stream)

    clock_khz = dev_attr("cudaDevAttrClockRate")
    cycles_hi = np.uint64(clock_khz * hi_ms)
    spin = cp.RawKernel(SPIN_SRC, "spin")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    with torch.cuda.stream(hi_stream):
        start.record()
        spin((1,), (256,), (cycles_hi,), stream=cp_hi)
        end.record()

    end.synchronize()
    ms = start.elapsed_time(end)
    destroy_stream(hi_ptr)
    return ms


def time_with_hog(use_priority: bool, hog_ms: float, hi_ms: float):
    """
    Both streams release simultaneously:
      - low stream queues massive hog kernel
      - high stream queues tiny kernel (timed)
    If use_priority=False, both streams use same priority (interference baseline).
    If use_priority=True, high stream gets highest priority, low stream lowest.
    """
    least_pri, greatest_pri = get_priority_range()

    if use_priority:
        hi_pri = greatest_pri
        lo_pri = least_pri
    else:
        hi_pri = least_pri
        lo_pri = least_pri

    hi_ptr, hi_stream = make_stream(hi_pri)
    lo_ptr, lo_stream = make_stream(lo_pri)

    cp_hi = cp.cuda.ExternalStream(hi_stream.cuda_stream)
    cp_lo = cp.cuda.ExternalStream(lo_stream.cuda_stream)

    clock_khz = dev_attr("cudaDevAttrClockRate")
    sm_count = dev_attr("cudaDevAttrMultiProcessorCount")

    cycles_hog = np.uint64(clock_khz * hog_ms)
    cycles_hi = np.uint64(clock_khz * hi_ms)

    # Make hog huge so if it starts first, it can saturate the GPU.
    hog_blocks = int(sm_count * 4096)
    block = 256

    spin = cp.RawKernel(SPIN_SRC, "spin")

    gate = torch.cuda.Event(enable_timing=False)
    ctrl = torch.cuda.Stream()

    # Ensure both streams wait for the gate
    lo_stream.wait_event(gate)
    hi_stream.wait_event(gate)

    torch.cuda.synchronize()

    # Queue hog on low stream
    with torch.cuda.stream(lo_stream):
        spin((hog_blocks,), (block,), (cycles_hog,), stream=cp_lo)

    # Queue timed tiny op on high stream
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(hi_stream):
        start.record()
    spin((1,), (block,), (cycles_hi,), stream=cp_hi)
    with torch.cuda.stream(hi_stream):
        end.record()

    # Release both at once
    with torch.cuda.stream(ctrl):
        gate.record()

    end.synchronize()
    ms = start.elapsed_time(end)

    destroy_stream(hi_ptr)
    destroy_stream(lo_ptr)
    return ms


def main():
    torch.cuda.init()
    torch.cuda.set_device(0)

    least_pri, greatest_pri = get_priority_range()
    print(
        f"[info] priority range: least={least_pri}, greatest={greatest_pri} (lower number = higher priority)"
    )

    # Adjust these if needed
    hi_ms = 0.2  # tiny kernel duration target
    hog_ms = 50.0  # hog kernel target

    ms_alone = time_hi_only(hi_ms=hi_ms)
    ms_no_pri = time_with_hog(use_priority=False, hog_ms=hog_ms, hi_ms=hi_ms)
    ms_pri = time_with_hog(use_priority=True, hog_ms=hog_ms, hi_ms=hi_ms)

    print("\n=== High-priority tiny work timing ===")
    print(f"[alone]        {ms_alone:.3f} ms")
    print(f"[hog no-pri]   {ms_no_pri:.3f} ms")
    print(f"[hog + pri]    {ms_pri:.3f} ms")

    if ms_no_pri > 0:
        print(f"\n[benefit vs no-pri]  x{(ms_no_pri / ms_pri):.2f}")
    if ms_alone > 0:
        print(f"[slowdown no-pri vs alone] x{(ms_no_pri / ms_alone):.2f}")
        print(f"[slowdown pri vs alone]    x{(ms_pri / ms_alone):.2f}")


if __name__ == "__main__":
    main()
