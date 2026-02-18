import numpy as np
import torch
import cupy as cp
import cuda.bindings.runtime as cudart


def check(err, where):
    if err != 0:
        raise RuntimeError(f"CUDA error {err} at {where}")


def dev_attr(attr_name: str, device: int = 0) -> int:
    # cuda-bindings exposes attributes either as cudart.cudaDevAttrX or cudart.cudaDeviceAttr.cudaDevAttrX
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


def run_trial(use_priority: bool, hog_ms: float = 50.0, hi_ms: float = 0.2):
    torch.cuda.synchronize()

    # priority range: (least, greatest), where lower number => higher priority
    err, least_pri, greatest_pri = cudart.cudaDeviceGetStreamPriorityRange()
    check(err, "cudaDeviceGetStreamPriorityRange")
    print(
        f"[info] priority range: least={least_pri}, greatest={greatest_pri} (lower number = higher priority)"
    )

    if use_priority:
        hi_pri = greatest_pri
        lo_pri = least_pri
    else:
        hi_pri = least_pri
        lo_pri = least_pri

    hi_ptr, hi_stream = make_stream(hi_pri)
    lo_ptr, lo_stream = make_stream(lo_pri)

    # Wrap torch streams for CuPy kernel launches
    cp_hi = cp.cuda.ExternalStream(hi_stream.cuda_stream)
    cp_lo = cp.cuda.ExternalStream(lo_stream.cuda_stream)

    # Calibrate cycles from SM clock (kHz): cycles_per_ms ~= clock_khz
    clock_khz = dev_attr("cudaDevAttrClockRate")
    sm_count = dev_attr("cudaDevAttrMultiProcessorCount")
    cycles_hog = np.uint64(clock_khz * hog_ms)
    cycles_hi = np.uint64(clock_khz * hi_ms)

    # Make hog grid large enough to keep device busy "for real"
    # (lots of blocks so if it starts first, it can fully occupy the GPU)
    hog_blocks = int(sm_count * 4096)
    block = 256

    spin = cp.RawKernel(SPIN_SRC, "spin")

    gate = torch.cuda.Event(enable_timing=False)
    ctrl = torch.cuda.Stream()  # regular stream, used to "release" the gate

    # Both streams wait on gate so they become runnable together
    lo_stream.wait_event(gate)
    hi_stream.wait_event(gate)

    # Queue hog on low stream
    with torch.cuda.stream(lo_stream):
        spin((hog_blocks,), (block,), (cycles_hog,), stream=cp_lo)

    # Time a tiny op on high stream
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(hi_stream):
        start.record()
        spin((1,), (block,), (cycles_hi,), stream=cp_hi)
        end.record()

    # Release both at once (after both have queued work)
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

    ms_no = run_trial(use_priority=False)
    ms_pr = run_trial(use_priority=True)

    print("\n=== Priority sanity (single GPU) ===")
    print(f"[no priority] {ms_no:.3f} ms")
    print(f"[priority]    {ms_pr:.3f} ms")
    print(f"[speedup]     x{(ms_no / ms_pr):.2f}")


if __name__ == "__main__":
    main()
