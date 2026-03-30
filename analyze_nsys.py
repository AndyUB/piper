#!/usr/bin/env python3
"""
Analyze Nsight Systems SQLite traces from Piper ZeRO runs.

Usage:
    python3 analyze_nsys.py <file1.sqlite> [file2.sqlite ...]

Produces per-file and aggregate reports covering:
  - NVTX range timing (forward / backward / collectives / update)
  - GPU kernel summary (top kernels by total GPU time)
  - NCCL vs compute overlap on the GPU
  - Bottleneck identification
"""

import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


# ─── NVTX range classification ───────────────────────────────────────────────

PATTERNS = [
    ("forward",         re.compile(r"^forward_s\d+_b\d+_mb\d+$")),
    ("backward",        re.compile(r"^backward_s\d+_b\d+_mb\d+$")),
    ("backward_input",  re.compile(r"^backward_input_stage_\d+_b\d+_mb_\d+$")),
    ("backward_weight", re.compile(r"^backward_weight_stage_\d+_mb_\d+$")),
    ("reduce_scatter",  re.compile(r"^reduce_scatter_s\d+_b\d+$")),
    ("all_gather",      re.compile(r"^all_gather_s\d+_b\d+$")),
    ("all_reduce",      re.compile(r"^all_reduce_s\d+_b\d+$")),
    ("fwd_p2p",         re.compile(r"^fwd_a2a_s\d+_b\d+_mb\d+$")),
    ("bwd_p2p",         re.compile(r"^bwd_a2a_s\d+_b\d+_mb\d+$")),
    ("update",          re.compile(r"^update$")),
]

NCCL_KERNEL_RE = re.compile(r"nccl", re.IGNORECASE)

COMPUTE_CATEGORY = {"forward", "backward", "backward_input", "backward_weight"}
COMM_CATEGORY    = {"reduce_scatter", "all_gather", "all_reduce", "fwd_p2p", "bwd_p2p"}


def classify(text: str) -> str | None:
    for label, pat in PATTERNS:
        if pat.match(text):
            return label
    return None


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def ns_to_ms(ns: int) -> float:
    return ns / 1e6


def stats(durations: list[float]) -> dict:
    if not durations:
        return {"n": 0, "total_ms": 0.0, "mean_ms": 0.0, "std_ms": 0.0}
    n = len(durations)
    total = sum(durations)
    mean = total / n
    variance = sum((d - mean) ** 2 for d in durations) / n
    std = variance ** 0.5
    return {"n": n, "total_ms": total, "mean_ms": mean, "std_ms": std}


def overlap_fraction(intervals_a: list[tuple], intervals_b: list[tuple]) -> float:
    """
    Given two lists of (start, end) intervals (in any unit), compute what fraction
    of the union of intervals_a is overlapped by intervals_b.
    Returns a float in [0, 1].
    """
    if not intervals_a or not intervals_b:
        return 0.0

    def union_length(ivs):
        merged = []
        for s, e in sorted(ivs):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append([s, e])
        return sum(e - s for s, e in merged)

    # Build combined intervals: intersection = a ∩ b
    overlap_ivs = []
    for as_, ae in intervals_a:
        for bs, be in intervals_b:
            lo = max(as_, bs)
            hi = min(ae, be)
            if lo < hi:
                overlap_ivs.append((lo, hi))

    a_len = union_length(intervals_a)
    if a_len == 0:
        return 0.0
    return union_length(overlap_ivs) / a_len


# ─── Per-file analysis ────────────────────────────────────────────────────────

def analyze_file(sqlite_path: Path) -> dict:
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    # ── 1. NVTX ranges ──
    c.execute(
        "SELECT start, end, text FROM NVTX_EVENTS "
        "WHERE text IS NOT NULL AND end IS NOT NULL AND end > start"
    )
    rows = c.fetchall()

    by_category: dict[str, list[float]] = defaultdict(list)
    raw_by_category: dict[str, list[tuple]] = defaultdict(list)

    for start_ns, end_ns, text in rows:
        cat = classify(text)
        if cat is None:
            continue
        dur_ms = ns_to_ms(end_ns - start_ns)
        by_category[cat].append(dur_ms)
        raw_by_category[cat].append((start_ns, end_ns))

    nvtx_stats = {cat: stats(durs) for cat, durs in by_category.items()}

    # ── 2. GPU kernels ──
    # shortName is a foreign key into StringIds
    c.execute(
        "SELECT k.start, k.end, s.value "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id "
        "WHERE k.end > k.start"
    )
    kernel_rows = c.fetchall()

    compute_ivs: list[tuple] = []
    nccl_ivs: list[tuple] = []
    kernel_time_by_name: dict[str, float] = defaultdict(float)
    kernel_calls_by_name: dict[str, int] = defaultdict(int)

    for start_ns, end_ns, name in kernel_rows:
        dur_ms = ns_to_ms(end_ns - start_ns)
        kernel_time_by_name[name] += dur_ms
        kernel_calls_by_name[name] += 1
        if NCCL_KERNEL_RE.search(name):
            nccl_ivs.append((start_ns, end_ns))
        else:
            compute_ivs.append((start_ns, end_ns))

    # ── 3. Overlap: compute covered by NCCL ──
    compute_covered_by_nccl = overlap_fraction(compute_ivs, nccl_ivs)
    nccl_covered_by_compute = overlap_fraction(nccl_ivs, compute_ivs)

    total_compute_gpu_ms = sum(ns_to_ms(e - s) for s, e in compute_ivs)
    total_nccl_gpu_ms = sum(ns_to_ms(e - s) for s, e in nccl_ivs)

    conn.close()

    return {
        "nvtx": nvtx_stats,
        "kernel_time": dict(kernel_time_by_name),
        "kernel_calls": dict(kernel_calls_by_name),
        "compute_gpu_ms": total_compute_gpu_ms,
        "nccl_gpu_ms": total_nccl_gpu_ms,
        "compute_covered_by_nccl": compute_covered_by_nccl,
        "nccl_covered_by_compute": nccl_covered_by_compute,
        "raw_nvtx": dict(raw_by_category),
    }


# ─── Formatting helpers ───────────────────────────────────────────────────────

def fmt_ms(ms: float) -> str:
    return f"{ms:8.2f} ms"


def print_nvtx_table(nvtx_stats: dict, title: str = "NVTX Range Summary"):
    order = [
        "forward", "backward", "backward_input", "backward_weight",
        "reduce_scatter", "all_gather", "all_reduce",
        "fwd_p2p", "bwd_p2p", "update",
    ]
    print(f"\n  {title}")
    print(f"  {'Category':<22} {'Count':>6} {'Total':>12} {'Mean':>12} {'Std':>12}")
    print(f"  {'-'*22} {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
    for cat in order:
        if cat not in nvtx_stats:
            continue
        s = nvtx_stats[cat]
        if s["n"] == 0:
            continue
        print(
            f"  {cat:<22} {s['n']:>6} "
            f"{fmt_ms(s['total_ms'])} "
            f"{fmt_ms(s['mean_ms'])} "
            f"{fmt_ms(s['std_ms'])}"
        )


def print_kernel_table(kernel_time: dict, kernel_calls: dict, topn: int = 15):
    print(f"\n  Top {topn} GPU Kernels by Total Time")
    print(f"  {'Kernel':<55} {'Calls':>6} {'Total':>12} {'Mean':>10}")
    print(f"  {'-'*55} {'-'*6} {'-'*12} {'-'*10}")
    sorted_kernels = sorted(kernel_time.items(), key=lambda x: x[1], reverse=True)[:topn]
    for name, total_ms in sorted_kernels:
        calls = kernel_calls[name]
        mean_ms = total_ms / calls
        short = name[:54] if len(name) > 54 else name
        print(f"  {short:<55} {calls:>6} {fmt_ms(total_ms)} {mean_ms:>8.3f} ms")


def print_overlap_report(result: dict):
    c_ms = result["compute_gpu_ms"]
    n_ms = result["nccl_gpu_ms"]
    total = c_ms + n_ms
    print(f"\n  GPU Time Breakdown")
    print(f"    Compute kernels : {fmt_ms(c_ms)}  ({100*c_ms/total:.1f}% of compute+comm)" if total else "")
    print(f"    NCCL kernels    : {fmt_ms(n_ms)}  ({100*n_ms/total:.1f}% of compute+comm)" if total else "")
    print(f"\n  Overlap Quality")
    print(f"    Fraction of compute overlapped by NCCL : {100*result['compute_covered_by_nccl']:5.1f}%")
    print(f"    Fraction of NCCL   overlapped by compute: {100*result['nccl_covered_by_compute']:5.1f}%")
    if result["compute_covered_by_nccl"] < 0.20:
        print("    ⚠  Low compute/NCCL overlap — communication is mostly sequential with compute.")
    elif result["compute_covered_by_nccl"] > 0.60:
        print("    ✓  Good overlap between compute and NCCL communication.")


def print_bottleneck_report(result: dict):
    nvtx = result["nvtx"]
    print(f"\n  Bottleneck Summary")

    # Dominant compute phase
    fwd_ms = nvtx.get("forward", {}).get("total_ms", 0)
    bwd_ms = (
        nvtx.get("backward", {}).get("total_ms", 0)
        + nvtx.get("backward_input", {}).get("total_ms", 0)
        + nvtx.get("backward_weight", {}).get("total_ms", 0)
    )
    rs_ms = nvtx.get("reduce_scatter", {}).get("total_ms", 0)
    ag_ms = nvtx.get("all_gather", {}).get("total_ms", 0)
    ar_ms = nvtx.get("all_reduce", {}).get("total_ms", 0)
    upd_ms = nvtx.get("update", {}).get("total_ms", 0)
    p2p_ms = (
        nvtx.get("fwd_p2p", {}).get("total_ms", 0)
        + nvtx.get("bwd_p2p", {}).get("total_ms", 0)
    )

    phases = [
        ("Forward compute", fwd_ms),
        ("Backward compute", bwd_ms),
        ("Reduce-scatter", rs_ms),
        ("All-gather", ag_ms),
        ("All-reduce", ar_ms),
        ("Optimizer update", upd_ms),
        ("P2P (send/recv)", p2p_ms),
    ]
    phases = [(n, m) for n, m in phases if m > 0]
    total = sum(m for _, m in phases)
    if total == 0:
        print("    No NVTX data found.")
        return

    phases.sort(key=lambda x: x[1], reverse=True)
    for name, ms in phases:
        print(f"    {name:<22} {fmt_ms(ms)}  ({100*ms/total:5.1f}%)")

    top_name, top_ms = phases[0]
    print(f"\n    Dominant bottleneck: {top_name} ({100*top_ms/total:.1f}% of traced time)")

    comm_total = rs_ms + ag_ms + ar_ms
    if comm_total > 0:
        print(f"    Communication fraction (CPU dispatch): {100*comm_total/total:.1f}%")
        if rs_ms > 0 and bwd_ms > 0:
            ratio = rs_ms / bwd_ms
            if ratio > 0.5:
                print(
                    f"    ⚠  Reduce-scatter CPU dispatch time is {ratio:.1f}x backward "
                    f"— check that RS is launched asynchronously."
                )
        if ag_ms > fwd_ms * 0.3 and ag_ms > 0:
            print(
                f"    ⚠  All-gather dispatch ({ag_ms:.1f} ms) is large relative to forward "
                f"({fwd_ms:.1f} ms). Ensure AG is pipelined with forward compute."
            )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Analyze Piper nsys traces (SQLite format)")
    ap.add_argument("sqlite_files", nargs="+", type=Path, help=".sqlite files exported from nsys")
    ap.add_argument("--topn", type=int, default=15, help="Top N kernels to display (default: 15)")
    args = ap.parse_args()

    all_results = []

    for path in args.sqlite_files:
        if not path.exists():
            print(f"[WARN] File not found: {path}", file=sys.stderr)
            continue

        print(f"\n{'='*72}")
        print(f"  File: {path.name}")
        print(f"{'='*72}")

        result = analyze_file(path)
        all_results.append((path.name, result))

        print_nvtx_table(result["nvtx"])
        print_kernel_table(result["kernel_time"], result["kernel_calls"], topn=args.topn)
        print_overlap_report(result)
        print_bottleneck_report(result)

    # ── Aggregate across ranks ──
    if len(all_results) > 1:
        print(f"\n{'='*72}")
        print(f"  AGGREGATE  ({len(all_results)} ranks)")
        print(f"{'='*72}")

        agg_nvtx: dict[str, list[float]] = defaultdict(list)
        agg_kernel_time: dict[str, float] = defaultdict(float)
        agg_kernel_calls: dict[str, int] = defaultdict(int)
        total_compute = 0.0
        total_nccl = 0.0
        avg_comp_cov = 0.0
        avg_nccl_cov = 0.0

        for _, r in all_results:
            for cat, s in r["nvtx"].items():
                agg_nvtx[cat].extend([s["mean_ms"]] if s["n"] > 0 else [])
            for k, v in r["kernel_time"].items():
                agg_kernel_time[k] += v
            for k, v in r["kernel_calls"].items():
                agg_kernel_calls[k] += v
            total_compute += r["compute_gpu_ms"]
            total_nccl += r["nccl_gpu_ms"]
            avg_comp_cov += r["compute_covered_by_nccl"]
            avg_nccl_cov += r["nccl_covered_by_compute"]

        n = len(all_results)
        avg_comp_cov /= n
        avg_nccl_cov /= n

        # Build aggregate nvtx from per-rank means
        agg_nvtx_stats = {}
        for cat, means in agg_nvtx.items():
            agg_nvtx_stats[cat] = {
                "n": len(means),
                "total_ms": sum(means),
                "mean_ms": sum(means) / len(means) if means else 0,
                "std_ms": (sum((m - sum(means)/len(means))**2 for m in means)/len(means))**0.5 if len(means) > 1 else 0,
            }

        print_nvtx_table(agg_nvtx_stats, title="NVTX Summary (per-rank means, averaged across ranks)")
        print_kernel_table(agg_kernel_time, agg_kernel_calls, topn=args.topn)

        t_total = total_compute + total_nccl
        if t_total > 0:
            print(f"\n  Aggregate GPU Time (all ranks summed)")
            print(f"    Compute : {fmt_ms(total_compute)}  ({100*total_compute/t_total:.1f}%)")
            print(f"    NCCL    : {fmt_ms(total_nccl)}  ({100*total_nccl/t_total:.1f}%)")

        print(f"\n  Average Overlap (across ranks)")
        print(f"    Compute covered by NCCL  : {100*avg_comp_cov:5.1f}%")
        print(f"    NCCL   covered by compute: {100*avg_nccl_cov:5.1f}%")


if __name__ == "__main__":
    main()
