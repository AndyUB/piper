#!/usr/bin/env python3
"""
Analyze Nsight Systems SQLite traces from Piper ZeRO runs.

Usage:
    python3 analyze_nsys.py <file1.sqlite> [file2.sqlite ...] [options]

Key options:
    --warmup N          warmup iterations to exclude (default: 2)
    --iters N           timed iterations to include (default: 5)
    --trace-iters N     tracing iterations to exclude at end (default: 3)
    --gap-ms N          inter-iteration gap threshold in ms (default: 500)
    --out-dir DIR       directory for output files (default: .)
    --topn N            top N kernels per file (default: 15)
"""

import argparse
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


# ─── NVTX range classification ───────────────────────────────────────────────

_RE_FWD        = re.compile(r"^forward_s(\d+)_b(\d+)_mb(\d+)$")
_RE_BWD        = re.compile(r"^backward_s(\d+)_b(\d+)_mb(\d+)$")
_RE_BWD_I      = re.compile(r"^backward_input_stage_(\d+)_b(\d+)_mb_(\d+)$")
_RE_BWD_W      = re.compile(r"^backward_weight_stage_(\d+)_mb_(\d+)$")
_RE_RS         = re.compile(r"^reduce_scatter_s(\d+)_b(\d+)$")
_RE_AG         = re.compile(r"^all_gather_s(\d+)_b(\d+)$")
_RE_AR         = re.compile(r"^all_reduce_s(\d+)_b(\d+)$")
_RE_FWD_A2A    = re.compile(r"^fwd_a2a_s(\d+)_b(\d+)_mb(\d+)$")
_RE_BWD_A2A    = re.compile(r"^bwd_a2a_s(\d+)_b(\d+)_mb(\d+)$")
_RE_UPDATE     = re.compile(r"^update$")

NCCL_RE = re.compile(r"nccl", re.IGNORECASE)


def parse_nvtx(text: str) -> dict | None:
    """Return structured info dict for a known NVTX range, else None."""
    m = _RE_FWD.match(text)
    if m:
        return dict(cat="fwd", stage=int(m[1]), bucket=int(m[2]), mb=int(m[3]))
    m = _RE_BWD.match(text)
    if m:
        return dict(cat="bwd", stage=int(m[1]), bucket=int(m[2]), mb=int(m[3]))
    m = _RE_BWD_I.match(text)
    if m:
        return dict(cat="bwd_i", stage=int(m[1]), bucket=int(m[2]), mb=int(m[3]))
    m = _RE_BWD_W.match(text)
    if m:
        return dict(cat="bwd_w", stage=int(m[1]), mb=int(m[2]))
    m = _RE_RS.match(text)
    if m:
        return dict(cat="rs", stage=int(m[1]), bucket=int(m[2]))
    m = _RE_AG.match(text)
    if m:
        return dict(cat="ag", stage=int(m[1]), bucket=int(m[2]))
    m = _RE_AR.match(text)
    if m:
        return dict(cat="ar", stage=int(m[1]), bucket=int(m[2]))
    m = _RE_FWD_A2A.match(text)
    if m:
        return dict(cat="fwd_a2a", stage=int(m[1]), bucket=int(m[2]), mb=int(m[3]))
    m = _RE_BWD_A2A.match(text)
    if m:
        return dict(cat="bwd_a2a", stage=int(m[1]), bucket=int(m[2]), mb=int(m[3]))
    if _RE_UPDATE.match(text):
        return dict(cat="update")
    return None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def ns2ms(ns): return ns / 1_000_000

def _stats(vals: list[float]) -> dict:
    if not vals:
        return dict(n=0, mean=0.0, std=0.0, total=0.0, min=0.0, max=0.0)
    n = len(vals)
    mu = sum(vals) / n
    sigma = (sum((v - mu)**2 for v in vals) / n) ** 0.5
    return dict(n=n, mean=mu, std=sigma, total=sum(vals), min=min(vals), max=max(vals))

def _overlap_frac(a_ivs, b_ivs) -> float:
    """Fraction of union(a_ivs) that is covered by b_ivs."""
    if not a_ivs or not b_ivs:
        return 0.0
    def union_len(ivs):
        merged = []
        for s, e in sorted(ivs):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append([s, e])
        return sum(e - s for s, e in merged)
    overlap = []
    for as_, ae in a_ivs:
        for bs, be in b_ivs:
            lo, hi = max(as_, bs), min(ae, be)
            if lo < hi:
                overlap.append((lo, hi))
    a_len = union_len(a_ivs)
    return 0.0 if a_len == 0 else union_len(overlap) / a_len


# ─── Iteration boundary detection ────────────────────────────────────────────

def find_timed_window(
    events: list[tuple],   # [(start_ns, end_ns, text), ...]
    n_warmup: int,
    n_iters: int,
    n_trace: int,
    gap_ms: float,
) -> tuple[int, int] | None:
    """
    Return (window_start_ns, window_end_ns) covering only the N timed iterations,
    excluding warmup and trace iterations.

    Strategy: use `update` NVTX events as iteration markers (one per iteration).
    If that count doesn't match, fall back to gap-based detection on all events.
    """
    gap_ns = int(gap_ms * 1_000_000)
    total_iters = n_warmup + n_iters + n_trace

    # Strategy 1: use `update` events as iteration markers
    update_evs = sorted(
        [(s, e) for s, e, t in events if t == "update"],
        key=lambda x: x[0],
    )
    if len(update_evs) == total_iters:
        # exactly one update per iteration — easy case
        timed_start = update_evs[n_warmup][0]
        timed_end   = update_evs[n_warmup + n_iters - 1][1]
        return (timed_start, timed_end)

    # Strategy 2: group update events by iteration using time gaps
    if update_evs:
        groups: list[list] = [[update_evs[0]]]
        for ev in update_evs[1:]:
            if ev[0] - groups[-1][-1][1] > gap_ns:
                groups.append([])
            groups[-1].append(ev)
        if len(groups) == total_iters:
            timed_start = groups[n_warmup][0][0]
            timed_end   = groups[n_warmup + n_iters - 1][-1][1]
            return (timed_start, timed_end)

    # Strategy 3: gap-based on all events
    sorted_ev = sorted(events, key=lambda x: x[0])
    boundaries = [0]
    for i in range(1, len(sorted_ev)):
        gap = sorted_ev[i][0] - sorted_ev[i-1][1]
        if gap > gap_ns:
            boundaries.append(i)
    # Only keep the regular inter-iteration gaps (filter init gaps by taking
    # the N most recent boundaries from the end, where N = total_iters - 1)
    if len(boundaries) >= total_iters:
        iter_bounds = boundaries[-(total_iters - 1):]
        timed_start_idx = iter_bounds[n_warmup - 1] if n_warmup > 0 else 0
        timed_end_idx   = iter_bounds[n_warmup + n_iters - 2] if n_warmup + n_iters - 2 < len(iter_bounds) else len(sorted_ev) - 1
        timed_start = sorted_ev[timed_start_idx][0]
        timed_end   = sorted_ev[min(timed_end_idx, len(sorted_ev)-1)][1] or sorted_ev[-1][0]
        return (timed_start, timed_end)

    # Fallback: use everything
    return None


def filter_events(events, window):
    if window is None:
        return events
    ws, we = window
    return [(s, e, t) for s, e, t in events if s >= ws and (e is None or e <= we)]


# ─── Per-file analysis ────────────────────────────────────────────────────────

def analyze_file(
    sqlite_path: Path,
    n_warmup: int,
    n_iters: int,
    n_trace: int,
    gap_ms: float,
    topn: int,
) -> dict:
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    # Load all NVTX events with text
    c.execute(
        "SELECT start, end, text FROM NVTX_EVENTS "
        "WHERE text IS NOT NULL AND end IS NOT NULL AND end > start"
    )
    all_events = c.fetchall()   # [(start_ns, end_ns, text), ...]

    # Detect timed window
    window = find_timed_window(all_events, n_warmup, n_iters, n_trace, gap_ms)
    timed_events = filter_events(all_events, window)

    # Count total update events and timed ones (for diagnostic)
    n_total_updates = sum(1 for _, _, t in all_events if t == "update")
    n_timed_updates = sum(1 for _, _, t in timed_events if t == "update")

    # Parse timed NVTX events into structured records
    # Key for compute ops: (cat, stage, bucket, mb) or (cat, stage, bucket) etc.
    # Duration accumulator: key → list of durations (ms)
    op_durations: dict[tuple, list[float]] = defaultdict(list)
    # Also store (start_ns, end_ns) for overlap analysis
    compute_ivs: list[tuple[int, int]] = []
    collective_ivs: dict[tuple, list[tuple[int, int]]] = defaultdict(list)  # (cat,s,b) → ivs

    for start_ns, end_ns, text in timed_events:
        info = parse_nvtx(text)
        if info is None:
            continue
        dur = ns2ms(end_ns - start_ns)
        cat = info["cat"]

        if cat == "fwd":
            key = ("fwd", info["stage"], info["bucket"], info["mb"])
            compute_ivs.append((start_ns, end_ns))
        elif cat == "bwd":
            key = ("bwd", info["stage"], info["bucket"], info["mb"])
            compute_ivs.append((start_ns, end_ns))
        elif cat == "bwd_i":
            key = ("bwd_i", info["stage"], info["bucket"], info["mb"])
            compute_ivs.append((start_ns, end_ns))
        elif cat == "bwd_w":
            key = ("bwd_w", info["stage"], info["mb"])
            compute_ivs.append((start_ns, end_ns))
        elif cat == "rs":
            key = ("rs", info["stage"], info["bucket"])
            collective_ivs[("rs", info["stage"], info["bucket"])].append((start_ns, end_ns))
        elif cat == "ag":
            key = ("ag", info["stage"], info["bucket"])
            collective_ivs[("ag", info["stage"], info["bucket"])].append((start_ns, end_ns))
        elif cat == "ar":
            key = ("ar", info["stage"], info["bucket"])
            collective_ivs[("ar", info["stage"], info["bucket"])].append((start_ns, end_ns))
        elif cat == "fwd_a2a":
            key = ("fwd_a2a", info["stage"], info["bucket"], info["mb"])
        elif cat == "bwd_a2a":
            key = ("bwd_a2a", info["stage"], info["bucket"], info["mb"])
        elif cat == "update":
            key = ("update",)
        else:
            continue
        op_durations[key].append(dur)

    # GPU kernels within timed window
    ws_ns = window[0] if window else 0
    we_ns = window[1] if window else 10**18
    c.execute(
        "SELECT k.start, k.end, s.value "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id "
        "WHERE k.end > k.start AND k.start >= ? AND k.end <= ?",
        (ws_ns, we_ns),
    )
    kernel_rows = c.fetchall()

    nccl_ivs: list[tuple[int, int]] = []
    gpu_compute_ivs: list[tuple[int, int]] = []
    kernel_time: dict[str, float] = defaultdict(float)
    kernel_calls: dict[str, int] = defaultdict(int)

    for kstart, kend, name in kernel_rows:
        dur = ns2ms(kend - kstart)
        kernel_time[name] += dur
        kernel_calls[name] += 1
        if NCCL_RE.search(name):
            nccl_ivs.append((kstart, kend))
        else:
            gpu_compute_ivs.append((kstart, kend))

    # NCCL kernel time per collective type (match by time overlap with NVTX)
    # Build per-collective-op GPU kernel time
    nccl_gpu_per_op: dict[tuple, float] = {}
    for op_key, ivs in collective_ivs.items():
        # sum GPU NCCL kernel time that overlaps with this op's NVTX ranges
        total = 0.0
        for ns, ne in nccl_ivs:
            for vs, ve in ivs:
                lo, hi = max(ns, vs), min(ne, ve)
                if lo < hi:
                    total += ns2ms(hi - lo)
                    break
        nccl_gpu_per_op[op_key] = total

    conn.close()

    return dict(
        window=window,
        n_total_updates=n_total_updates,
        n_timed_updates=n_timed_updates,
        op_durations=dict(op_durations),
        compute_ivs=compute_ivs,
        nccl_ivs=nccl_ivs,
        gpu_compute_ivs=gpu_compute_ivs,
        kernel_time=dict(kernel_time),
        kernel_calls=dict(kernel_calls),
        nccl_gpu_per_op=nccl_gpu_per_op,
    )


# ─── Report generation ───────────────────────────────────────────────────────

def _op_label(key: tuple) -> str:
    cat = key[0]
    if cat == "fwd":
        _, s, b, mb = key
        return f"FWD   s{s}_b{b}_mb{mb}"
    if cat == "bwd":
        _, s, b, mb = key
        return f"BWD   s{s}_b{b}_mb{mb}"
    if cat == "bwd_i":
        _, s, b, mb = key
        return f"BWD_I s{s}_b{b}_mb{mb}"
    if cat == "bwd_w":
        _, s, mb = key
        return f"BWD_W s{s}_mb{mb}"
    if cat == "rs":
        _, s, b = key
        return f"RS    s{s}_b{b}"
    if cat == "ag":
        _, s, b = key
        return f"AG    s{s}_b{b}"
    if cat == "ar":
        _, s, b = key
        return f"AR    s{s}_b{b}"
    if cat == "fwd_a2a":
        _, s, b, mb = key
        return f"FWD_A2A s{s}_b{b}_mb{mb}"
    if cat == "bwd_a2a":
        _, s, b, mb = key
        return f"BWD_A2A s{s}_b{b}_mb{mb}"
    if cat == "update":
        return "UPDATE"
    return str(key)


CAT_ORDER = ["fwd", "bwd", "bwd_i", "bwd_w", "rs", "ag", "ar", "fwd_a2a", "bwd_a2a", "update"]

def _sort_key(key):
    cat = key[0]
    idx = CAT_ORDER.index(cat) if cat in CAT_ORDER else 99
    return (idx,) + key[1:]


def generate_summary(
    filename: str,
    result: dict,
    n_warmup: int,
    n_iters: int,
    n_trace: int,
    topn: int,
) -> list[str]:
    lines = []
    w = result["window"]
    dur_s = (w[1] - w[0]) / 1e9 if w else 0
    lines.append(f"File: {filename}")
    lines.append(
        f"Warmup={n_warmup} excluded, Timed={n_iters} analyzed, Trace={n_trace} excluded"
    )
    if w:
        lines.append(
            f"Timed window: {w[0]/1e9:.3f}s – {w[1]/1e9:.3f}s  "
            f"(span={dur_s:.1f}s,  update events: {result['n_timed_updates']})"
        )
    else:
        lines.append("WARNING: Could not detect timed window — analyzing all events")
    lines.append("")

    op_dur = result["op_durations"]

    # ── Phase totals ──────────────────────────────────────────────────────────
    phase_totals: dict[str, float] = defaultdict(float)
    for key, durs in op_dur.items():
        phase_totals[key[0]] += sum(durs)
    total_ms = sum(phase_totals.values())

    if total_ms > 0:
        lines.append("  Phase Totals (timed window, % of all traced NVTX time)")
        lines.append(f"  {'Phase':<14} {'Total':>10}  {'% traced':>9}  {'per iter':>9}")
        lines.append(f"  {'-'*14} {'-'*10}  {'-'*9}  {'-'*9}")
        n_iter = max(result["n_timed_updates"], 1)
        for cat in CAT_ORDER:
            if cat not in phase_totals:
                continue
            ms = phase_totals[cat]
            lines.append(
                f"  {cat:<14} {ms:>9.1f}ms  {100*ms/total_ms:>8.1f}%  {ms/n_iter:>8.1f}ms"
            )
        lines.append("")

    # ── Per-stage/bucket summary (aggregated over microbatches) ───────────────
    # Group by (cat, stage, bucket) and pool all samples across mbs
    stage_bucket: dict[tuple, list[float]] = defaultdict(list)
    for key, durs in op_dur.items():
        cat = key[0]
        if cat in ("fwd", "bwd", "bwd_i", "bwd_w"):
            if cat in ("fwd", "bwd", "bwd_i"):
                agg_key = (cat, key[1], key[2])   # (cat, stage, bucket)
            else:  # bwd_w: no bucket
                agg_key = (cat, key[1], -1)
            stage_bucket[agg_key].extend(durs)
        elif cat in ("rs", "ag", "ar", "fwd_a2a", "bwd_a2a"):
            agg_key = (cat, key[1], key[2])
            stage_bucket[agg_key].extend(durs)
        elif cat == "update":
            stage_bucket[("update", -1, -1)].extend(durs)

    def _agg_label(k):
        cat, s, b = k
        if cat == "update":
            return "UPDATE"
        stage_str = f"s{s}"
        bucket_str = f"_b{b}" if b >= 0 else ""
        return f"{cat.upper():<7} {stage_str}{bucket_str}"

    lines.append(
        "  Per-Stage/Bucket Summary  (mean ± std aggregated over microbatches & iters)"
    )
    lines.append(
        "  Note: N = total calls (iters × microbatches).  Collectives have N = iters."
    )
    lines.append(
        f"  {'Operation':<18} {'N':>5} {'Mean/call':>10} {'Std':>8} {'Total':>10}"
    )
    lines.append(f"  {'-'*18} {'-'*5} {'-'*10} {'-'*8} {'-'*10}")

    def _sb_sort(k):
        cat, s, b = k
        idx = CAT_ORDER.index(cat) if cat in CAT_ORDER else 99
        return (idx, s, b)

    for k in sorted(stage_bucket.keys(), key=_sb_sort):
        s = _stats(stage_bucket[k])
        if s["n"] == 0:
            continue
        label = _agg_label(k)
        lines.append(
            f"  {label:<18} {s['n']:>5} {s['mean']:>9.2f}ms {s['std']:>7.2f}ms {s['total']:>9.1f}ms"
        )
    lines.append("")

    # ── Microbatch variance callout (only where mb-to-mb variation is notable) ─
    mb_variance_notes = []
    for key, durs in op_dur.items():
        cat = key[0]
        if cat not in ("fwd", "bwd"):
            continue
        s = _stats(durs)
        if s["n"] < 2 or s["mean"] < 0.5:
            continue
        cv = s["std"] / s["mean"]
        if cv > 0.40 and s["std"] > 3.0:
            label = _op_label(key).strip()
            mb_variance_notes.append(
                f"    {label:<30} mean={s['mean']:6.2f}ms std={s['std']:6.2f}ms  CV={cv:.0%}"
            )
    if mb_variance_notes:
        lines.append("  High-Variance Per-Microbatch Operations  (CV > 40%, std > 3ms)")
        lines += mb_variance_notes
        lines.append("")

    # ── Collective CPU dispatch vs GPU kernel ──────────────────────────────────
    coll_keys = [(k, v) for k, v in op_dur.items() if k[0] in ("rs", "ag", "ar")]
    if coll_keys:
        lines.append("  Collective: CPU Dispatch Time vs GPU Kernel Time")
        lines.append(
            f"  {'Op':<14} {'CPU total':>11} {'CPU mean':>10} {'GPU total':>11} {'Inflation':>10}"
        )
        lines.append(f"  {'-'*14} {'-'*11} {'-'*10} {'-'*11} {'-'*10}")
        for key, durs in sorted(coll_keys, key=lambda x: _sort_key(x[0])):
            cpu_total = sum(durs)
            cpu_mean  = cpu_total / len(durs) if durs else 0
            gpu_total = result["nccl_gpu_per_op"].get(key, 0.0)
            inflation = cpu_total / gpu_total if gpu_total > 0 else float("inf")
            label     = _op_label(key).strip()
            inf_str   = f"{inflation:.1f}x" if inflation < 1e6 else "∞"
            lines.append(
                f"  {label:<14} {cpu_total:>10.1f}ms {cpu_mean:>9.2f}ms "
                f"{gpu_total:>10.1f}ms {inf_str:>10}"
            )
        lines.append("")

    # ── Compute/NCCL overlap ──────────────────────────────────────────────────
    comp_cov      = _overlap_frac(result["compute_ivs"], result["nccl_ivs"])
    nccl_cov      = _overlap_frac(result["nccl_ivs"], result["compute_ivs"])
    total_comp_gpu = sum(ns2ms(e - s) for s, e in result["gpu_compute_ivs"])
    total_nccl_gpu = sum(ns2ms(e - s) for s, e in result["nccl_ivs"])
    total_gpu      = total_comp_gpu + total_nccl_gpu
    lines.append("  GPU Time & Overlap")
    if total_gpu > 0:
        lines.append(
            f"    Compute kernels : {total_comp_gpu:8.1f} ms  ({100*total_comp_gpu/total_gpu:.1f}%)"
        )
        lines.append(
            f"    NCCL kernels    : {total_nccl_gpu:8.1f} ms  ({100*total_nccl_gpu/total_gpu:.1f}%)"
        )
    lines.append(f"    Compute covered by NCCL (CPU dispatch): {100*comp_cov:.1f}%")
    lines.append(f"    NCCL   covered by compute (CPU dispatch): {100*nccl_cov:.1f}%")
    lines.append("")

    # ── Top GPU kernels ────────────────────────────────────────────────────────
    lines.append(f"  Top {topn} GPU Kernels (timed window only)")
    lines.append(f"  {'Kernel':<52} {'Calls':>6} {'Total':>10} {'Mean':>9}")
    lines.append(f"  {'-'*52} {'-'*6} {'-'*10} {'-'*9}")
    for name, total in sorted(result["kernel_time"].items(), key=lambda x: x[1], reverse=True)[:topn]:
        calls  = result["kernel_calls"][name]
        mean   = total / calls
        short  = name[:51] if len(name) > 51 else name
        lines.append(f"  {short:<52} {calls:>6} {total:>9.1f}ms {mean:>8.3f}ms")

    return lines


def generate_unexpected(filename: str, result: dict, n_iters: int) -> list[str]:
    issues = []

    op_dur = result["op_durations"]

    # 1. Collective CPU dispatch inflation (>10x GPU kernel = blocking)
    for key, durs in op_dur.items():
        if key[0] not in ("rs", "ag", "ar"):
            continue
        cpu_total = sum(durs)
        gpu_total = result["nccl_gpu_per_op"].get(key, 0.0)
        if gpu_total > 0:
            inflation = cpu_total / gpu_total
            if inflation > 10:
                issues.append(
                    f"[BLOCKING_COLLECTIVE] {_op_label(key).strip()}: "
                    f"CPU dispatch={cpu_total:.1f}ms vs GPU kernel={gpu_total:.1f}ms "
                    f"({inflation:.0f}x inflation). "
                    f"Collective is blocking the CPU thread — verify async_op=True "
                    f"and that no synchronization occurs inside the NVTX range."
                )

    # 2. Poor compute/NCCL overlap
    comp_cov = _overlap_frac(result["compute_ivs"], result["nccl_ivs"])
    if result["nccl_ivs"] and comp_cov < 0.15:
        issues.append(
            f"[POOR_OVERLAP] Compute covered by NCCL: {100*comp_cov:.1f}% "
            f"(expected >15% with bucketing). "
            f"Communication and computation are mostly sequential. "
            f"Check that RS is launched asynchronously during backward, "
            f"and AG is dispatched before the corresponding forward."
        )

    # 3. Dominant phase
    phase_totals: dict[str, float] = defaultdict(float)
    for key, durs in op_dur.items():
        phase_totals[key[0]] += sum(durs)
    total_ms = sum(phase_totals.values())
    if total_ms > 0:
        for cat, ms in sorted(phase_totals.items(), key=lambda x: x[1], reverse=True)[:1]:
            pct = 100 * ms / total_ms
            if pct > 50:
                issues.append(
                    f"[DOMINANT_PHASE] '{cat}' accounts for {pct:.1f}% of all traced NVTX time. "
                    f"This phase is the primary bottleneck."
                )

    # 4. High variance in forward/backward
    for key, durs in op_dur.items():
        if key[0] not in ("fwd", "bwd"):
            continue
        s = _stats(durs)
        if s["n"] < 2 or s["mean"] < 0.5:
            continue
        cv = s["std"] / s["mean"]  # coefficient of variation
        if cv > 0.5 and s["std"] > 5.0:  # >50% CV and >5ms std
            issues.append(
                f"[HIGH_VARIANCE] {_op_label(key).strip()}: "
                f"mean={s['mean']:.1f}ms std={s['std']:.1f}ms (CV={cv:.0%}). "
                f"High variability may indicate pipeline bubbles, load imbalance, "
                f"or contention with async communication."
            )

    # 5. Asymmetry: update >> backward
    upd_total = sum(sum(v) for k, v in op_dur.items() if k[0] == "update")
    bwd_total = sum(sum(v) for k, v in op_dur.items() if k[0] in ("bwd", "bwd_i", "bwd_w"))
    if bwd_total > 0 and upd_total > bwd_total * 2:
        issues.append(
            f"[SLOW_UPDATE] Optimizer update ({upd_total:.1f}ms) is "
            f"{upd_total/bwd_total:.1f}x larger than backward compute ({bwd_total:.1f}ms). "
            f"Consider fused optimizers or check if update is inadvertently blocking on collectives."
        )

    # 6. RS NVTX dispatch time much longer than forward (shouldn't dominate iteration)
    rs_total = sum(sum(v) for k, v in op_dur.items() if k[0] == "rs")
    fwd_total = sum(sum(v) for k, v in op_dur.items() if k[0] == "fwd")
    if rs_total > 0 and fwd_total > 0 and rs_total > fwd_total * 0.5:
        issues.append(
            f"[RS_DOMINATES_FWD] Reduce-scatter CPU dispatch ({rs_total:.1f}ms) is "
            f"{rs_total/fwd_total:.1%} of forward compute ({fwd_total:.1f}ms). "
            f"If RS is async, its NVTX range should be tiny (dispatch only). "
            f"A large RS NVTX time suggests the CPU waits for the collective to complete."
        )

    # 7. All-gather dispatch time much longer than actual AG GPU time
    ag_total_cpu = sum(sum(v) for k, v in op_dur.items() if k[0] == "ag")
    ag_total_gpu = sum(v for k, v in result["nccl_gpu_per_op"].items() if k[0] == "ag")
    if ag_total_gpu > 0 and ag_total_cpu > ag_total_gpu * 10:
        issues.append(
            f"[AG_BLOCKING] All-gather CPU dispatch ({ag_total_cpu:.1f}ms) is "
            f"{ag_total_cpu/ag_total_gpu:.0f}x actual GPU AG time ({ag_total_gpu:.1f}ms). "
            f"The AG NVTX range includes wait time for a preceding RS to complete, "
            f"meaning parameters are not pre-fetched before they are needed."
        )

    return issues


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Analyze Piper nsys SQLite traces (timed iters only)")
    ap.add_argument("sqlite_files", nargs="+", type=Path)
    ap.add_argument("--warmup",      type=int, default=2)
    ap.add_argument("--iters",       type=int, default=5)
    ap.add_argument("--trace-iters", type=int, default=3)
    ap.add_argument("--gap-ms",      type=float, default=500.0,
                    help="Min gap (ms) between consecutive NVTX events to mark an iteration boundary")
    ap.add_argument("--out-dir",     type=Path, default=Path("."))
    ap.add_argument("--topn",        type=int, default=15)
    ap.add_argument("--label",       default="",
                    help="Label string included in output filenames (e.g. 'zero1_3b')")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or "piper"
    summary_path   = args.out_dir / f"analysis_summary_{label}.txt"
    unexpected_path = args.out_dir / f"analysis_unexpected_{label}.txt"

    all_summary_lines: list[str] = [
        f"=== Piper NSys Analysis: {label} ===",
        f"Warmup={args.warmup}  Timed={args.iters}  TraceIters={args.trace_iters}",
        "",
    ]
    all_unexpected: list[str] = [
        f"=== Unexpected Findings: {label} ===",
        "",
    ]

    for path in args.sqlite_files:
        if not path.exists():
            print(f"[WARN] Not found: {path}", file=sys.stderr)
            continue
        print(f"Analyzing {path.name} ...", end=" ", flush=True)
        result = analyze_file(
            path, args.warmup, args.iters, args.trace_iters, args.gap_ms, args.topn,
        )
        print(f"updates={result['n_timed_updates']}/{result['n_total_updates']}", flush=True)

        sep = "=" * 70
        all_summary_lines += [sep, f"  Rank: {path.name}", sep]
        all_summary_lines += generate_summary(
            path.name, result, args.warmup, args.iters, args.trace_iters, args.topn
        )
        all_summary_lines.append("")

        issues = generate_unexpected(path.name, result, args.iters)
        if issues:
            all_unexpected.append(f"--- {path.name} ---")
            all_unexpected += issues
            all_unexpected.append("")

    # Write files
    summary_path.write_text("\n".join(all_summary_lines))
    unexpected_path.write_text("\n".join(all_unexpected))
    print(f"\nSummary     → {summary_path}")
    print(f"Unexpected  → {unexpected_path}")


if __name__ == "__main__":
    main()
