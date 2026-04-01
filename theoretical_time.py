"""
theoretical_time.py — Compute the theoretical minimum iteration time for a
Schedule2D assuming zero communication overhead.

Usage (CLI):
    python theoretical_time.py --schedule interleaved_pp2_mb4 --t-single 1.057

    --schedule   : name of a predefined schedule (see SCHEDULES dict below), or
                   'custom' to supply --pp and --mbs for a dynamic 1F1B schedule
    --t-single   : single-GPU full-model iteration time in seconds (fwd+bwd only,
                   no UPD).  Used to derive per-stage times as
                       t_stage = t_single / num_baseline_stages
                   with fwd_ratio controlling the fwd/bwd split.
    --t-fwd      : per-stage forward time (overrides --t-single derivation)
    --t-bwd      : per-stage backward time (overrides --t-single derivation)
    --t-upd      : optimizer step time per rank (default 0)
    --fwd-ratio  : fraction of t_stage that is forward (default 1/3)
    --baseline-stages : number of stages in the single-GPU baseline (default 4)
    --pp         : pipeline degree for dynamic 1F1B schedule (with --schedule 1f1b)
    --mbs        : number of microbatches (with --schedule 1f1b)

Example (reproduce the numbers in baseline_comparison.txt):
    python theoretical_time.py --schedule interleaved_pp2_mb4 --t-single 1.057
    python theoretical_time.py --schedule no_pp_4s --t-single 1.057

Algorithm:
    1. Build a dependency DAG from the Schedule2D grid.
       - Within-rank sequential edges: each non-None task depends on the previous
         non-None task on the same rank.
       - Cross-rank data edges for FWD: FWD(stage s, mb m) → FWD(stage s+1, mb m)
         (successor stage needs activations from predecessor).
       - Cross-rank data edges for BWD: BWD(stage s+1, mb m) → BWD(stage s, mb m)
         (predecessor stage needs gradients from successor).
       - Communication tasks (SEND, RECV, ALL_REDUCE, REDUCE_SCATTER, ALL_GATHER)
         are assigned zero duration.
    2. Topological sort + ASAP scheduling → critical-path makespan.
    3. Report makespan, per-rank compute time, bubble fraction.
"""

import argparse
import sys
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# Imports from piper_exec — guarded so the file is importable even when the
# src package is not on sys.path (fall back to a minimal stub for unit tests).
# ---------------------------------------------------------------------------
try:
    from src.piper_exec import TaskType, Schedule2D, Task, BatchMeta
    from test.schedule_helpers import (
        NO_PP_SCHEDULE,
        NO_PP_4STAGE_SCHEDULE,
        INTERLEAVED_1F1B_PP2_MB4_SCHEDULE,
        INTERLEAVED_1F1B_PP2_MB6_SCHEDULE,
        build_1f1b_schedule,
    )
    _PIPER_AVAILABLE = True
except ImportError:
    _PIPER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Predefined schedules exposed by the CLI
# ---------------------------------------------------------------------------
def _get_schedules():
    if not _PIPER_AVAILABLE:
        raise RuntimeError("piper src/test packages not importable")
    return {
        "no_pp":            NO_PP_SCHEDULE,
        "no_pp_4s":         NO_PP_4STAGE_SCHEDULE,
        "interleaved_pp2_mb4": INTERLEAVED_1F1B_PP2_MB4_SCHEDULE,
        "interleaved_pp2_mb6": INTERLEAVED_1F1B_PP2_MB6_SCHEDULE,
    }


# ---------------------------------------------------------------------------
# Duration assignment
# ---------------------------------------------------------------------------
_COMM_TYPES = {
    TaskType.SEND, TaskType.RECV,
    TaskType.ALL_REDUCE, TaskType.REDUCE_SCATTER, TaskType.ALL_GATHER,
    TaskType.FWD_A2A, TaskType.BWD_A2A,
}

def task_duration(task: "Task", t_fwd: float, t_bwd: float, t_upd: float,
                  fwd_i_ratio: float = 0.5) -> float:
    """Return the compute duration of a single task in seconds.

    BWD_I / BWD_W split t_bwd according to fwd_i_ratio (fraction that is BWD_I).
    Communication tasks get 0 (the whole point of this theoretical analysis).
    """
    tt = task.type
    if tt in _COMM_TYPES:
        return 0.0
    if tt == TaskType.FWD:
        return t_fwd
    if tt == TaskType.BWD:
        return t_bwd
    if tt == TaskType.BWD_I:
        return t_bwd * fwd_i_ratio
    if tt == TaskType.BWD_W:
        return t_bwd * (1.0 - fwd_i_ratio)
    if tt == TaskType.FWD_BWD:
        return t_fwd + t_bwd
    if tt == TaskType.UPD:
        return t_upd
    return 0.0


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def theoretical_iter_time(
    schedule: "Schedule2D",
    t_fwd: float,
    t_bwd: float,
    t_upd: float = 0.0,
) -> dict:
    """
    Compute theoretical minimum iteration time via ASAP critical-path analysis.

    Parameters
    ----------
    schedule : Schedule2D
    t_fwd    : per-stage forward pass time (seconds)
    t_bwd    : per-stage backward pass time (seconds)
    t_upd    : optimizer update time per rank (seconds, default 0)

    Returns
    -------
    dict with keys:
        makespan        : float — critical-path length (seconds)
        throughput_tps  : float — tokens/s estimate (if you know batch size)
        rank_compute    : list[float] — total compute (excl. bubbles) per rank
        rank_bubble     : list[float] — bubble time per rank
        bubble_fraction : list[float] — bubble / makespan per rank
        finish_times    : dict[(pp_rank, time_step), float] — ASAP finish time
                          for every non-None task node
    """
    grid = schedule.grid
    n_ranks = len(grid)

    # ------------------------------------------------------------------
    # Step 1: enumerate all (rank, t_step) nodes and build dependency edges
    # ------------------------------------------------------------------
    # node key: (rank, time_step)
    nodes = []          # list of (rank, t_step, task)
    node_set = set()
    for r, row in enumerate(grid):
        for t, task in enumerate(row):
            if task is not None:
                nodes.append((r, t, task))
                node_set.add((r, t))

    # predecessors: node -> list of predecessor nodes
    preds: dict[tuple, list[tuple]] = defaultdict(list)
    succs: dict[tuple, list[tuple]] = defaultdict(list)

    def add_edge(src, dst):
        if src in node_set and dst in node_set:
            succs[src].append(dst)
            preds[dst].append(src)

    # Within-rank sequential edges
    for r, row in enumerate(grid):
        prev = None
        for t, task in enumerate(row):
            if task is not None:
                if prev is not None:
                    add_edge(prev, (r, t))
                prev = (r, t)

    # Cross-rank data dependency edges via stage adjacency
    # Build: (stage_id, mb_idx, task_type_category) -> (rank, time_step)
    # category: "fwd" or "bwd"
    stage_task: dict[tuple, tuple] = {}  # (stage_id, mb_idx, cat) -> (rank, t_step)

    for r, row in enumerate(grid):
        for t, task in enumerate(row):
            if task is None:
                continue
            tt = task.type
            if tt in (TaskType.FWD, TaskType.FWD_BWD):
                for bm in task.batches:
                    if tt == TaskType.FWD_BWD:
                        # For FWD_BWD, the batches list has [bwd_batch, fwd_batch]
                        # convention varies; treat all as both fwd and bwd for safety
                        stage_task[(bm.stage_id, bm.mb_idx, "fwd")] = (r, t)
                        stage_task[(bm.stage_id, bm.mb_idx, "bwd")] = (r, t)
                    else:
                        stage_task[(bm.stage_id, bm.mb_idx, "fwd")] = (r, t)
            if tt in (TaskType.BWD, TaskType.BWD_I, TaskType.FWD_BWD):
                for bm in task.batches:
                    stage_task[(bm.stage_id, bm.mb_idx, "bwd")] = (r, t)

    stage_to_device = schedule.stage_to_device()
    all_stages = sorted(stage_to_device.keys())

    for i, s in enumerate(all_stages):
        if i + 1 >= len(all_stages):
            break
        s_next = all_stages[i + 1]
        # Only add cross-rank edges when stages are on different ranks
        if stage_to_device[s] == stage_to_device[s_next]:
            continue
        num_mbs = schedule.num_mbs()
        for mb in range(num_mbs):
            # FWD: stage s → stage s+1
            src_fwd = stage_task.get((s, mb, "fwd"))
            dst_fwd = stage_task.get((s_next, mb, "fwd"))
            if src_fwd and dst_fwd:
                add_edge(src_fwd, dst_fwd)
            # BWD: stage s+1 → stage s
            src_bwd = stage_task.get((s_next, mb, "bwd"))
            dst_bwd = stage_task.get((s, mb, "bwd"))
            if src_bwd and dst_bwd:
                add_edge(src_bwd, dst_bwd)

    # ------------------------------------------------------------------
    # Step 2: ASAP scheduling via topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------
    in_degree = defaultdict(int)
    for node in node_set:
        in_degree[node]  # ensure key exists
    for node in node_set:
        for succ in succs[node]:
            in_degree[succ] += 1

    # Duration map
    dur: dict[tuple, float] = {}
    for r, row in enumerate(grid):
        for t, task in enumerate(row):
            if task is not None:
                dur[(r, t)] = task_duration(task, t_fwd, t_bwd, t_upd)

    finish: dict[tuple, float] = {}
    queue = deque()
    for node in node_set:
        if in_degree[node] == 0:
            queue.append(node)

    while queue:
        node = queue.popleft()
        pred_finish = max((finish[p] for p in preds[node]), default=0.0)
        finish[node] = pred_finish + dur[node]
        for succ in succs[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(finish) != len(node_set):
        raise ValueError("Cycle detected in schedule DAG — check the schedule definition.")

    makespan = max(finish.values())

    # ------------------------------------------------------------------
    # Step 3: Per-rank statistics
    # ------------------------------------------------------------------
    rank_compute = [0.0] * n_ranks
    for r, row in enumerate(grid):
        for t, task in enumerate(row):
            if task is not None:
                rank_compute[r] += dur[(r, t)]

    rank_bubble = [makespan - rc for rc in rank_compute]
    bubble_fraction = [rb / makespan if makespan > 0 else 0.0 for rb in rank_bubble]

    return dict(
        makespan=makespan,
        rank_compute=rank_compute,
        rank_bubble=rank_bubble,
        bubble_fraction=bubble_fraction,
        finish_times=dict(finish),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _derive_times(args):
    """Derive t_fwd / t_bwd from CLI args."""
    if args.t_fwd is not None and args.t_bwd is not None:
        return args.t_fwd, args.t_bwd
    if args.t_single is None:
        raise ValueError("Provide either --t-single or both --t-fwd and --t-bwd")
    t_single = args.t_single
    n_stages = args.baseline_stages
    fwd_ratio = args.fwd_ratio
    t_stage = t_single / n_stages
    t_fwd = t_stage * fwd_ratio
    t_bwd = t_stage * (1.0 - fwd_ratio)
    return t_fwd, t_bwd


def main():
    parser = argparse.ArgumentParser(
        description="Theoretical minimum iteration time for a Piper Schedule2D."
    )
    parser.add_argument("--schedule", default="interleaved_pp2_mb4",
                        help="Schedule name or '1f1b' for dynamic build_1f1b_schedule")
    parser.add_argument("--pp", type=int, default=2,
                        help="Pipeline degree (for --schedule 1f1b)")
    parser.add_argument("--mbs", type=int, default=4,
                        help="Number of microbatches (for --schedule 1f1b)")
    parser.add_argument("--t-single", type=float, default=None,
                        help="Single-GPU iteration time in seconds (e.g. 1.057)")
    parser.add_argument("--t-fwd", type=float, default=None,
                        help="Per-stage forward time in seconds")
    parser.add_argument("--t-bwd", type=float, default=None,
                        help="Per-stage backward time in seconds")
    parser.add_argument("--t-upd", type=float, default=0.0,
                        help="Optimizer update time per rank in seconds (default 0)")
    parser.add_argument("--fwd-ratio", type=float, default=1/3,
                        help="Fraction of t_stage that is forward (default 1/3)")
    parser.add_argument("--baseline-stages", type=int, default=4,
                        help="Number of stages in the single-GPU baseline (default 4)")
    args = parser.parse_args()

    schedules = _get_schedules()
    if args.schedule == "1f1b":
        schedule = build_1f1b_schedule(args.mbs, args.pp)
        sched_name = f"1f1b_pp{args.pp}_mb{args.mbs}"
    elif args.schedule in schedules:
        schedule = schedules[args.schedule]
        sched_name = args.schedule
    else:
        print(f"Unknown schedule '{args.schedule}'. Available: {list(schedules)} + '1f1b'",
              file=sys.stderr)
        sys.exit(1)

    t_fwd, t_bwd = _derive_times(args)
    t_upd = args.t_upd

    print(f"Schedule : {sched_name}")
    print(f"  ranks  : {schedule.num_ranks()}, stages: {schedule.num_stages()}, mbs: {schedule.num_mbs()}")
    print(f"  t_fwd  : {t_fwd*1000:.2f} ms/stage,  t_bwd: {t_bwd*1000:.2f} ms/stage,  t_upd: {t_upd*1000:.2f} ms")
    print()

    result = theoretical_iter_time(schedule, t_fwd, t_bwd, t_upd)

    print(f"Theoretical makespan  : {result['makespan']*1000:.1f} ms  ({result['makespan']:.4f} s)")
    print()
    print(f"{'Rank':<6} {'Compute (ms)':<16} {'Bubble (ms)':<14} {'Bubble %'}")
    for r in range(schedule.num_ranks()):
        comp = result['rank_compute'][r] * 1000
        bub  = result['rank_bubble'][r] * 1000
        frac = result['bubble_fraction'][r] * 100
        print(f"  {r:<4} {comp:<16.1f} {bub:<14.1f} {frac:.1f}%")


if __name__ == "__main__":
    main()
