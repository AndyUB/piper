#!/usr/bin/env python3
"""Parse Piper memdebug logs and plot per-task allocated/reserved memory.

Each task contributes two points per series:
1) peak memory observed during task execution
2) memory immediately after task execution
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

HEADER_RE = re.compile(r"=== Rank\s+(?P<rank>\d+) per-task memory debug ===")
ITER_START_RE = re.compile(
    r"iter start:\s+alloc=(?P<alloc>[0-9.]+) GiB\s+reserved=(?P<reserved>[0-9.]+) GiB"
)
TASK_RE = re.compile(
    r"\s*(?P<task>t-?\d+\s+[^:]+):\s+"
    r"peak_alloc=(?P<peak_alloc>[0-9.]+) GiB\s+"
    r"peak_reserved=(?P<peak_reserved>[0-9.]+) GiB\s+"
    r"alloc_after=(?P<alloc_after>[0-9.]+) GiB\s+"
    r"reserved_after=(?P<reserved_after>[0-9.]+) GiB"
)


@dataclass
class TaskMem:
    task: str
    peak_alloc: float
    peak_reserved: float
    alloc_after: float
    reserved_after: float


def parse_rank_entries(
    log_path: Path, rank: int
) -> tuple[tuple[float, float] | None, list[TaskMem]]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_section = False
    iter_start: tuple[float, float] | None = None
    entries: list[TaskMem] = []

    for line in lines:
        header_match = HEADER_RE.search(line)
        if header_match:
            in_section = int(header_match.group("rank")) == rank
            continue

        if not in_section:
            continue

        # Stop at the next rank section.
        if "=== Rank" in line and "per-task memory debug" in line:
            break

        if iter_start is None:
            iter_match = ITER_START_RE.search(line)
            if iter_match:
                iter_start = (
                    float(iter_match.group("alloc")),
                    float(iter_match.group("reserved")),
                )
                continue

        task_match = TASK_RE.search(line)
        if task_match:
            entries.append(
                TaskMem(
                    task=task_match.group("task"),
                    peak_alloc=float(task_match.group("peak_alloc")),
                    peak_reserved=float(task_match.group("peak_reserved")),
                    alloc_after=float(task_match.group("alloc_after")),
                    reserved_after=float(task_match.group("reserved_after")),
                )
            )

    return iter_start, entries


def build_series(
    entries: list[TaskMem],
) -> tuple[list[int], list[float], list[float], list[str]]:
    x: list[int] = []
    alloc_y: list[float] = []
    reserved_y: list[float] = []
    labels: list[str] = []

    for i, e in enumerate(entries):
        x_peak = 2 * i
        x_after = 2 * i + 1

        x.extend([x_peak, x_after])
        alloc_y.extend([e.peak_alloc, e.alloc_after])
        reserved_y.extend([e.peak_reserved, e.reserved_after])
        labels.extend([f"{e.task} [peak]", f"{e.task} [after]"])

    return x, alloc_y, reserved_y, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to memdebug log file")
    parser.add_argument("--rank", type=int, default=0, help="Rank to plot (default: 0)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <log_stem>-rank<rank>-memdebug.png)",
    )
    parser.add_argument(
        "--label-every",
        type=int,
        default=8,
        help="Show x tick labels every N tasks (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse and print summary; skip plotting (useful if matplotlib is unavailable).",
    )
    args = parser.parse_args()

    iter_start, entries = parse_rank_entries(args.log, args.rank)
    if not entries:
        raise SystemExit(
            f"No per-task memory entries found for rank {args.rank} in {args.log}."
        )

    x, alloc_y, reserved_y, labels = build_series(entries)
    del labels  # Keep for potential future extensions (e.g., interactive hover labels).

    if args.dry_run:
        print(f"Parsed {len(entries)} task entries for rank {args.rank}.")
        if entries:
            print(f"First task: {entries[0].task}")
            print(f"Last task:  {entries[-1].task}")
        return

    output = args.output
    if output is None:
        output = args.log.with_name(f"{args.log.stem}-rank{args.rank}-memdebug.png")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install it (e.g. `pip install matplotlib`) "
            "or run with --dry-run to just validate parsing."
        ) from exc

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(x, alloc_y, marker="o", ms=2.5, lw=1.3, label="allocated (peak/after)")
    ax.plot(x, reserved_y, marker="o", ms=2.5, lw=1.3, label="reserved (peak/after)")

    if iter_start is not None:
        ax.axhline(
            iter_start[0],
            color="C0",
            alpha=0.25,
            ls="--",
            lw=1,
            label="iter-start alloc",
        )
        ax.axhline(
            iter_start[1],
            color="C1",
            alpha=0.25,
            ls="--",
            lw=1,
            label="iter-start reserved",
        )

    task_tick_x: list[int] = []
    task_tick_labels: list[str] = []
    step = max(1, args.label_every)
    for task_i in range(0, len(entries), step):
        peak_idx = 2 * task_i
        task_tick_x.append(peak_idx)
        task_tick_labels.append(entries[task_i].task)

    ax.set_xticks(task_tick_x)
    ax.set_xticklabels(task_tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Task progression (2 points per task: peak, then after)")
    ax.set_ylabel("Memory (GiB)")
    ax.set_title(f"Per-task GPU memory for rank {args.rank}: {args.log.name}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)

    # Add a light vertical guide every task boundary (every 2 points).
    for i in range(0, len(x), 2):
        ax.axvline(i, color="gray", alpha=0.06, lw=0.8)

    fig.tight_layout()
    fig.savefig(output, dpi=160)

    print(f"Parsed {len(entries)} task entries for rank {args.rank}.")
    print(f"Wrote plot to: {output}")


if __name__ == "__main__":
    main()
