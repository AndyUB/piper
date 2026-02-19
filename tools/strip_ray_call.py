#!/usr/bin/env python3
"""
Remove __ray_call__ entries from Ray timeline JSON traces.

Usage:
  python tools/strip_ray_call.py path/to/trace.json [more.json] [--output OUTPUT]

If --output is omitted, files are rewritten in place. When multiple input files
are provided, --output is ignored and each file is cleaned in place.
"""
import argparse
import json
import pathlib
import sys
from typing import Iterable, Tuple


def clean_trace(path: pathlib.Path) -> Tuple[int, int]:
    """Return (removed, remaining) after filtering a single file."""
    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} is not a Ray timeline list")

    before = len(data)
    filtered = [event for event in data
                if not (isinstance(event, dict) and event.get("name") == "__ray_call__")]
    removed = before - len(filtered)

    with path.open("w") as f:
        json.dump(filtered, f, indent=2)

    return removed, len(filtered)


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description="Strip __ray_call__ events from Ray timeline JSON traces.")
    parser.add_argument("traces", nargs="+", type=pathlib.Path, help="Trace file(s) to clean")
    parser.add_argument("--output", "-o", type=pathlib.Path,
                        help="Optional output file (only valid when cleaning a single input)")
    args = parser.parse_args(list(argv))

    if args.output and len(args.traces) != 1:
        parser.error("--output can only be used with a single input file")

    total_removed = 0

    for trace_path in args.traces:
        target = args.output if args.output else trace_path
        # If writing elsewhere, copy first to avoid overwriting input before read.
        if args.output and target != trace_path:
            target.write_text(trace_path.read_text())
            trace_path = target

        removed, remaining = clean_trace(trace_path)
        total_removed += removed
        print(f"{trace_path}: removed {removed} __ray_call__ events, kept {remaining}")

    return 0 if total_removed >= 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
