#!/usr/bin/env python3
import argparse
import glob
import os
from collections import defaultdict

import torch


def _parse_key(meta: dict) -> tuple:
    return (
        meta["pass"],
        meta["src_stage"],
        meta["dst_stage"],
        meta["mb_idx"],
        meta["tensor_idx"],
        meta.get("dp_rank", 0),
    )


def compare_logged_tensors(log_dir: str):
    tensor_files = glob.glob(os.path.join(log_dir, "*.pt"))
    if not tensor_files:
        raise RuntimeError(f"No tensor logs found in {log_dir}")

    sends = defaultdict(list)
    recvs = defaultdict(list)

    for path in tensor_files:
        payload = torch.load(path, map_location="cpu")
        meta = payload["meta"]
        key = _parse_key(meta)
        item = (path, payload["tensor"])
        if meta["direction"] == "send":
            sends[key].append(item)
        elif meta["direction"] == "recv":
            recvs[key].append(item)
        else:
            raise RuntimeError(f"Unknown direction {meta['direction']} for {path}")

    errors = []
    checked_pairs = 0

    for key in sorted(set(sends.keys()) | set(recvs.keys())):
        send_items = sends.get(key, [])
        recv_items = recvs.get(key, [])
        if len(send_items) != 1 or len(recv_items) != 1:
            errors.append(
                f"Key {key}: expected 1 send and 1 recv, got {len(send_items)} send / {len(recv_items)} recv"
            )
            continue

        send_path, send_tensor = send_items[0]
        recv_path, recv_tensor = recv_items[0]
        checked_pairs += 1

        if send_tensor.shape != recv_tensor.shape:
            errors.append(
                f"Key {key}: shape mismatch send {tuple(send_tensor.shape)} ({send_path}) recv {tuple(recv_tensor.shape)} ({recv_path})"
            )
            continue

        diff = (send_tensor - recv_tensor).abs()
        max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
        if not torch.equal(send_tensor, recv_tensor):
            errors.append(
                f"Key {key}: tensor mismatch max_diff={max_diff:.8e} send={send_path} recv={recv_path}"
            )

    return checked_pairs, errors


def main():
    parser = argparse.ArgumentParser(description="Compare Piper P2P sent and received tensor logs.")
    parser.add_argument("--log-dir", required=True, help="Directory containing *.pt P2P tensor logs.")
    args = parser.parse_args()

    checked_pairs, errors = compare_logged_tensors(args.log_dir)

    if errors:
        print("P2P tensor comparison failed:")
        for err in errors:
            print(f" - {err}")
        raise SystemExit(1)

    print(f"P2P tensor comparison passed. Checked {checked_pairs} send/recv pairs.")


if __name__ == "__main__":
    main()
