import argparse
import os
import time
import numpy as np
import ray
import torch

from piper.exec import piper_exec
from piper.compile import piper_setup
from piper.piper import piper
from piper.utils import piper_metadata

from .models.llama import Transformer, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B
from .schedule_helpers import (
    build_1f1b_schedule,
    build_zb1p_schedule,
    print_schedule,
    no_pp_schedule,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CUDA-event timing for LLaMA pipeline schedules")
    parser.add_argument(
        "--model",
        choices=["LLAMA_DEBUG", "LLAMA_1B", "LLAMA_3B", "LLAMA_8B"],
        default="LLAMA_DEBUG",
    )
    parser.add_argument("--schedule", choices=["1f1b", "zb1p", "no-pp"], default="1f1b")
    parser.add_argument("--num_stages", type=int, default=2, help="Number of stages (default: 2)")
    parser.add_argument("--dp_degree", type=int, default=1, help="Data parallel degree (default: 1)")
    parser.add_argument("--pp_degree", type=int, default=2, help="Pipeline parallel degree (default: 2)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_mbs", type=int, default=4, help="Number of microbatches")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--tracing",
        action="store_true",
        default=False,
        help="Enable CUDA-event tracing inside actors (required for timings)",
    )
    return parser.parse_args()


def select_model(name: str):
    if name == "LLAMA_DEBUG":
        return LLAMA_DEBUG
    if name == "LLAMA_1B":
        return LLAMA_1B
    if name == "LLAMA_3B":
        return LLAMA_3B
    if name == "LLAMA_8B":
        return LLAMA_8B
    raise ValueError(f"Unknown model {name}")


def build_schedule(schedule_name: str, num_mbs: int, num_stages: int):
    if schedule_name == "1f1b":
        sched = build_1f1b_schedule(num_mbs, num_stages)
        sched[0][2] = sched[0][4]
        sched[0][4] = sched[0][6]
        sched[0][6] = None
        return sched
    if schedule_name == "zb1p":
        return build_zb1p_schedule(num_mbs, num_stages)
    if schedule_name == "no-pp":
        return no_pp_schedule
    raise NotImplementedError(f"Schedule {schedule_name} not supported for event timing.")


def print_mean_timing_data(trace_data: dict, actor_id: int) -> None:
    """Pretty-print mean timings collected via CUDA events."""
    print(f"\nTiming statistics for Actor {actor_id}:")
    for stage_id, stage_data in trace_data.items():
        if stage_id == "update":
            print("  Update:")
            for metric, values in stage_data.items():
                mean_val = float("nan") if not values else float(np.mean(values))
                unit = "GB" if "memory" in metric else "ms"
                print(f"    {metric}: {mean_val:.3f} {unit}")
            continue

        print(f"  Stage {stage_id}:")
        for phase_name, phase_data in stage_data.items():
            if not phase_data:
                continue
            print(f"    {phase_name}:")
            for metric, values in sorted(phase_data.items()):
                mean_val = float("nan") if not values else float(np.mean(values))
                unit = "GB" if "memory" in metric else "ms"
                if metric == "graph_pruning":
                    print(f"      {metric} (graph analysis/pruning overhead): {mean_val:.3f} {unit}")
                else:
                    print(f"      {metric}: {mean_val:.3f} {unit}")


def main(args):
    llama_config = select_model(args.model)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda"

    batch_size = args.batch_size
    num_mbs = args.num_mbs
    seq_len = args.seq_len
    warmup = args.warmup
    iters = args.iters

    x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len)).to(device)
    y = torch.zeros((batch_size, llama_config.vocab_size), dtype=torch.long).to(device)

    if args.dp_degree > 1:
        dp_rank = int(os.environ["PIPER_DP_RANK"])
        torch.manual_seed(dp_rank)
        x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len)).to(device)
        torch.manual_seed(0)

    model = Transformer(llama_config, seq_len, device).to(device)
    compiled = piper_setup(model, [x], backend=piper, num_stages=args.num_stages, pp_size=args.pp_degree)
    assert args.num_stages == len(piper_metadata.dag) + 1

    schedule = build_schedule(args.schedule, num_mbs, args.num_stages)
    print("SCHEDULE:")
    print_schedule(schedule)

    actors = piper_metadata.actors

    ray.get([actor.set_tracing.remote(True if args.tracing is None else args.tracing) for actor in actors.values()])

    def iter_schedule():
        out = piper_exec(compiled, schedule, [x], y, loss_fn, num_mbs, args.num_stages)
        ray.get(out)

    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        iter_schedule()

    ray.get([actor.clear_trace_data.remote() for actor in actors.values()])
    ray.get([actor.reset_peak_memory.remote() for actor in actors.values()])

    print(f"Running {iters} timed iterations...")
    torch.cuda.synchronize()
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    iter_start.record()
    for _ in range(iters):
        iter_schedule()
    torch.cuda.synchronize()
    iter_end.record()

    elapsed_ms = iter_start.elapsed_time(iter_end)
    total_s = elapsed_ms / 1000.0

    print(f"Iteration time: {elapsed_ms / iters:.0f} ms")
    print(f"{args.schedule} throughput: {(iters * batch_size * num_mbs * seq_len) / total_s:.0f} tokens/sec")

    for actor in actors.values():
        trace = ray.get(actor.get_trace_data.remote())
        actor_id = ray.get(actor.id.remote())
        print_mean_timing_data(trace, actor_id)

    ray.timeline(f"out/{args.model}-pp{args.pp_degree}-dp{args.dp_degree}-{args.schedule}-events.json")


if __name__ == "__main__":
    ray.init(include_dashboard=False, log_to_driver=True, namespace="llama-events")
    args = parse_args()
    main(args)
    ray.shutdown()
