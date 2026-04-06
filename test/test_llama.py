import logging
import ray
import torch
import time
import argparse
import os
import numpy as np

logger = logging.getLogger(__name__)

from ray.util.placement_group import (
    placement_group,
    placement_group_table,
)

from src.piper_coordinator import PiperProgramCoordinator
from src.piper_compile import piper_setup
from src.piper import piper_exec_dag
from src.piper_utils import piper_metadata

from .models.llama import Transformer, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B, LLAMA_70B
from .schedule_helpers import (
    build_1f1b_schedule,
    build_gpipe_schedule,
    print_schedule,
    INTERLEAVED_1F1B_PP2_MB4_SCHEDULE,
    INTERLEAVED_1F1B_PP2_MB6_SCHEDULE,
    INTERLEAVED_1F1B_PP4_MB8_SCHEDULE,
    INTERLEAVED_GPIPE_PP2_MB4_SCHEDULE,
    NO_PP_SCHEDULE,
    NO_PP_4STAGE_SCHEDULE,
    DUALPIPEV_MB6_SCHEDULE,
    DUALPIPEV_NOZB_MB6_SCHEDULE,
    ZEROBUBBLE_MB4_SCHEDULE,
)


def main(args, pg):
    match args.model:
        case 'debug':
            llama_config = LLAMA_DEBUG
        case '1b':
            llama_config = LLAMA_1B
        case '3b':
            llama_config = LLAMA_3B
        case '8b':
            llama_config = LLAMA_8B
        case '70b':
            llama_config = LLAMA_70B
    print(args)

    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randint(0, llama_config.vocab_size, (args.batch_size, args.seq_len))
    y = torch.randn((args.batch_size, args.seq_len, llama_config.vocab_size))

    match args.schedule:
        case "no-pp":
            schedule = NO_PP_SCHEDULE
        case "no-pp-4s":
            schedule = NO_PP_4STAGE_SCHEDULE
        case "interleaved-1f1b":
            if args.pp == 2:
                if args.mbs == 6:
                    schedule = INTERLEAVED_1F1B_PP2_MB6_SCHEDULE
                elif args.mbs == 4:
                    schedule = INTERLEAVED_1F1B_PP2_MB4_SCHEDULE
                else:
                    raise ValueError(f"Unsupported number of microbatches for interleaved-1f1b with PP={args.pp}: {args.mbs}")
            elif args.pp == 4:
                assert args.mbs == 8
                schedule = INTERLEAVED_1F1B_PP4_MB8_SCHEDULE
        case "1f1b":
            schedule = build_1f1b_schedule(args.mbs, args.pp)
        case "gpipe":
            schedule = build_gpipe_schedule(args.mbs, args.pp)
        case "interleaved-gpipe":
            assert args.pp == 2 and args.mbs == 4
            schedule = INTERLEAVED_GPIPE_PP2_MB4_SCHEDULE
        case "dualpipev":
            assert args.pp == 2 and args.mbs == 6
            schedule = DUALPIPEV_MB6_SCHEDULE
        case "dualpipev-nozb":
            assert args.pp == 2 and args.mbs == 6
            schedule = DUALPIPEV_NOZB_MB6_SCHEDULE
        case "zerobubble":
            assert args.pp == 2 and args.mbs == 4
            schedule = ZEROBUBBLE_MB4_SCHEDULE

    print("Schedule:")
    print_schedule(schedule)

    piper_setup(
        Transformer,
        model_args=(llama_config, args.seq_len),
        optim_fn=torch.optim.Adam,
        example_inputs=[x],
        example_outputs=y,
        schedule=schedule,
        naive_gradient_sync=args.naive_grad_sync,
        activation_checkpointing=args.activation_checkpointing,
        bucketing=args.bucketing,
        bucket_size=args.bucket_size,
        zero_stage=args.zero_stage,
        schedule_name=args.schedule,
        visualize_dag_render=not args.no_render_dag,
        no_nvtx=args.no_nvtx,
        model_dtype=torch.bfloat16,
        pg=pg,
        nsight=args.nsight,
    )

    print(f"Running {args.warmup} warmup iterations...")
    for _ in range(args.warmup):
        piper_exec_dag(loss_fn)
        time.sleep(1)

    actors = piper_metadata.actors

    print(f"Running {args.iters} timed iterations...")
    # Reset peak memory stats before the timed block so reported peaks reflect only
    # steady-state training, not model loading / warmup allocations.
    ray.get([actor.reset_peak_memory.remote() for actor in actors.values()])
    iter_times = []
    for i, _ in enumerate(range(args.iters)):
        start = time.perf_counter()
        piper_exec_dag(loss_fn)
        end = time.perf_counter()
        iter_times.append(end - start)
        # Collect memory breakdown after the first timed iteration while the
        # allocator is at steady state (params + grads + optimizer states all live).
        if i == 0:
            mem_breakdown = ray.get([actor.get_memory_breakdown.remote() for actor in actors.values()])

    dp_rank = int(os.environ['PIPER_DP_RANK'])
    print(
        f"rank {dp_rank} iter time= {np.mean(iter_times):.5f} ± {np.std(iter_times):.5f} s "
        f"({len(iter_times)} samples)\n"
        f"rank {dp_rank} throughput= "
        f"{(args.batch_size * args.mbs * args.seq_len) / np.mean(iter_times):.3f} tokens/s"
    )

    # Peak GPU memory per actor
    mem_data = ray.get([actor.get_peak_memory.remote() for actor in actors.values()])
    for rank, mem_gb in sorted(mem_data):
        print(f"rank {rank} peak_memory= {mem_gb:.3f} GiB")

    # Fine-grained memory breakdown (measured after first timed iter)
    print("Memory breakdown (after 1st timed iter):")
    for rank, bd in sorted(mem_breakdown):
        print(
            f"  rank {rank}  allocated={bd['allocated_gb']:.3f} GiB  "
            f"reserved={bd['reserved_gb']:.3f} GiB  "
            f"params={bd['params_gb']:.3f} GiB  "
            f"grads={bd['grads_gb']:.3f} GiB  "
            f"shard={bd['shard_gb']:.3f} GiB  "
            f"other(acts+optim)={bd['other_gb']:.3f} GiB"
        )

    if args.tracing:
        ray.get([actor.set_tracing.remote(True) for actor in actors.values()])
        print(f"Running {args.trace_iters} tracing iterations...")
        for _ in range(args.trace_iters):
            piper_exec_dag(loss_fn)
            ray.get([actor.flush_timing_events.remote() for actor in actors.values()])
            time.sleep(1)
        trace_data_ret = ray.get([actor.get_trace_data.remote() for actor in actors.values()])
        for rank, trace_data in trace_data_ret:
            for key in trace_data:
                all_times = trace_data[key]
                print(
                    f"rank {rank} {key} time= {np.mean(all_times):.3f} ± "
                    f"{np.std(all_times):.3f} ms ({len(all_times)} samples)"
                )

    os.makedirs("out", exist_ok=True)
    bucketed_str = "-bucketed" if args.bucketing else ""
    ts = time.strftime("%Y%m%d_%H%M%S")
    timeline_filename = f"out/llama-dag-pp{args.pp}-dp{args.dp}-{args.schedule}-zero{args.zero_stage}{bucketed_str}-{ts}"
    ray.timeline(timeline_filename)
    print(f"Ray timeline saved to: {timeline_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description='Run LLaMA model with pipeline parallelism')
    parser.add_argument('--model', choices=['debug', '1b', '3b', '8b', '70b'], default='debug')
    parser.add_argument(
        '--schedule',
        choices=['gpipe', '1f1b', 'interleaved-1f1b', 'interleaved-gpipe',
                 'dualpipev-nozb', 'dualpipev', 'zerobubble', 'no-pp', 'no-pp-4s'],
        default='1f1b',
    )
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--pp', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--mbs', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--trace-iters', type=int, default=3)
    parser.add_argument('--tracing', action='store_true', default=False)
    parser.add_argument('--naive-grad-sync', action='store_true', default=False)
    parser.add_argument('--activation-checkpointing', action='store_true', default=False)
    parser.add_argument('--bucketing', action='store_true', default=False,
                        help='Split stages into per-param-bucket sub-modules for overlapped all-reduce')
    parser.add_argument('--bucket-size', type=int, default=25 * 1024 * 1024,
                        help='Target bucket size in bytes for --bucketing (default 25 MB)')
    parser.add_argument('--zero-stage', type=int, default=0, choices=[0, 1, 2, 3],
                        help='ZeRO stage: 0=disabled, 1=optim states, 2=+gradients, 3=+parameters')
    parser.add_argument('--nsight', action='store_true', default=False,
                        help='Whether to use Nsight Systems for tracing')
    parser.add_argument('--no-render-dag', action='store_true', default=False,
                        help='Save DAG as .dot source only, skip graphviz rendering (use for large graphs)')
    parser.add_argument('--no-nvtx', action='store_true', default=False,
                        help='Disable NVTX range annotations (note: NVTX is CPU-only and does NOT cause GPU sync; '
                             'the 10x CPU overhead seen in nsys is NCCL per-call overhead, not NVTX)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    ray.init(
        namespace="llama",
        log_to_driver=True,
        include_dashboard=False,
    )
    pg = placement_group([{"CPU": args.pp, "GPU": args.pp}] * args.dp, strategy="PACK")
    ray.get(pg.ready(), timeout=600)
    print(placement_group_table(pg))
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=args.pp, dp_degree=args.dp)
    handles = piper_coordinator.run_program.remote(main, args, pg)
    ray.get(handles)
    ray.shutdown()
