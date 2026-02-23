from graphviz import Digraph

from src.piper_exec import (
    Task,
    CompType,
    Schedule2D,
)
# DAG types used only by legacy _dag builders (kept for reference)
from src.piper_exec import (
    Schedule,
    ScheduleTask,
    BatchMeta,
    ScheduleEdge,
    TemporalEdge,
    DAGEdge,
    _topological_order,
)


def _task_label(task: Task) -> str:
    """Short label for a task: stage:mb:f/b/u, or first+second for dual batches."""
    c = (
        "u"
        if task.type == CompType.UPD
        else "f"
        if task.type == CompType.FWD
        else "b"
        if task.type == CompType.BWD
        else "fb"
    )
    if task.type == CompType.UPD:
        label = "   u    "
    else:
        first = task.batches[0]
        label = f"{first.stage_id}:{first.mb_idx}:{c}"
    if len(task.batches) > 1:
        second = task.batches[1]
        label += f"+{second.stage_id}:{second.mb_idx}"
    else:
        label += "    "
    return label


def print_schedule(schedule: Schedule2D, path: str = "schedule") -> None:
    """Print the 2D schedule grid to stdout. Rows = ranks, columns = time steps. None cells shown as ' --- '."""
    for row in schedule.grid:
        for task in row:
            if task is None:
                print(" ------ ", end="\t")
            else:
                print(_task_label(task), end="\t")
        print()


def print_schedule_dag(schedule: Schedule, path: str = "schedule") -> None:
    """
    Use graphviz to produce a visual of a Schedule DAG (legacy; not used in Piper workflow).
    """
    dot = Digraph(comment="Schedule DAG")
    dot.attr(rankdir="TB")
    dot.attr("node", shape="box", style="rounded,filled", fontname="sans")

    topo_order = _topological_order(schedule)
    task_to_order: dict[int, int] = {task_idx: order_idx for order_idx, task_idx in enumerate(topo_order)}

    rank_to_tasks: dict[int, list[int]] = {}
    task_labels: dict[int, str] = {}
    for ti, task in enumerate(schedule.tasks):
        rank_to_tasks.setdefault(task.pp_rank, []).append(ti)
        mb = task.microbatch
        c = "u" if mb.comp_type == CompType.UPD else "f" if mb.comp_type == CompType.FWD else "b"
        order_num = task_to_order.get(ti, -1)
        task_labels[ti] = f"[{order_num}]\\nrank{task.pp_rank}\\n{mb.stage_id}:{mb.mb_idx}:{c}"

    for rank in sorted(rank_to_tasks.keys()):
        with dot.subgraph() as sub:
            sub.attr(rank="same")
            for ti in rank_to_tasks[rank]:
                sub.node(str(ti), label=task_labels[ti])

    for e in schedule.edges:
        dot.edge(str(e.from_task_idx), str(e.to_task_idx), color="red")

    for e in schedule.temporal_edges or []:
        dot.edge(str(e.from_task_idx), str(e.to_task_idx), style="dotted", color="blue")

    dot.render(path, format="pdf", cleanup=True)


def build_gpipe_schedule(n_mbs: int, n_stages: int) -> Schedule2D:
    steps = n_mbs + n_stages - 1
    schedule = [[None] * (steps * 2 + 1) for _ in range(n_stages)]
    for step in range(steps):
        for stage_id in range(n_stages):
            mb_idx = step - stage_id
            if mb_idx >= 0 and mb_idx < n_mbs:
                schedule[stage_id][step] = Task(
                    pp_rank=stage_id,
                    batches=[BatchMeta(stage_id=stage_id, mb_idx=mb_idx)],
                    type=CompType.FWD,
                )

    for step in range(steps, steps * 2):
        for stage_id in reversed(range(n_stages)):
            mb_idx = (step - steps) - (n_stages - stage_id - 1)
            if mb_idx >= 0 and mb_idx < n_mbs:
                schedule[stage_id][step] = Task(
                    pp_rank=stage_id,
                    batches=[BatchMeta(stage_id=stage_id, mb_idx=mb_idx)],
                    type=CompType.BWD,
                )

    for i, stage in enumerate(range(n_stages)):
        schedule[stage][-i-1] = Task(
            pp_rank=stage,
            batches=[BatchMeta(stage_id=stage, mb_idx=0)],
            type=CompType.UPD,
        )
    return Schedule2D(grid=schedule)


def build_1f1b_schedule(n_mbs: int, n_stages: int) -> Schedule2D:
    steps = n_mbs + n_stages - 1
    schedule = [[None] * (steps * 2 + 1) for _ in range(n_stages)]
    stage_mb = [[0, 0] for _ in range(n_stages)]
    for step in range(n_stages):
        for stage_id in range(n_stages):
            if step >= stage_id:
                mb_idx = stage_mb[stage_id][0]
                if mb_idx >= 0 and mb_idx < n_mbs:
                    schedule[stage_id][step] = Task(
                        pp_rank=stage_id,
                        batches=[BatchMeta(stage_id=stage_id, mb_idx=mb_idx)],
                        type=CompType.FWD,
                    )
                    stage_mb[stage_id][0] += 1
    for step in range(n_stages, 2 * steps):
        relative_step = step - n_stages
        for stage_id in range(n_stages):
            inv_stage = n_stages - stage_id - 1
            if relative_step >= inv_stage:
                fwd_or_bwd = 1 - (relative_step + inv_stage) % 2
                task_type = CompType.FWD if fwd_or_bwd == 0 else CompType.BWD
                mb_idx = stage_mb[stage_id][fwd_or_bwd]
                if mb_idx >= 0 and mb_idx < n_mbs:
                    schedule[stage_id][step] = Task(
                        pp_rank=stage_id,
                        batches=[BatchMeta(stage_id=stage_id, mb_idx=mb_idx)],
                        type=task_type,
                    )
                    stage_mb[stage_id][fwd_or_bwd] += 1
    for i, stage in enumerate(range(n_stages)):
        schedule[stage][-i-1] = Task(
            pp_rank=stage,
            batches=[BatchMeta(stage_id=stage, mb_idx=n_mbs - 1)],
            type=CompType.UPD,
        )
    return Schedule2D(grid=schedule)


# --- DAG-based schedule builders (kept for reference) ---


def build_gpipe_schedule_dag(n_mbs: int, n_stages: int) -> Schedule:
    """
    Build a GPipe schedule as a DAG: all FWD, then all BWD, then UPD.
    One task per (stage_id, mb_idx, comp_type); edges follow pipeline data dependencies.
    Temporal: forwards ordered by microbatch per stage (s:m:f → s:m+1:f); all forwards before
    backwards (last fwd → first bwd on every stage); backwards in ascending microbatch per stage
    (s:m:b → s:m+1:b); update after final backward (s:M-1:b → s:UPD).
    """
    tasks: list[ScheduleTask] = []
    edges: list[ScheduleEdge] = []
    temporal_edges: list[TemporalEdge] = []
    producer: dict[tuple[int, int, CompType], int] = {}

    for mb_idx in range(n_mbs):
        for stage_id in range(n_stages):
            mb = BatchMeta(stage_id=stage_id, mb_idx=mb_idx, comp_type=CompType.FWD)
            ti = len(tasks)
            tasks.append(ScheduleTask(pp_rank=stage_id, microbatch=mb))
            if stage_id > 0:
                edges.append(ScheduleEdge(producer[(stage_id - 1, mb_idx, CompType.FWD)], ti))
            producer[(stage_id, mb_idx, CompType.FWD)] = ti

    for mb_idx in range(n_mbs):
        for stage_id in range(n_stages - 1, -1, -1):
            mb = BatchMeta(stage_id=stage_id, mb_idx=mb_idx, comp_type=CompType.BWD)
            ti = len(tasks)
            tasks.append(ScheduleTask(pp_rank=stage_id, microbatch=mb))
            # Same-stage fwd→bwd only on last stage (loss and backward there)
            if stage_id == n_stages - 1:
                edges.append(ScheduleEdge(producer[(stage_id, mb_idx, CompType.FWD)], ti))
            if stage_id < n_stages - 1:
                edges.append(ScheduleEdge(producer[(stage_id + 1, mb_idx, CompType.BWD)], ti))
            producer[(stage_id, mb_idx, CompType.BWD)] = ti

    for stage_id in range(n_stages):
        for m in range(n_mbs - 1):
            temporal_edges.append(
                TemporalEdge(producer[(stage_id, m, CompType.FWD)], producer[(stage_id, m + 1, CompType.FWD)])
            )
    for stage_id in range(n_stages):
        temporal_edges.append(
            TemporalEdge(producer[(stage_id, n_mbs - 1, CompType.FWD)], producer[(stage_id, 0, CompType.BWD)])
        )

    for stage_id in range(n_stages):
        for m in range(n_mbs - 1):
            temporal_edges.append(
                TemporalEdge(producer[(stage_id, m, CompType.BWD)], producer[(stage_id, m + 1, CompType.BWD)])
            )

    for stage_id in range(n_stages):
        ti = len(tasks)
        mb = BatchMeta(stage_id=stage_id, mb_idx=0, comp_type=CompType.UPD)
        tasks.append(ScheduleTask(pp_rank=stage_id, microbatch=mb))
        temporal_edges.append(TemporalEdge(producer[(stage_id, n_mbs - 1, CompType.BWD)], ti))

    return Schedule(tasks=tasks, edges=edges, temporal_edges=temporal_edges)


def _dualpipe_pp_rank(stage_id: int, mb_idx: int) -> int:
    """Stage 0 appears on both ranks; mb 0 uses rank 0 for stage 0, mb 1 uses rank 1 for stage 0."""
    return (stage_id + mb_idx) % 2


def build_1f1b_schedule_dag(n_mbs: int, n_stages: int) -> Schedule:
    """
    Build a 1F1B (one forward one backward) schedule as a DAG.
    FWD and BWD are interleaved per stage; one UPD per stage at the end.
    Temporal dependencies per rank (R = number of ranks):
    - First rank: first R forwards, then alternate backward/forward for the remainder
    - Rank r: first (R - r) forwards, then alternate backward/forward
    - Last rank: alternate forward/backward from the beginning (f0, b0, f1, b1, ...)
    """
    R = n_stages

    # Build all tasks first (one task per stage_id, mb_idx, comp_type)
    tasks: list[ScheduleTask] = []
    edges: list[ScheduleEdge] = []
    temporal_edges: list[TemporalEdge] = []
    producer: dict[tuple[int, int, CompType], int] = {}
    
    # Create all forward tasks
    for mb_idx in range(n_mbs):
        for stage_id in range(n_stages):
            mb = BatchMeta(stage_id=stage_id, mb_idx=mb_idx, comp_type=CompType.FWD)
            ti = len(tasks)
            tasks.append(ScheduleTask(pp_rank=stage_id, microbatch=mb))
            if stage_id > 0:
                edges.append(ScheduleEdge(producer[(stage_id - 1, mb_idx, CompType.FWD)], ti))
            producer[(stage_id, mb_idx, CompType.FWD)] = ti
    
    # Create all backward tasks (in reverse stage order to handle dependencies)
    for mb_idx in range(n_mbs):
        for stage_id in range(n_stages - 1, -1, -1):
            mb = BatchMeta(stage_id=stage_id, mb_idx=mb_idx, comp_type=CompType.BWD)
            ti = len(tasks)
            tasks.append(ScheduleTask(pp_rank=stage_id, microbatch=mb))
            # Same-stage fwd→bwd only on last stage (loss and backward there)
            if stage_id == n_stages - 1:
                edges.append(ScheduleEdge(producer[(stage_id, mb_idx, CompType.FWD)], ti))
            if stage_id < n_stages - 1:
                edges.append(ScheduleEdge(producer[(stage_id + 1, mb_idx, CompType.BWD)], ti))
            producer[(stage_id, mb_idx, CompType.BWD)] = ti
    
    # Create update tasks
    for stage_id in range(n_stages):
        mb = BatchMeta(stage_id=stage_id, mb_idx=0, comp_type=CompType.UPD)
        ti = len(tasks)
        tasks.append(ScheduleTask(pp_rank=stage_id, microbatch=mb))
        temporal_edges.append(TemporalEdge(producer[(stage_id, n_mbs - 1, CompType.BWD)], ti))
    
    # Build temporal edges per rank according to the specified pattern
    for rank in range(n_stages):
        stage_id = rank
        
        if rank == n_stages - 1:
            # Last rank: alternate forward/backward (f0, b0, f1, b1, ...)
            for mb_idx in range(n_mbs):
                if mb_idx == 0:
                    # First forward doesn't have a predecessor
                    pass
                else:
                    # Forward follows previous backward
                    temporal_edges.append(
                        TemporalEdge(
                            producer[(stage_id, mb_idx - 1, CompType.BWD)],
                            producer[(stage_id, mb_idx, CompType.FWD)]
                        )
                    )
                # Backward follows its corresponding forward
                temporal_edges.append(
                    TemporalEdge(
                        producer[(stage_id, mb_idx, CompType.FWD)],
                        producer[(stage_id, mb_idx, CompType.BWD)]
                    )
                )
        else:
            # Rank r: first (R - r) forwards, then alternate backward/forward
            num_initial_fwds = R - rank
            # Initial forwards in sequence: f0 → f1 → ... → f(num_initial_fwds - 1)
            for mb_idx in range(num_initial_fwds):
                if mb_idx > 0:
                    temporal_edges.append(
                        TemporalEdge(
                            producer[(stage_id, mb_idx - 1, CompType.FWD)],
                            producer[(stage_id, mb_idx, CompType.FWD)]
                        )
                    )
            # Then alternate: b0, f(num_initial_fwds), b1, f(num_initial_fwds+1), ...
            bwd_idx = 0
            fwd_idx = num_initial_fwds
            # First in alternation is b0, following last initial forward
            prev_mb_idx = num_initial_fwds - 1
            prev_type = CompType.FWD

            while bwd_idx < n_mbs or fwd_idx < n_mbs:
                if prev_type == CompType.FWD:
                    # Next is backward(mb bwd_idx)
                    if bwd_idx < n_mbs:
                        temporal_edges.append(
                            TemporalEdge(
                                producer[(stage_id, prev_mb_idx, CompType.FWD)],
                                producer[(stage_id, bwd_idx, CompType.BWD)]
                            )
                        )
                        prev_mb_idx = bwd_idx
                        prev_type = CompType.BWD
                        bwd_idx += 1
                    elif fwd_idx < n_mbs:
                        temporal_edges.append(
                            TemporalEdge(
                                producer[(stage_id, prev_mb_idx, CompType.FWD)],
                                producer[(stage_id, fwd_idx, CompType.FWD)]
                            )
                        )
                        prev_mb_idx = fwd_idx
                        fwd_idx += 1
                else:  # prev_type == CompType.BWD
                    # Next is forward(mb fwd_idx)
                    if fwd_idx < n_mbs:
                        temporal_edges.append(
                            TemporalEdge(
                                producer[(stage_id, prev_mb_idx, CompType.BWD)],
                                producer[(stage_id, fwd_idx, CompType.FWD)]
                            )
                        )
                        prev_mb_idx = fwd_idx
                        prev_type = CompType.FWD
                        fwd_idx += 1
                    elif bwd_idx < n_mbs:
                        temporal_edges.append(
                            TemporalEdge(
                                producer[(stage_id, prev_mb_idx, CompType.BWD)],
                                producer[(stage_id, bwd_idx, CompType.BWD)]
                            )
                        )
                        prev_mb_idx = bwd_idx
                        bwd_idx += 1
    
    return Schedule(tasks=tasks, edges=edges, temporal_edges=temporal_edges)


NO_PP_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.UPD),
        ]
    ],
)

DUALPIPEV_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=2), BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4), BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=3), BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5), BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4), BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5), BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=0, batches=[], type=CompType.UPD),
        ],
        [
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2), BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD_BWD),
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=2), BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3), BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=3), BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4), BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4), BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5), BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5), BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=1, batches=[], type=CompType.UPD),
            None,
        ]
    ])

# 2D grid literals for interleaved 1F1B (used only to build DAGs below)
INTERLEAVED_1F1B_PP2_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.UPD),
        ],
        [
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.UPD),
            None,
        ],
    ])

INTERLEAVED_1F1B_PP4_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.FWD),
            None,
            None,
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.UPD),
        ],
        [
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.FWD),
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.BWD),
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.BWD),
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.BWD),
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.UPD),
            None,
        ],
        [
            None,
            None,
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.BWD),
            None,
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.BWD),
            None,
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.UPD),
            None,
            None,
        ],
        [
            None,
            None,
            None,
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.UPD),
            None,
            None,
            None,
        ],
    ])
