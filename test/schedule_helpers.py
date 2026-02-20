from graphviz import Digraph

from src.piper_exec import (
    Task,
    CompType,
    Schedule,
    ScheduleTask,
    MicrobatchMetadata,
    ScheduleEdge,
    TemporalEdge,
    DAGEdge,
    _topological_order,
)


def print_schedule(schedule: Schedule, path: str = "schedule") -> None:
    """Render Schedule DAG to a visual (graphviz) file. Default output: schedule.pdf."""
    print_schedule_dag(schedule, path=path)


def print_schedule_dag(schedule: Schedule, path: str = "schedule") -> None:
    """
    Use graphviz to produce a visual of the schedule DAG.
    Tasks for the same pp_rank are drawn on the same row (one row per rank).
    Creates path.pdf in the current directory.
    """
    dot = Digraph(comment="Schedule DAG")
    dot.attr(rankdir="TB")
    dot.attr("node", shape="box", style="rounded,filled", fontname="sans")

    rank_to_tasks: dict[int, list[int]] = {}
    task_labels: dict[int, str] = {}
    for ti, task in enumerate(schedule.tasks):
        rank_to_tasks.setdefault(task.pp_rank, []).append(ti)
        mb_parts = []
        for mb in task.microbatches:
            c = "u" if mb.comp_type == CompType.UPD else "f" if mb.comp_type == CompType.FWD else "b"
            mb_parts.append(f"{mb.stage_id}:{mb.mb_idx}:{c}")
        task_labels[ti] = f"rank{task.pp_rank}\\n" + "\\n".join(mb_parts)

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


def build_gpipe_schedule(n_mbs: int, n_stages: int) -> Schedule:
    """
    Build a GPipe schedule as a DAG: all FWD, then all BWD, then UPD.
    One task per (stage_id, mb_idx, comp_type); edges follow pipeline data dependencies.
    Temporal: forwards ordered by microbatch per stage (s:m:f → s:m+1:f); all forwards before
    backwards (last fwd → first bwd on last stage); backwards in ascending microbatch per stage
    (s:m:b → s:m+1:b); update after final backward (s:M-1:b → s:UPD).
    """
    tasks: list[ScheduleTask] = []
    edges: list[ScheduleEdge] = []
    temporal_edges: list[TemporalEdge] = []
    producer: dict[tuple[int, int, CompType], int] = {}

    for mb_idx in range(n_mbs):
        for stage_id in range(n_stages):
            mb = MicrobatchMetadata(stage_id=stage_id, mb_idx=mb_idx, comp_type=CompType.FWD)
            ti = len(tasks)
            tasks.append(ScheduleTask(pp_rank=stage_id, microbatches=(mb,)))
            if stage_id > 0:
                edges.append(ScheduleEdge(producer[(stage_id - 1, mb_idx, CompType.FWD)], ti))
            producer[(stage_id, mb_idx, CompType.FWD)] = ti

    for mb_idx in range(n_mbs):
        for stage_id in range(n_stages - 1, -1, -1):
            mb = MicrobatchMetadata(stage_id=stage_id, mb_idx=mb_idx, comp_type=CompType.BWD)
            ti = len(tasks)
            tasks.append(ScheduleTask(pp_rank=stage_id, microbatches=(mb,)))
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
    last_stage = n_stages - 1
    temporal_edges.append(
        TemporalEdge(producer[(last_stage, n_mbs - 1, CompType.FWD)], producer[(last_stage, 0, CompType.BWD)])
    )

    for stage_id in range(n_stages):
        for m in range(n_mbs - 1):
            temporal_edges.append(
                TemporalEdge(producer[(stage_id, m, CompType.BWD)], producer[(stage_id, m + 1, CompType.BWD)])
            )

    for stage_id in range(n_stages):
        ti = len(tasks)
        mb = MicrobatchMetadata(stage_id=stage_id, mb_idx=0, comp_type=CompType.UPD)
        tasks.append(ScheduleTask(pp_rank=stage_id, microbatches=(mb,)))
        temporal_edges.append(TemporalEdge(producer[(stage_id, n_mbs - 1, CompType.BWD)], ti))

    return Schedule(tasks=tasks, edges=edges, temporal_edges=temporal_edges)


def build_1f1b_schedule(n_mbs: int, n_stages: int) -> Schedule:
    """
    Build a 1F1B (one forward one backward) schedule as a DAG.
    FWD and BWD are interleaved per stage; one UPD per stage at the end.
    Temporal (per rank): backwards in ascending microbatch order (s:m:b → s:m+1:b);
    update after final backward (s:M-1:b → s:UPD).
    """
    steps = n_mbs + n_stages - 1
    num_steps = steps * 2 + 1
    schedule_2d: list[list[Task | None]] = [[None] * num_steps for _ in range(n_stages)]
    stage_mb = [[0, 0] for _ in range(n_stages)]

    for step in range(n_stages):
        for stage_id in range(n_stages):
            if step >= stage_id:
                mb_idx = stage_mb[stage_id][0]
                if 0 <= mb_idx < n_mbs:
                    schedule_2d[stage_id][step] = Task(
                        pp_rank=stage_id, stage_id=stage_id, mb_idx=mb_idx, type=CompType.FWD
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
                if 0 <= mb_idx < n_mbs:
                    schedule_2d[stage_id][step] = Task(
                        pp_rank=stage_id, stage_id=stage_id, mb_idx=mb_idx, type=task_type
                    )
                    stage_mb[stage_id][fwd_or_bwd] += 1

    for i, stage in enumerate(range(n_stages)):
        schedule_2d[stage][-i - 1] = Task(
            stage_id=stage, pp_rank=stage, mb_idx=n_mbs - 1, type=CompType.UPD
        )

    tasks = []
    edges = []
    temporal_edges: list[TemporalEdge] = []
    producer: dict[tuple[int, int, CompType], int] = {}
    for time_step in range(num_steps):
        for rank in range(n_stages - 1, -1, -1):
            cell = schedule_2d[rank][time_step]
            if cell is None:
                continue
            mb = MicrobatchMetadata(
                stage_id=cell.stage_id, mb_idx=cell.mb_idx, comp_type=cell.type
            )
            ti = len(tasks)
            tasks.append(ScheduleTask(pp_rank=cell.pp_rank, microbatches=(mb,)))
            c, s, m = cell.type, cell.stage_id, cell.mb_idx
            if c == CompType.FWD:
                if s > 0:
                    edges.append(ScheduleEdge(producer[(s - 1, m, c)], ti))
                producer[(s, m, c)] = ti
            elif c == CompType.BWD:
                # Same-stage fwd→bwd only on last stage (loss and backward there)
                if s == n_stages - 1:
                    edges.append(ScheduleEdge(producer[(s, m, CompType.FWD)], ti))
                if s < n_stages - 1:
                    edges.append(ScheduleEdge(producer[(s + 1, m, c)], ti))
                producer[(s, m, c)] = ti
            elif c == CompType.UPD:
                temporal_edges.append(TemporalEdge(producer[(s, n_mbs - 1, CompType.BWD)], ti))

    for s in range(n_stages):
        for m in range(n_mbs - 1):
            temporal_edges.append(
                TemporalEdge(producer[(s, m, CompType.BWD)], producer[(s, m + 1, CompType.BWD)])
            )

    return Schedule(tasks=tasks, edges=edges, temporal_edges=temporal_edges)


def _grid_to_dag(
    schedule_2d: list[list[Task | None]],
    n_stages: int,
) -> Schedule:
    """Convert a 2D grid (device x time_step) of Tasks into a Schedule DAG. Sequential pipeline."""
    dag_edges = [DAGEdge(i, i + 1) for i in range(n_stages - 1)]
    num_devices = len(schedule_2d)
    num_steps = len(schedule_2d[0]) if schedule_2d else 0
    tasks_list: list[ScheduleTask] = []
    cell_to_task: dict[tuple[int, int], int] = {}
    for j in range(num_devices):
        for i in range(num_steps):
            task = schedule_2d[j][i] if j < len(schedule_2d) and i < len(schedule_2d[j]) else None
            if task is not None:
                mb = MicrobatchMetadata(
                    stage_id=task.stage_id, mb_idx=task.mb_idx, comp_type=task.type
                )
                tasks_list.append(ScheduleTask(pp_rank=task.pp_rank, microbatches=(mb,)))
                cell_to_task[(j, i)] = len(tasks_list) - 1
    order = []
    for i in range(num_steps):
        for j in range(num_devices - 1, -1, -1):
            if (j, i) in cell_to_task:
                order.append((j, i))
    edges_list: list[ScheduleEdge] = []
    producer: dict[tuple[int, int, CompType], int] = {}
    for (j, i) in order:
        ti = cell_to_task[(j, i)]
        st = tasks_list[ti]
        for mb in st.microbatches:
            s, m, c = mb.stage_id, mb.mb_idx, mb.comp_type
            if c == CompType.UPD:
                continue
            if c == CompType.FWD and s > 0:
                pred_key = (s - 1, m, c)
                if pred_key in producer:
                    edges_list.append(ScheduleEdge(producer[pred_key], ti))
            if c == CompType.BWD:
                if s == n_stages - 1 and (s, m, CompType.FWD) in producer:
                    edges_list.append(ScheduleEdge(producer[(s, m, CompType.FWD)], ti))
                if s < n_stages - 1:
                    pred_key = (s + 1, m, c)
                    if pred_key in producer:
                        edges_list.append(ScheduleEdge(producer[pred_key], ti))
            producer[(s, m, c)] = ti
    return Schedule(tasks=tasks_list, edges=edges_list, temporal_edges=[])


no_pp_schedule = Schedule(
    tasks=[
        ScheduleTask(0, (MicrobatchMetadata(0, 0, CompType.FWD),)),
        ScheduleTask(0, (MicrobatchMetadata(0, 0, CompType.BWD),)),
        ScheduleTask(0, (MicrobatchMetadata(0, 0, CompType.UPD),)),
    ],
    edges=[],
    temporal_edges=[],
)

# 2D grid literals for interleaved 1F1B (used only to build DAGs below)
_PP2_GRID = [
    [
        Task(pp_rank=0, stage_id=0, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=0, stage_id=0, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=0, stage_id=2, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=0, stage_id=2, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=0, stage_id=0, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=0, stage_id=0, mb_idx=3, type=CompType.FWD),
        Task(pp_rank=0, stage_id=2, mb_idx=0, type=CompType.BWD),
        None,
        Task(pp_rank=0, stage_id=2, mb_idx=1, type=CompType.BWD),
        Task(pp_rank=0, stage_id=2, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=0, stage_id=0, mb_idx=0, type=CompType.BWD),
        Task(pp_rank=0, stage_id=2, mb_idx=3, type=CompType.FWD),
        Task(pp_rank=0, stage_id=0, mb_idx=1, type=CompType.BWD),
        None,
        Task(pp_rank=0, stage_id=2, mb_idx=2, type=CompType.BWD),
        Task(pp_rank=0, stage_id=2, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=0, stage_id=0, mb_idx=2, type=CompType.BWD),
        Task(pp_rank=0, stage_id=0, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=0, stage_id=0, mb_idx=0, type=CompType.UPD),
    ],
    [
        None,
        Task(pp_rank=1, stage_id=1, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=1, stage_id=1, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=1, stage_id=3, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=1, stage_id=3, mb_idx=0, type=CompType.BWD),
        Task(pp_rank=1, stage_id=3, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=1, stage_id=3, mb_idx=1, type=CompType.BWD),
        Task(pp_rank=1, stage_id=1, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=1, stage_id=1, mb_idx=0, type=CompType.BWD),
        Task(pp_rank=1, stage_id=1, mb_idx=3, type=CompType.FWD),
        Task(pp_rank=1, stage_id=1, mb_idx=1, type=CompType.BWD),
        Task(pp_rank=1, stage_id=3, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=1, stage_id=3, mb_idx=2, type=CompType.BWD),
        Task(pp_rank=1, stage_id=3, mb_idx=3, type=CompType.FWD),
        Task(pp_rank=1, stage_id=3, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=1, stage_id=1, mb_idx=2, type=CompType.BWD),
        Task(pp_rank=1, stage_id=1, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=1, stage_id=1, mb_idx=0, type=CompType.UPD),
        None,
    ],
]
pp2_interleaved_1f1b_grid_schedule = _grid_to_dag(_PP2_GRID, 4)

_PP4_GRID = [
    [
        Task(pp_rank=0, stage_id=0, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=0, stage_id=0, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=0, stage_id=0, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=0, stage_id=0, mb_idx=3, type=CompType.FWD),
        Task(pp_rank=0, stage_id=4, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=0, stage_id=4, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=0, stage_id=4, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=0, stage_id=4, mb_idx=3, type=CompType.FWD),
        None,
        None,
        None,
        Task(pp_rank=0, stage_id=4, mb_idx=0, type=CompType.BWD),
        None,
        Task(pp_rank=0, stage_id=4, mb_idx=1, type=CompType.BWD),
        None,
        Task(pp_rank=0, stage_id=4, mb_idx=2, type=CompType.BWD),
        None,
        Task(pp_rank=0, stage_id=4, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=0, stage_id=0, mb_idx=0, type=CompType.BWD),
        Task(pp_rank=0, stage_id=0, mb_idx=1, type=CompType.BWD),
        Task(pp_rank=0, stage_id=0, mb_idx=2, type=CompType.BWD),
        Task(pp_rank=0, stage_id=0, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=0, stage_id=0, mb_idx=0, type=CompType.UPD),
    ],
    [
        None,
        Task(pp_rank=1, stage_id=1, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=1, stage_id=1, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=1, stage_id=1, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=1, stage_id=1, mb_idx=3, type=CompType.FWD),
        Task(pp_rank=1, stage_id=5, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=1, stage_id=5, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=1, stage_id=5, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=1, stage_id=5, mb_idx=3, type=CompType.FWD),
        None,
        Task(pp_rank=1, stage_id=5, mb_idx=0, type=CompType.BWD),
        None,
        Task(pp_rank=1, stage_id=5, mb_idx=1, type=CompType.BWD),
        None,
        Task(pp_rank=1, stage_id=5, mb_idx=2, type=CompType.BWD),
        None,
        Task(pp_rank=1, stage_id=5, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=1, stage_id=1, mb_idx=0, type=CompType.BWD),
        Task(pp_rank=1, stage_id=1, mb_idx=1, type=CompType.BWD),
        Task(pp_rank=1, stage_id=1, mb_idx=2, type=CompType.BWD),
        Task(pp_rank=1, stage_id=1, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=1, stage_id=1, mb_idx=0, type=CompType.UPD),
        None,
    ],
    [
        None,
        None,
        Task(pp_rank=2, stage_id=2, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=2, stage_id=2, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=2, stage_id=2, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=2, stage_id=2, mb_idx=3, type=CompType.FWD),
        Task(pp_rank=2, stage_id=6, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=2, stage_id=6, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=2, stage_id=6, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=2, stage_id=6, mb_idx=0, type=CompType.BWD),
        Task(pp_rank=2, stage_id=6, mb_idx=3, type=CompType.FWD),
        Task(pp_rank=2, stage_id=6, mb_idx=1, type=CompType.BWD),
        None,
        Task(pp_rank=2, stage_id=6, mb_idx=2, type=CompType.BWD),
        None,
        Task(pp_rank=2, stage_id=6, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=2, stage_id=2, mb_idx=0, type=CompType.BWD),
        Task(pp_rank=2, stage_id=2, mb_idx=1, type=CompType.BWD),
        Task(pp_rank=2, stage_id=2, mb_idx=2, type=CompType.BWD),
        Task(pp_rank=2, stage_id=2, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=2, stage_id=2, mb_idx=0, type=CompType.UPD),
        None,
        None,
    ],
    [
        None,
        None,
        None,
        Task(pp_rank=3, stage_id=3, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=3, stage_id=3, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=3, stage_id=3, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=3, stage_id=3, mb_idx=3, type=CompType.FWD),
        Task(pp_rank=3, stage_id=7, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=3, stage_id=7, mb_idx=0, type=CompType.BWD),
        Task(pp_rank=3, stage_id=7, mb_idx=1, type=CompType.FWD),
        Task(pp_rank=3, stage_id=7, mb_idx=1, type=CompType.BWD),
        Task(pp_rank=3, stage_id=7, mb_idx=2, type=CompType.FWD),
        Task(pp_rank=3, stage_id=7, mb_idx=2, type=CompType.BWD),
        Task(pp_rank=3, stage_id=7, mb_idx=3, type=CompType.FWD),
        Task(pp_rank=3, stage_id=7, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=3, stage_id=3, mb_idx=0, type=CompType.BWD),
        Task(pp_rank=3, stage_id=3, mb_idx=1, type=CompType.BWD),
        Task(pp_rank=3, stage_id=3, mb_idx=2, type=CompType.BWD),
        Task(pp_rank=3, stage_id=3, mb_idx=3, type=CompType.BWD),
        Task(pp_rank=3, stage_id=3, mb_idx=0, type=CompType.UPD),
        None,
        None,
        None,
    ],
]
pp4_interleaved_1f1b_grid_schedule = _grid_to_dag(_PP4_GRID, 8)