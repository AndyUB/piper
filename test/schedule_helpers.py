from src.piper_exec import Task, CompType


def print_schedule(schedule):
    for stage in schedule:
        for step in stage:
            if step:
                type_char = 'u' if step.type == CompType.UPD else 'f' if step.type == CompType.FWD else 'b'
                string = f"{step.stage_id}:{step.mb_idx}:{type_char}"
            else:
                string = " --- "
            print(string, end="\t")
        print()

def build_gpipe_schedule(n_mbs: int, n_stages: int):
    steps = n_mbs + n_stages - 1
    schedule = [[None] * (steps * 2 + 1) for _ in range(n_stages)]
    for step in range(steps):
        for stage_id in range(n_stages):
            mb_idx = step - stage_id
            if mb_idx >= 0 and mb_idx < n_mbs:
                schedule[stage_id][step] = Task(stage_id, stage_id, mb_idx, CompType.FWD)

    for step in range(steps, steps * 2):
        for stage_id in reversed(range(n_stages)):
            mb_idx = (step - steps) - (n_stages - stage_id - 1)
            if mb_idx >= 0 and mb_idx < n_mbs:
                schedule[stage_id][step] = Task(stage_id, stage_id, mb_idx, CompType.BWD)
    for i, stage in enumerate(range(n_stages)):
        schedule[stage][-i-1] = Task(stage_id=stage, pp_rank=stage, mb_idx=0, type=CompType.UPD)
    return schedule

def build_1f1b_schedule(n_mbs: int, n_stages: int):
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
                        stage_id=stage_id,
                        mb_idx=mb_idx, 
                        type=CompType.FWD
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
                        stage_id=stage_id,
                        mb_idx=mb_idx, 
                        type=task_type
                    )
                    stage_mb[stage_id][fwd_or_bwd] += 1
    for i, stage in enumerate(range(n_stages)):
        schedule[stage][-i-1] = Task(
            stage_id=stage, 
            pp_rank=stage, 
            mb_idx=n_mbs - 1,
            type=CompType.UPD
        )

    return schedule

no_pp_schedule = [
    [
        Task(pp_rank=0, stage_id=0, mb_idx=0, type=CompType.FWD),
        Task(pp_rank=0, stage_id=0, mb_idx=0, type=CompType.BWD),
        Task(pp_rank=0, stage_id=0, mb_idx=0, type=CompType.UPD),
    ]
]

pp2_interleaved_1f1b_grid_schedule = [
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

pp4_interleaved_1f1b_grid_schedule = [
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