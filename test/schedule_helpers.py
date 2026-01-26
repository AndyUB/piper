from src.piper_exec import Task


def print_schedule(schedule):
    for stage in schedule:
        for step in stage:
            if step:
                string = f"{step.stage_id}:{step.mb_idx}:{'u' if step.upd else 'f' if step.is_fwd else 'b'}"
            else:
                string = " -- "
            print(string, end="\t")
        print()

def build_gpipe_schedule(n_mbs: int, n_stages: int):
    steps = n_mbs + n_stages - 1
    schedule = [[None] * (steps * 2 + 1) for _ in range(n_stages)]
    for step in range(steps):
        for stage_id in range(n_stages):
            mb_idx = step - stage_id
            if mb_idx >= 0 and mb_idx < n_mbs:
                schedule[stage_id][step] = Task(stage_id, stage_id, mb_idx, True, False)

    for step in range(steps, steps * 2):
        for stage_id in reversed(range(n_stages)):
            mb_idx = (step - steps) - (n_stages - stage_id - 1)
            if mb_idx >= 0 and mb_idx < n_mbs:
                schedule[stage_id][step] = Task(stage_id, stage_id, mb_idx, False, False)
    for i, stage in enumerate(range(n_stages)):
        schedule[stage][-i-1] = Task(stage_id=stage, device_id=stage, mb_idx=0, is_fwd=False, upd=True)
    return schedule

# def build_1f1b_schedule(n_mbs: int, n_stages: int):
#     steps = n_mbs + n_stages - 1
#     schedule = [[None] * (steps * 2 + 1) for _ in range(n_stages)]
#     stage_mb = [[0, 0] for _ in range(n_stages)]
#     for step in range(n_stages):
#         for stage_id in range(n_stages):
#             if step >= stage_id:
#                 mb_idx = stage_mb[stage_id][0]
#                 if mb_idx >= 0 and mb_idx < n_mbs:
#                     schedule[stage_id][step] = Task(
#                         device_id=stage_id, stage_id=stage_id, 
#                         mb_idx=mb_idx, is_fwd=True, upd=False
#                     )
#                     stage_mb[stage_id][0] += 1
#     for step in range(n_stages, 2 * steps):
#         relative_step = step - n_stages
#         for stage_id in range(n_stages):
#             inv_stage = n_stages - stage_id - 1
#             if relative_step >= inv_stage:
#                 fwd_or_bwd = 1 - (relative_step + inv_stage) % 2
#                 task_type = True if fwd_or_bwd == 0 else False
#                 mb_idx = stage_mb[stage_id][fwd_or_bwd]
#                 if mb_idx >= 0 and mb_idx < n_mbs:
#                     schedule[stage_id][step] = Task(
#                         device_id=stage_id, stage_id=stage_id,
#                         mb_idx=mb_idx, is_fwd=task_type, upd=False
#                     )
#                     stage_mb[stage_id][fwd_or_bwd] += 1
#     return schedule
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
                        device_id=stage_id, stage_id=stage_id, 
                        mb_idx=mb_idx, is_fwd=True, upd=False
                    )
                    stage_mb[stage_id][0] += 1
    for step in range(n_stages, 2 * steps):
        relative_step = step - n_stages
        for stage_id in range(n_stages):
            inv_stage = n_stages - stage_id - 1
            if relative_step >= inv_stage:
                fwd_or_bwd = 1 - (relative_step + inv_stage) % 2
                task_type = True if fwd_or_bwd == 0 else False
                mb_idx = stage_mb[stage_id][fwd_or_bwd]
                if mb_idx >= 0 and mb_idx < n_mbs:
                    schedule[stage_id][step] = Task(
                        device_id=stage_id, stage_id=stage_id,
                        mb_idx=mb_idx, is_fwd=task_type, upd=False
                    )
                    stage_mb[stage_id][fwd_or_bwd] += 1
    for i, stage in enumerate(range(n_stages)):
        schedule[stage][-i-1] = Task(stage_id=stage, device_id=stage, mb_idx=n_mbs-1, is_fwd=False, upd=True)
    return schedule

no_pp_schedule = [
    [
        Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=False, upd=True),
    ]
]

pp2_interleaved_1f1b_grid_schedule = [
    [
        Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=2, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=2, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=3, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=2, mb_idx=0, is_fwd=False, upd=False),
        None,
        Task(device_id=0, stage_id=2, mb_idx=1, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=2, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=2, mb_idx=3, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=1, is_fwd=False, upd=False),
        None,
        Task(device_id=0, stage_id=2, mb_idx=2, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=2, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=2, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=False, upd=True),
    ],
    [
        None,
        Task(device_id=1, stage_id=1, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=3, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=3, mb_idx=0, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=3, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=3, mb_idx=1, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=0, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=3, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=1, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=3, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=3, mb_idx=2, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=3, mb_idx=3, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=3, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=2, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=0, is_fwd=False, upd=True),
        None,
    ],
]

pp4_interleaved_1f1b_grid_schedule = [
    [
        Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=3, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=4, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=4, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=4, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=0, stage_id=4, mb_idx=3, is_fwd=True, upd=False),
        None,
        None,
        None,
        Task(device_id=0, stage_id=4, mb_idx=0, is_fwd=False, upd=False),
        None,
        Task(device_id=0, stage_id=4, mb_idx=1, is_fwd=False, upd=False),
        None,
        Task(device_id=0, stage_id=4, mb_idx=2, is_fwd=False, upd=False),
        None,
        Task(device_id=0, stage_id=4, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=1, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=2, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=False, upd=True),
    ],
    [
        None,
        Task(device_id=1, stage_id=1, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=3, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=5, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=5, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=5, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=1, stage_id=5, mb_idx=3, is_fwd=True, upd=False),
        None,
        Task(device_id=1, stage_id=5, mb_idx=0, is_fwd=False, upd=False),
        None,
        Task(device_id=1, stage_id=5, mb_idx=1, is_fwd=False, upd=False),
        None,
        Task(device_id=1, stage_id=5, mb_idx=2, is_fwd=False, upd=False),
        None,
        Task(device_id=1, stage_id=5, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=0, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=1, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=2, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=1, stage_id=1, mb_idx=0, is_fwd=False, upd=True),
        None,
    ],
    [
        None,
        None,
        Task(device_id=2, stage_id=2, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=2, stage_id=2, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=2, stage_id=2, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=2, stage_id=2, mb_idx=3, is_fwd=True, upd=False),
        Task(device_id=2, stage_id=6, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=2, stage_id=6, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=2, stage_id=6, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=2, stage_id=6, mb_idx=0, is_fwd=False, upd=False),
        Task(device_id=2, stage_id=6, mb_idx=3, is_fwd=True, upd=False),
        Task(device_id=2, stage_id=6, mb_idx=1, is_fwd=False, upd=False),
        None,
        Task(device_id=2, stage_id=6, mb_idx=2, is_fwd=False, upd=False),
        None,
        Task(device_id=2, stage_id=6, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=2, stage_id=2, mb_idx=0, is_fwd=False, upd=False),
        Task(device_id=2, stage_id=2, mb_idx=1, is_fwd=False, upd=False),
        Task(device_id=2, stage_id=2, mb_idx=2, is_fwd=False, upd=False),
        Task(device_id=2, stage_id=2, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=2, stage_id=2, mb_idx=0, is_fwd=False, upd=True),
        None,
        None,
    ],
    [
        None,
        None,
        None,
        Task(device_id=3, stage_id=3, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=3, stage_id=3, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=3, stage_id=3, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=3, stage_id=3, mb_idx=3, is_fwd=True, upd=False),
        Task(device_id=3, stage_id=7, mb_idx=0, is_fwd=True, upd=False),
        Task(device_id=3, stage_id=7, mb_idx=0, is_fwd=False, upd=False),
        Task(device_id=3, stage_id=7, mb_idx=1, is_fwd=True, upd=False),
        Task(device_id=3, stage_id=7, mb_idx=1, is_fwd=False, upd=False),
        Task(device_id=3, stage_id=7, mb_idx=2, is_fwd=True, upd=False),
        Task(device_id=3, stage_id=7, mb_idx=2, is_fwd=False, upd=False),
        Task(device_id=3, stage_id=7, mb_idx=3, is_fwd=True, upd=False),
        Task(device_id=3, stage_id=7, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=3, stage_id=3, mb_idx=0, is_fwd=False, upd=False),
        Task(device_id=3, stage_id=3, mb_idx=1, is_fwd=False, upd=False),
        Task(device_id=3, stage_id=3, mb_idx=2, is_fwd=False, upd=False),
        Task(device_id=3, stage_id=3, mb_idx=3, is_fwd=False, upd=False),
        Task(device_id=3, stage_id=3, mb_idx=0, is_fwd=False, upd=True),
        None,
        None,
        None,
    ],
]