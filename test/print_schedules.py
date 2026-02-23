from .schedule_helpers import print_schedule, build_1f1b_schedule, build_gpipe_schedule

schedule = build_1f1b_schedule(4, 2)
print("1F1B Schedule")
print_schedule(schedule)

schedule = build_gpipe_schedule(4, 2)
print("GPipe Schedule")
print_schedule(schedule)

schedule = build_1f1b_schedule(8, 4)
print("1F1B Schedule")
print_schedule(schedule)

schedule = build_gpipe_schedule(8, 4)
print("GPipe Schedule")
print_schedule(schedule)