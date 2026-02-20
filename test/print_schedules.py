from .schedule_helpers import print_schedule, build_1f1b_schedule, build_gpipe_schedule

num_mbs = 4
num_stages = 2
schedule = build_1f1b_schedule(num_mbs, num_stages)
print("1F1B Schedule -> 1f1b_schedule.pdf")
print_schedule(schedule, path="1f1b_schedule_pp2")

schedule = build_gpipe_schedule(num_mbs, num_stages)
print("GPipe Schedule -> gpipe_schedule.pdf")
print_schedule(schedule, path="gpipe_schedule_pp2")

num_mbs = 8
num_stages = 4
schedule = build_1f1b_schedule(num_mbs, num_stages)
print("1F1B Schedule -> 1f1b_schedule.pdf")
print_schedule(schedule, path="1f1b_schedule_pp4")

schedule = build_gpipe_schedule(num_mbs, num_stages)
print("GPipe Schedule -> gpipe_schedule.pdf")
print_schedule(schedule, path="gpipe_schedule_pp4")