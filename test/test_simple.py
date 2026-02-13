import torch
import ray

from src.piper_coordinator import PiperProgramCoordinator
from src.piper_compile import piper_setup
from src.piper_exec import piper_exec
from src.piper_utils import piper_metadata

from .schedule_helpers import build_1f1b_schedule, print_schedule

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)
        self.freqs_cis = torch.randn(10, 10)

    def forward(self, x):
        with torch.fx.traceback.annotate({"stage": 0}):
            freqs_cis = self.freqs_cis
            x = self.layer1(x) + freqs_cis
        with torch.fx.traceback.annotate({"stage": 1}):
            x = self.layer2(x) + freqs_cis
        return x

def main():
    stages = 2
    mbs = 1
    x = torch.randn(10)
    y = torch.randn((10, 10))

    schedule = build_1f1b_schedule(mbs, stages)
    print_schedule(schedule)

    model = piper_setup(SimpleModel, [], torch.optim.Adam, [x], schedule)

    actors = piper_metadata.actors
    ray.get(actors[0].load_input.remote([x]))
    ray.get(actors[len(actors)-1].load_labels.remote(y))

    piper_exec(model, schedule, [x], y, torch.nn.MSELoss())

    ray.timeline(f"out/simple.json")

if __name__ == "__main__":
    ray.init(include_dashboard=False, log_to_driver=True, namespace="llama", _temp_dir="/m-coriander/coriander/mfris/tmp/ray")
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=2, dp_degree=1)
    handles = piper_coordinator.run_program.remote(main)
    ray.get(handles)
    ray.shutdown()