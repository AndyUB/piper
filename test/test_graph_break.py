import torch
import torch.nn as nn
import ray
import time

from src.piper_compile import piper_setup
from src.piper_exec import piper_exec
from src.piper import distributed_stage
from src.piper_coordinator import PiperProgramCoordinator

from .schedule_helpers import build_1f1b_schedule, print_schedule

def main():
    class TwoLayerNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.layer3 = nn.Linear(output_dim, output_dim)

        def forward(self, x):
            distributed_stage(0)
            x = self.layer1(x)
            print(f"graph break: {x.data} {x + 1} {x[0]}")
            x = self.layer2(x)
            
            distributed_stage(1)
            x = self.layer3(x)
            return x

    x = torch.randn(2)
    y = torch.randn(2)

    model = piper_setup(TwoLayerNN, (2, 2, 2), torch.optim.Adam, [x], num_stages=2, num_devices=2)

    return

    num_mbs = 1
    num_stages = 2
    schedule = build_1f1b_schedule(num_mbs, num_stages)
    print_schedule(schedule)
    loss_fn = torch.nn.CrossEntropyLoss()

    from src.piper_utils import piper_metadata
    stage1_weights = ray.get(piper_metadata.actors[0].get_weights.remote(0))
    stage2_weights = ray.get(piper_metadata.actors[1].get_weights.remote(1))
    print("stage 1 weights:", stage1_weights)
    print("stage 2 weights:", stage2_weights)

    piper_exec(model, schedule, [x], y, loss_fn, num_mbs, num_stages)

    stage1_weights = ray.get(piper_metadata.actors[0].get_weights.remote(0))
    stage2_weights = ray.get(piper_metadata.actors[1].get_weights.remote(1))
    print("stage 1 weights:", stage1_weights)
    print("stage 2 weights:", stage2_weights)

if __name__ == "__main__":
    ray.init(include_dashboard=False, log_to_driver=True, namespace="llama")
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=2, dp_degree=1, world_size=2)
    ray.get(piper_coordinator.run_program.remote(main))
    time.sleep(3)
    ray.shutdown()