import torch
import ray
import time
from torch import nn
from src.piper import piper, distributed_stage
from collections.abc import Iterable
import threading

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x, dynamo_mb):
        distributed_stage(0, mb=dynamo_mb, optim=torch.optim.Adam)
        x = self.relu(self.fc1(x))
        distributed_stage(1, mb=dynamo_mb, optim=torch.optim.Adam)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from src.piper_utils import piper_tls

def run_model(model, x, dynamo_mb):
    out = model(x, dynamo_mb).get()

def main():

    model = torch.compile(MyModel().to('cuda'), backend=piper)
    x = torch.randn(10, 1024).to('cuda')
    y = torch.randn(10, 10).to('cuda')

    # Compile the model

    mb_idx = 0
    num_stages = 2
    piper_tls.events[mb_idx] = [threading.Event() for _ in range(num_stages)]
    for event in piper_tls.events[mb_idx]:
        event.set()
    run_model(model, x, mb_idx)

    print(f"Compiled.")

    threads = []
    for mb_idx in range(2):
        piper_tls.events[mb_idx] = [threading.Event() for _ in range(num_stages)]
        thread = threading.Thread(target=run_model, args=(model, x, mb_idx))
        thread.start()
        threads.append(thread)

    for stage_id in range(num_stages):
        for mb_idx in range(2):
            piper_tls.events[mb_idx][stage_id].set()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()