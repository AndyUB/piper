import torch
from torch import nn
from src.piper import piper, distributed_stage

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        distributed_stage(0)
        x = self.relu(self.fc1(x))
        distributed_stage(1)
        x = self.relu(self.fc2(x))
        distributed_stage(2)
        x = self.fc3(x)
        return x

model = torch.compile(MyModel(), backend=piper)

x = torch.randn(10, 1024)
out = model(x).get()
print(out)

