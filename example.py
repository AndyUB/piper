import torch

class Model(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(10, 10)

   def forward(self, x):
      return self.linear(x)

def custom_backend(gm: torch.fx.GraphModule, *args, **kwargs):
    print(gm.graph)
    return gm

torch._dynamo.config.compiled_autograd = True
@torch.compile(backend=custom_backend)
def train(model, x):
   loss = model(x).sum()
   loss.backward()

def main():
    model = Model()
    x = torch.randn(10)
    train(model, x)

if __name__ == "__main__":
    main()