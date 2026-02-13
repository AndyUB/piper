from test.models.llama import Transformer, LLAMA_1B

from torch._dynamo.backends.debugging import eager

model = Transformer(LLAMA_1B, 1024)
dummy = torch.compile(model, backend=eager, fullgraph=True)
dummy(torch.randint(0, 1024, (1, 1024)))