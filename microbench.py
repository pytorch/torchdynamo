import torch
import torchinductor
from torch import fx
from torchinductor.lowering import GraphLowering

import torchdynamo
from torchdynamo.optimizations.python_key import python_key_normalize


class MyModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
        )

    def forward(self, input):
        # return (self.model(input) + 1,)
        return (self.model(input),)


class MyModel2(torch.nn.Module):
    def forward(self, x):
        # return x / (torch.abs(x) + 1.0),
        return (x + x,)


model = MyModel2().eval()

inputs = (torch.rand(32, 64, 1024).transpose(0, 1),)

gm, wrap = python_key_normalize(fx.symbolic_trace(model), inputs)

gm.graph.print_tabular()

graph = GraphLowering(gm)
wrap(graph.run)(*inputs)

print(graph.cpp())
