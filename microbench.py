import torch
import torchinductor
from torch import fx
from torchinductor.codecache import PyCodeCache
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
    def forward(self, x, y):
        # return x / (torch.abs(x) + 1.0),
        return (x + y,)


def test_model(model, example_inputs):
    print("\n\nBEGIN\n")
    gm, wrap = python_key_normalize(fx.symbolic_trace(model), example_inputs)
    gm.graph.print_tabular()
    print()

    graph = GraphLowering(gm)
    wrap(graph.run)(*example_inputs)

    code = graph.codegen()
    print(code)
    (actual,) = PyCodeCache.load(code).call(*example_inputs)
    (correct,) = model(*example_inputs)
    torch.testing.assert_close(actual, correct)
    print("correct!")


mod1 = MyModel2().eval()
inputs1 = (torch.rand(10, 10), torch.rand(10, 10))
inputs2 = (torch.rand(10, 10, device="cuda"), torch.rand(10, 10, device="cuda"))
test_model(mod1, inputs1)
test_model(mod1, inputs2)
