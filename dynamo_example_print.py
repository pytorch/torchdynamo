import os
import torch
import torch.fx as fx
import torch.nn as nn
import torch.optim as optim

from typing import List

import torchdynamo
from torchdynamo.optimizations import BACKENDS

from torch.profiler import profile, record_function, ProfilerActivity

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10000)
        self.net2 = nn.Linear(10000, 10000)
        self.net3 = nn.Linear(10000, 10000)
        self.relu = nn.ReLU()
        self.net4 = nn.Linear(10000, 5)

    def forward(self, x):
        output1 = self.relu(self.net1(x))
        print()
        output2 = self.relu(self.net2(output1))
        print()
        output3 = self.relu(self.net3(output2))
        print()
        return self.net4(output3)

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()

    compiled = BACKENDS["aot_autograd"](gm, example_inputs)
    if compiled is not None:
        print("aot compiled")
        return compiled

    return gm.forward  # return a python callable

def hook(grad):
  print("gradient hook fired")
  grad + 1
  return grad

def demo_basic():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      with torchdynamo.optimize(my_compiler):
        device = "cuda"

        # create model and move it to the device with id rank
        model = ToyModel().to(device)
        for parameter in model.parameters():
          parameter.register_hook(hook)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        for i in range(1):
            optimizer.zero_grad()
            outputs = model(torch.randn(20, 10).to(device))
            labels = torch.randn(20, 5).to(device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"{os.getpid()}: iteration {i}, loss {loss}")

    prof.export_chrome_trace("trace.json")

if __name__ == "__main__":
    demo_basic()
