import os
import torch
import torch.fx as fx
import torch.nn as nn
import torch.optim as optim

from typing import List

import torchdynamo
from torchdynamo.optimizations import BACKENDS

from torch.profiler import profile, ProfilerActivity

from functorch.compile import aot_module, clear_compile_cache

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
        output2 = self.relu(self.net2(output1))
        output3 = self.relu(self.net3(output2))
        return self.net4(output3)

def hook(grad):
  print("gradient hook fired")
  return grad + 1

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    # fx_module.graph.print_tabular()
    return fx_module

# A basic AOT example to demonstrate that gradient hooks are all
# fired after the compiled aot module.
def demo_basic():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        device = "cuda"

        # create model and move it to the device with id rank
        model = ToyModel().to(device)
        for parameter in model.parameters():
          parameter.register_hook(hook)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        aot_print_module = aot_module(model, fw_compiler=compiler_fn, bw_compiler=compiler_fn)

        for i in range(1):
            optimizer.zero_grad()
            outputs = aot_print_module(torch.randn(20, 10).to(device))
            labels = torch.randn(20, 5).to(device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"{os.getpid()}: iteration {i}, loss {loss}")

        clear_compile_cache()

    prof.export_chrome_trace("aot_1.json")

if __name__ == "__main__":
    demo_basic()
