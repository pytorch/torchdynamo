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

class ToyModel1(nn.Module):
    def __init__(self):
        super(ToyModel1, self).__init__()
        self.net1 = nn.Linear(10, 10000)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net1(x))

class ToyModel2(nn.Module):
    def __init__(self):
        super(ToyModel2, self).__init__()
        self.net2 = nn.Linear(10000, 10000)
        self.relu = nn.ReLU()

    def forward(self, x):
        return  self.relu(self.net2(x))

class ToyModel3(nn.Module):
    def __init__(self):
        super(ToyModel3, self).__init__()
        self.net3 = nn.Linear(10000, 10000)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net3(x))

class ToyModel4(nn.Module):
    def __init__(self):
        super(ToyModel4, self).__init__()
        self.net4 = nn.Linear(10000, 5)

    def forward(self, x):
        return self.net4(x)

def hook(grad):
  print("gradient hook fired")
  return grad + 1

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    # fx_module.graph.print_tabular()
    return fx_module

# An AOT example to demonstrate that gradient hooks can be
# fired in between the chained compiled aot module.
def demo_basic():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        device = "cuda"

        # create model and move it to the device with id rank
        models = []
        models.append(ToyModel1().to(device))
        models.append(ToyModel2().to(device))
        models.append(ToyModel3().to(device))
        models.append(ToyModel4().to(device))

        for model in models:
            for parameter in model.parameters():
                parameter.register_hook(hook)

        loss_fn = nn.MSELoss()
        aot_print_modules = []
        for model in models:
            aot_print_modules.append(aot_module(model, fw_compiler=compiler_fn, bw_compiler=compiler_fn))

        for i in range(1):
            outputs = torch.randn(20, 10).to(device)
            for aot_print_module in aot_print_modules:
                outputs = aot_print_module(outputs)
            labels = torch.randn(20, 5).to(device)
            loss = loss_fn(outputs, labels)
            loss.backward()

            print(f"{os.getpid()}: iteration {i}, loss {loss}")

        clear_compile_cache()

    prof.export_chrome_trace("aot_2.json")

if __name__ == "__main__":
    demo_basic()
