import itertools
import os
import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
from typing import List

import torchdynamo
from torch.nn.parallel import DistributedDataParallel as DDP
from torchdynamo.optimizations import BACKENDS

from torch.profiler import profile, record_function, ProfilerActivity

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10000)
        self.net2 = nn.Linear(10000, 10000)
        self.net3 = nn.Linear(10000, 10000)
        self.relu = nn.ReLU()
        self.net4 = nn.Linear(10000, 5)

    def forward(self, x):
        return self.net4(self.relu(self.net3(self.relu(self.net2(self.relu(self.net1(x)))))))


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    with torchdynamo.optimize(my_compiler):
      outputs = ddp_model(torch.randn(20, 10))
      labels = torch.randn(20, 5).to(rank)
      loss_fn(outputs, labels).backward()
      optimizer.step()

    cleanup()

# torchdynamo.config.trace = True
# torchdynamo.config.debug = True

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    # prof.export_chrome_trace("new_trace.json")

if __name__ == "__main__":
  run_demo(demo_basic, 1)