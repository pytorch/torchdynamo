import itertools
import os
from typing import List

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function

import torchdynamo
from torchdynamo.optimizations import BACKENDS


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10000)
        self.net2 = nn.Linear(10000, 10000)
        self.net3 = nn.Linear(10000, 10000)
        self.relu = nn.ReLU()
        self.net4 = nn.Linear(10000, 5)

    @torchdynamo.optimize("aot_print")
    def forward(self, x):
        return self.net4(
            self.relu(self.net3(self.relu(self.net2(self.relu(self.net1(x))))))
        )


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank], bucket_cap_mb=400)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def setup_torchbench():
    import os
    import sys
    for torchbench_dir in (
        "./torchbenchmark",
        "../torchbenchmark",
        "../torchbench",
        "../benchmark",
        "../../torchbenchmark",
        "../../torchbench",
        "../../benchmark",
    ):
        if os.path.exists(torchbench_dir):
            break

    assert os.path.exists(torchbench_dir), "../../torchbenchmark does not exist"
    original_dir = os.path.abspath(os.getcwd())
    torchbench_dir = os.path.abspath(torchbench_dir)

    os.chdir(torchbench_dir)
    sys.path.append(torchbench_dir)



def hf_bert(rank, world_size):
    print(f"Running hf_bert on rank {rank}.")
    setup(rank, world_size)

    setup_torchbench() 
    from torchbenchmark.models.hf_Bert import Model
    model_container = Model('train', 'cuda')
    model, (inputs, ) = model_container.get_module()
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], bucket_cap_mb=400)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    @torchdynamo.optimize("aot_nvfuser")
    def run_model():
        outputs = ddp_model(inputs)
        return outputs
    outputs = run_model()
    labels = torch.randn(outputs.logits.shape).to(rank)
    loss_fn(outputs.logits, labels).backward()
    optimizer.step()
    cleanup()


# torchdynamo.config.trace = True
torchdynamo.config.debug = False
torchdynamo.config.optimize_ddp = True
torchdynamo.config.debug_optimize_ddp = True


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    # run_demo(demo_basic, 1)
    # run_demo(hf_bert, 1)
    hf_bert(0, 1)
