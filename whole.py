from typing import List

import torch
import torch.fx as fx
import torch.nn as nn
import torch.optim as optim

import torchdynamo
from torchdynamo.optimizations import BACKENDS


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

    def forward(self, x):
        return self.net4(
            self.relu(self.net3(self.relu(self.net2(self.relu(self.net1(x))))))
        )

def train_loop(inputs, labels, model, loss_fn, optimizer):
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss

def demo(nsteps=10):
    model = ToyModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    
    opt_train_loop = torchdynamo.optimize(my_compiler)(train_loop)

    for i in range(nsteps):
        labels = torch.randn(20, 5)
        inputs = torch.randn(20, 10)
        loss = opt_train_loop(inputs, labels, model, loss_fn, optimizer)
        print(loss)


if __name__ == "__main__":
    demo()
