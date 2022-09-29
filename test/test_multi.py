import torch

import torchdynamo


@torchdynamo.optimize()
def f(a):
    return a.softmax(dim=1) + a.sum()


a = torch.rand([100, 100])
f(a)
