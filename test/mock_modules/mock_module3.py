import torch


def method1(x, y):
    z = torch.ones(1, 1)
    x.append(y)
    return x
