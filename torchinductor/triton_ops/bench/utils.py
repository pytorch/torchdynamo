import torch
import math


def rounded_linspace(low, high, steps, div):
    ret = torch.linspace(low, high, steps)
    ret = (ret.int() + div - 1) // div * div
    ret = torch.unique(ret)
    return list(map(int, ret))


def powspace(start, stop, pow, step):
    start = math.log(start, pow)
    stop = math.log(stop, pow)
    ret = torch.pow(pow, torch.arange(start, stop, step))
    return list(map(int, ret))
