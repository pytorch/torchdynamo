#!/usr/bin/env pytest
import inspect
import unittest

import torch
from torch import sub
from torch.nn import functional as F

from torchdynamo import eval_frame
from torchdynamo.symbolic_convert import convert_frame_assert
from torchdynamo.testing import same

d = torch.ones(10, 10)
e = torch.nn.Linear(10, 10)


def add(a, b):
    return a + b


def constant(a, b, c):
    return a - b * c + 1.0


def globalvar(a, b):
    return a - b + d


def globalfn(a, b):
    return sub(a, b)


def viatorch(a, b):
    return torch.sub(a, b)


def viamethod(a, b):
    return a.sub(b)


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


def make_test(fn):
    def test_fn(self):
        torch.fx.symbolic_trace(fn).graph.print_tabular()
        nargs = len(inspect.signature(fn).parameters) - int(isinstance(fn, torch.nn.Module))
        args = [torch.randn(10, 10) for _ in range(nargs)]
        correct = fn(*args)
        with eval_frame.context(convert_frame_assert):
            val1 = fn(*args)
            val2 = fn(*args)
        self.assertTrue(same(val1, correct))
        self.assertTrue(same(val2, correct))

    return test_fn


class SymblicConversionTests(unittest.TestCase):
    test_add = make_test(add)
    test_constant = make_test(constant)
    test_globalvar = make_test(globalvar)
    test_globalfn = make_test(globalfn)
    test_viatorch = make_test(viatorch)
    test_viamethod = make_test(viamethod)
    test_mymodule1 = make_test(MyModule())
    test_mymodule2 = make_test(MyModule())
