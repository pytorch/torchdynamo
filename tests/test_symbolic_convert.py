#!/usr/bin/env pytest
import inspect
import unittest

import torch
from torch import sub
from torch.nn import functional as F

import torchdynamo
from torchdynamo import eval_frame
from torchdynamo.symbolic_convert import convert_frame_assert
from torchdynamo.testing import same

torchdynamo.symbolic_convert.DEBUG = True

d = torch.ones(10, 10)
e = torch.nn.Linear(10, 10)


def add(a, b):
    return a + b


def constant1(a, b, c):
    return a - b * c + 1.0


def constant2(a, b, c):
    return a - b * c + 1


def constant3(a, b):
    return a - b + (1.0 + 2)


def globalvar(a, b):
    return a - b + d


def globalfn(a, b):
    return sub(a, b)


def viatorch(a, b):
    return torch.sub(a, b)


def viamethod(a, b):
    return a.sub(b)


def indirect1(a, b):
    t = a.sub
    return t(b)


def indirect2(a, b):
    t = a.sub
    args = (b,)
    return t(*args)


def indirect3(a, b):
    t = a.sub
    args = (b,)
    kwargs = {}
    return t(*args, **kwargs)


def globalmodule(x):
    return e(x)


def method_call(a, b, c):
    return constant3(a, b) * c


def tuple1(a, b):
    args = (a, b)
    return sub(*args)


def tuple2(a, b):
    args = [a, b]
    return sub(*args)


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
        args1 = [torch.randn(10, 10) for _ in range(nargs)]
        args2 = [torch.randn(10, 10) for _ in range(nargs)]
        correct1 = fn(*args1)
        correct2 = fn(*args2)
        with eval_frame.context(convert_frame_assert):
            val1a = fn(*args1)
            val2a = fn(*args2)
            val1b = fn(*args1)
            val2b = fn(*args2)
        self.assertTrue(same(val1a, correct1))
        self.assertTrue(same(val1b, correct1))
        self.assertTrue(same(val2a, correct2))
        self.assertTrue(same(val2b, correct2))

    return test_fn


class SymblicConversionTests(unittest.TestCase):
    test_add = make_test(add)
    test_constant1 = make_test(constant1)
    test_constant2 = make_test(constant2)
    test_constant3 = make_test(constant3)
    test_globalvar = make_test(globalvar)
    test_globalfn = make_test(globalfn)
    test_viatorch = make_test(viatorch)
    test_viamethod = make_test(viamethod)
    test_mymodule1 = make_test(MyModule())
    test_mymodule2 = make_test(MyModule())
    # test_globalmodule = make_test(globalmodule)
    test_indirect1 = make_test(indirect1)
    test_indirect2 = make_test(indirect2)
    test_indirect3 = make_test(indirect3)
    # test_methodcall = make_test(methodcall)
    test_tuple1 = make_test(tuple1)
    test_tuple2 = make_test(tuple2)
