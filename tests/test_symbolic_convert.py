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


def methodcall1(a, b, c):
    return constant3(a, b) * c


def methodcall2(a, b):
    return constant3(a=b, b=a) + 1


def methodcall3(a, b):
    return constant3(a, b=1.0) + b


def tuple1(a, b):
    args = (a, b)
    return sub(*args)


def tuple2(a, b):
    args = [a, b]
    return sub(*args)


def listarg1(a, b):
    return torch.cat([a, b])


def listarg2(a, b):
    return torch.cat((a, b), dim=0)


def listarg3(a, b):
    kwargs = {"tensors": (a, b), "dim": 0}
    return torch.cat(**kwargs)


def listarg4(a, b):
    return torch.cat(tensors=[a, b], dim=0)


def listarg5(a, b):
    args = [(a, b)]
    kwargs = {"dim": 0}
    return torch.cat(*args, **kwargs)


def slice1(a):
    return a[5]


def slice2(a):
    return a[:5]


def slice3(a):
    return a[5:]


def slice4(a):
    return a[2:5]


def slice5(a):
    return a[::2]


def slice6(a):
    return torch.unsqueeze(a, 0)[:, 2:]


def unpack1(a):
    a, b = a[:5], a[5:]
    return a - b


def unpack2(a):
    l = [a[:5], a[5:]]
    a, b = l
    return a - b


def unpack3(a):
    l = (a[:5], a[5:])
    a, b = l
    return a - b


def inplace1(a, b):
    o = torch.empty((10,))
    o.copy_(a)
    o -= b
    return o


def fn_with_self_set(a, b):
    # avg_pool2d is an odd one with __self__ set
    return F.avg_pool2d(torch.unsqueeze(a, 0) * torch.unsqueeze(b, 1),
                        kernel_size=2, padding=1)


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


class SubmoduleExample(torch.nn.Module):
    def __init__(self):
        super(SubmoduleExample, self).__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x * self.scale


class IsTrainingCheck(torch.nn.Module):
    def __init__(self):
        super(IsTrainingCheck, self).__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.train(True)

    def forward(self, x):
        if self.training:
            mod = self.linear1
        else:
            mod = self.linear2
        return F.relu(mod(x))


class IsEvalCheck(IsTrainingCheck):
    def __init__(self):
        super(IsEvalCheck, self).__init__()
        self.train(False)


class ModuleMethodCall(torch.nn.Module):
    def __init__(self):
        super(ModuleMethodCall, self).__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        return x * self.scale

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        x2 = self.call_and_scale(self.layer2, x)
        return x1 + x2


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


class SymbolicConversionTests(unittest.TestCase):
    def test_boolarg(self):
        def boolarg(aa, bb, flag):
            if flag:
                return aa - bb
            else:
                return bb - aa

        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        correct1 = boolarg(a, b, True)
        correct2 = boolarg(a, b, False)
        correct3 = boolarg(a, b, None)
        with eval_frame.context(convert_frame_assert):
            val1 = boolarg(a, b, True)
            val2 = boolarg(a, b, False)
            val3 = boolarg(a, b, None)
        self.assertTrue(same(val1, correct1))
        self.assertTrue(same(val2, correct2))
        self.assertTrue(same(val3, correct3))

    @unittest.skip("not implemented yet")
    def test_callpacked(self):
        def call_packed(args):
            a, b, c = args
            return a - b * c

        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        correct = call_packed([a, b, c])
        with eval_frame.context(convert_frame_assert):
            val1 = call_packed([a, b, c])
            val2 = call_packed((a, b, c))
            val3 = call_packed([a, b, c])
        self.assertTrue(same(val1, correct))
        self.assertTrue(same(val2, correct))
        self.assertTrue(same(val3, correct))

    test_add = make_test(add)
    test_constant1 = make_test(constant1)
    test_constant2 = make_test(constant2)
    test_constant3 = make_test(constant3)
    test_globalfn = make_test(globalfn)
    test_viatorch = make_test(viatorch)
    test_viamethod = make_test(viamethod)
    test_indirect1 = make_test(indirect1)
    test_indirect2 = make_test(indirect2)
    test_indirect3 = make_test(indirect3)
    test_tuple1 = make_test(tuple1)
    test_tuple2 = make_test(tuple2)
    test_slice1 = make_test(slice1)
    test_slice2 = make_test(slice2)
    test_slice3 = make_test(slice3)
    test_slice4 = make_test(slice4)
    test_slice5 = make_test(slice5)
    test_slice6 = make_test(slice6)
    test_listarg1 = make_test(listarg1)
    test_listarg2 = make_test(listarg2)
    test_listarg3 = make_test(listarg3)
    test_listarg4 = make_test(listarg4)
    test_listarg5 = make_test(listarg5)
    test_unpack1 = make_test(unpack1)
    test_unpack2 = make_test(unpack2)
    test_unpack3 = make_test(unpack3)
    test_fn_with_self_set = make_test(fn_with_self_set)
    test_methodcall1 = make_test(methodcall1)
    test_methodcall2 = make_test(methodcall2)
    test_methodcall3 = make_test(methodcall3)

    test_basicmodule1 = make_test(BasicModule())
    test_basicmodule2 = make_test(BasicModule())

    # TODO(jansel): these ones aren't implemented yet
    # test_inplace1 = make_test(inplace1)
    # test_submodules1 = make_test(SubmoduleExample())
    # test_submodules2 = make_test(SubmoduleExample())
    # test_istraining1 = make_test(IsTrainingCheck())
    # test_istraining2 = make_test(IsTrainingCheck())
    # test_iseval1 = make_test(IsEvalCheck())
    # test_iseval2 = make_test(IsEvalCheck())
    # test_modulemethod1 = make_test(ModuleMethodCall())
    # test_modulemethod2 = make_test(ModuleMethodCall())
    # test_globalmodule = make_test(globalmodule)

    # TODO(jansel): need to debug a crash on this one
    # test_globalvar = make_test(globalvar)

    # TODO(jansel): we should make sure to expand nn.Sequential
