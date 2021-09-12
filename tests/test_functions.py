#!/usr/bin/env pytest
import inspect
import unittest

import torch
from torch import sub
from torch.nn import functional as F

import torchdynamo
from torchdynamo import eval_frame
from torchdynamo.symbolic_convert import convert_frame_assert, dummy_fx_compile
from torchdynamo.testing import same

torchdynamo.symbolic_convert.DEBUG = True

d = torch.ones(10, 10)
e = torch.nn.Linear(10, 10)


def constant3(a, b):
    return a - b + (1.0 + 2)


def make_test(fn):
    nargs = len(inspect.signature(fn).parameters)

    def test_fn(self):
        return torchdynamo.testing.standard_test(self, fn=fn, nargs=nargs)

    return test_fn


class FunctionTests(unittest.TestCase):
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
        with eval_frame.optimize(convert_frame_assert(dummy_fx_compile)):
            val1 = boolarg(a, b, True)
            val2 = boolarg(a, b, False)
            val3 = boolarg(a, b, None)
        self.assertTrue(same(val1, correct1))
        self.assertTrue(same(val2, correct2))
        self.assertTrue(same(val3, correct3))

    @unittest.skip("need to debug this")
    def test_callpacked(self):
        def call_packed(args):
            a, b, c = args
            return a - b * c

        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        correct = call_packed([a, b, c])
        with eval_frame.optimize(convert_frame_assert(dummy_fx_compile)):
            val1 = call_packed([a, b, c])
            val2 = call_packed((a, b, c))
        self.assertTrue(same(val1, correct))
        self.assertTrue(same(val2, correct))

    @make_test
    def test_add(a, b):
        return a + b

    @make_test
    def test_constant1(a, b, c):
        return a - b * c + 1.0

    @make_test
    def test_constant2(a, b, c):
        return a - b * c + 1

    @make_test
    def test_globalfn(a, b):
        return sub(a, b)

    @make_test
    def test_viatorch(a, b):
        return torch.sub(a, b)

    @make_test
    def test_viamethod(a, b):
        return a.sub(b)

    @make_test
    def test_indirect1(a, b):
        t = a.sub
        return t(b)

    @make_test
    def test_indirect2(a, b):
        t = a.sub
        args = (b,)
        return t(*args)

    @make_test
    def test_indirect3(a, b):
        t = a.sub
        args = (b,)
        kwargs = {}
        return t(*args, **kwargs)

    @make_test
    def test_methodcall1(a, b, c):
        return constant3(a, b) * c

    @make_test
    def test_methodcall2(a, b):
        return constant3(a=b, b=a) + 1

    @make_test
    def test_methodcall3(a, b):
        return constant3(a, b=1.0) + b

    @make_test
    def test_tuple1(a, b):
        args = (a, b)
        return sub(*args)

    @make_test
    def test_tuple2(a, b):
        args = [a, b]
        return sub(*args)

    @make_test
    def test_listarg1(a, b):
        return torch.cat([a, b])

    @make_test
    def test_listarg2(a, b):
        return torch.cat((a, b), dim=0)

    @make_test
    def test_listarg3(a, b):
        kwargs = {"tensors": (a, b), "dim": 0}
        return torch.cat(**kwargs)

    @make_test
    def test_listarg4(a, b):
        return torch.cat(tensors=[a, b], dim=0)

    @make_test
    def test_listarg5(a, b):
        args = [(a, b)]
        kwargs = {"dim": 0}
        return torch.cat(*args, **kwargs)

    @make_test
    def test_slice1(a):
        return a[5]

    @make_test
    def test_slice2(a):
        return a[:5]

    @make_test
    def test_slice3(a):
        return a[5:]

    @make_test
    def test_slice4(a):
        return a[2:5]

    @make_test
    def test_slice5(a):
        return a[::2]

    @make_test
    def test_slice6(a):
        return torch.unsqueeze(a, 0)[:, 2:]

    @make_test
    def test_unpack1(a):
        a, b = a[:5], a[5:]
        return a - b

    @make_test
    def test_unpack2(a):
        l = [a[:5], a[5:]]
        a, b = l
        return a - b

    @make_test
    def test_unpack3(a):
        l = (a[:5], a[5:])
        a, b = l
        return a - b

    @make_test
    def test_fn_with_self_set(a, b):
        # avg_pool2d is an odd one with __self__ set
        return F.avg_pool2d(torch.unsqueeze(a, 0) * torch.unsqueeze(b, 1),
                            kernel_size=2, padding=1)

    def test_inplace(self):
        def inplace1(a, b):
            o = torch.empty((10, 10))
            o.copy_(a)
            o -= b
            return o

        torchdynamo.testing.standard_test(self, inplace1, 2, expected_ops=3)

    def test_unpack4(self):
        def unpack4(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.size()
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torchdynamo.testing.standard_test(self, unpack4, 2, expected_ops=8)

    def test_unpack5(self):
        def unpack5(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.shape
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torchdynamo.testing.standard_test(self, unpack5, 2, expected_ops=8)

    def test_matmul1(self):
        def matmul_op1(a, b):
            return a @ b

        # TODO(jansel): FX doesn't support this, should add upstream support
        torchdynamo.testing.standard_test(self, matmul_op1, 2, expected_ops=1)

    @unittest.skip("buggy")
    @make_test
    def test_globalvar(a, b):
        return a - b + d

    @unittest.skip("not implemented")
    @make_test
    def test_globalmodule(x):
        return e(x)
