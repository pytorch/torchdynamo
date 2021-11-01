#!/usr/bin/env pytest
import inspect
import math

import torch
from torch import sub
from torch.nn import functional as F

import torchdynamo.testing
from torchdynamo import eval_frame
from torchdynamo.convert_frame import convert_frame
from torchdynamo.convert_frame import convert_frame_assert
from torchdynamo.testing import CompileCounter
from torchdynamo.testing import same

d = torch.ones(10, 10)
e = torch.nn.Linear(10, 10)
flag = True


def constant3(a, b):
    return a - b + (1.0 + 2)


def func_with_default(a, b, some_default_arg=True):
    if some_default_arg:
        return a - b


def make_test(fn):
    nargs = len(inspect.signature(fn).parameters)

    def test_fn(self):
        return torchdynamo.testing.standard_test(self, fn=fn, nargs=nargs)

    return test_fn


class FunctionTests(torchdynamo.testing.TestCase):
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
        counter = CompileCounter()
        with eval_frame.optimize(convert_frame_assert(counter)):
            val1 = boolarg(a, b, True)
            val2 = boolarg(a, b, False)
            val3 = boolarg(a, b, None)
            val4 = boolarg(a, b, True)
        self.assertTrue(same(val1, correct1))
        self.assertTrue(same(val2, correct2))
        self.assertTrue(same(val3, correct3))
        self.assertTrue(same(val4, correct1))
        self.assertEqual(counter.frame_count, 3)

    def test_callpacked(self):
        def call_packed(args):
            a, b, c = args
            return a - b * c

        counter = CompileCounter()
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        correct = call_packed([a, b, c])
        with eval_frame.optimize(convert_frame_assert(counter)):
            val1 = call_packed([a, b, c])
            val2 = call_packed((a, b, c))
            val3 = call_packed([a, b, c])
            val4 = call_packed((a, b, c))
        self.assertTrue(same(val1, correct))
        self.assertTrue(same(val2, correct))
        self.assertTrue(same(val3, correct))
        self.assertTrue(same(val4, correct))
        self.assertEqual(counter.frame_count, 2)

    def test_raises(self):
        def fn(a, b, c, cls):
            x = a + b - c * 10
            raise cls(str(x))

        counter = CompileCounter()
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        with eval_frame.optimize(convert_frame(counter)):
            self.assertRaises(AssertionError, lambda: fn(a, b, c, AssertionError))
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)

    @make_test
    def test_add(a, b):
        return a + b

    @make_test
    def test_is_not_null(a, b):
        if a is not None and b is not None:
            return a + b

    @make_test
    def test_constant1(a, b, c):
        return a - b * c + 1.0

    @make_test
    def test_constant2(a, b, c):
        return a - b * c + 1

    @make_test
    def test_constant3(a):
        b = 1
        c = 2
        d = 3
        return b + c - d + a

    @make_test
    def test_constant4(a, b):
        c = 2
        d = 3
        if c > d:
            return a - b
        return b - a

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
        packed = [a[:5], a[5:]]
        a, b = packed
        return a - b

    @make_test
    def test_unpack3(a):
        packed = (a[:5], a[5:])
        a, b = packed
        return a - b

    @make_test
    def test_fn_with_self_set(a, b):
        # avg_pool2d is an odd one with __self__ set
        return F.avg_pool2d(
            torch.unsqueeze(a, 0) * torch.unsqueeze(b, 1), kernel_size=2, padding=1
        )

    @make_test
    def test_return_tuple1(a, b):
        return (a - b, b - a, a, b)

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

    @make_test
    def test_globalvar(a, b):
        return a - b + d

    @make_test
    def test_globalmodule(x):
        return e(x)

    @make_test
    def test_inline_with_default(a, b, c):
        return func_with_default(a, b) * c

    @make_test
    def test_inner_function(x):
        def fn(x):
            return torch.add(x, x)

        return fn(x)

    @make_test
    def test_transpose_for_scores(x):
        new_x_shape = x.size()[:-1] + (2, 5)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    @make_test
    def test_return_tuple2(x):
        return (torch.add(x, x), x)

    @make_test
    def test_load_global_bool(x):
        if flag:
            return torch.add(x, x)
        else:
            return x

    @make_test
    def test_len_tensor(x):
        z = len(x)
        return torch.add(x, z)

    @make_test
    def test_len_constant_list(x):
        z = len([1, 2, 3])
        return torch.add(x, z)

    @make_test
    def test_len_constant_dict(x):
        z = len({"foo": "bar"})
        return torch.add(x, z)

    @make_test
    def test_len_constant_misc_iterables(x):
        a = len((1, 2, 3))
        b = len("test str")
        c = a + b
        return torch.add(x, c)

    def test_builtin_isinstance(self):
        def fn(x):
            t = torch.arange(1, 3)
            a = isinstance(x, torch.Tensor)
            b = isinstance(t, torch.Tensor)
            c = isinstance(x, int)
            d = isinstance(3, int)
            e = isinstance([1, 2, 3], list)
            f = isinstance({"foo": 1, "bar": 2}, dict)
            res = [a, b, c, d, e, f]
            # Can't run yet due to other unimplemented instructions
            # res += [isinstance(torch.nn.LazyLinear(2, 3), torch.nn.Linear)]
            return res

        torchdynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    @make_test
    def test_float(x):
        y = float(1.2)
        y += float("1.2")
        return torch.add(x, y)

    @make_test
    def test_dtype(x):
        if x.dtype == torch.float32:
            return x + 1

    @make_test
    def test_device(x):
        if not x.is_cuda:
            return x + 1

    @make_test
    def test_ndim(x):
        if x.ndim == 2 and x.ndimension() == 2 and x.dim() == 2:
            return x + 1

    @make_test
    def test_shape1(x):
        if x.shape[0] == 10:
            return x + 1

    @make_test
    def test_shape2(x):
        if x.size(1) == 10:
            return x + 1

    @make_test
    def test_del(a, b):
        c = a + b
        del a, b
        return c

    def test_fold(self):
        def fn(a):
            return a + math.sqrt(63)

        torchdynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_shape_unpack(self):
        def fn(x):
            a, b = x.size()
            return x * b

        i = torch.randn(5, 10)
        r1 = fn(i)
        with eval_frame.optimize(convert_frame_assert(lambda gm: gm.forward)):
            r2 = fn(i)
        self.assertTrue(same(r1, r2))

    def test_empty_list(self):
        def fn(x, l):
            if len(l) == 0 and not l and l is not None:
                return x + 1

        i = torch.randn(5, 10)
        r1 = fn(i, [])
        with eval_frame.optimize(convert_frame_assert(lambda gm: gm.forward)):
            r2 = fn(i, [])
            r3 = fn(i, tuple())
        self.assertTrue(same(r1, r2))
        self.assertTrue(same(r1, r3))

    @make_test
    def test_chunks1(x):
        chunk_size = 5
        assert x.shape[0] % chunk_size == 0
        assert x.shape[0] // chunk_size == 2
        return x[:chunk_size] - x[chunk_size:]

    def test_config_obj(self):
        class Cfg:
            def __init__(self):
                self.val = 0.5
                self.count = 3

        def fn(x, cfg):
            for i in range(cfg.count):
                x = x + cfg.val
            return x

        cfg1 = Cfg()
        cfg1.val = 1.0
        cfg2 = Cfg()
        v = torch.zeros(1)
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnts)):
            v = fn(v, cfg1)  # 3
            v = fn(v, cfg2)  # 4.5
            cfg2.count = 1
            v = fn(v, cfg2)  # 5
            cfg2.val = 2.0
            v = fn(v, cfg2)  # 7
        self.assertEqual(v[0], 7)
        self.assertTrue(cnts.op_count, 8)
