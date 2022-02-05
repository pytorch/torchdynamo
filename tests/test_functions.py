#!/usr/bin/env pytest
import collections
import functools
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
from torchdynamo.testing import unsupported

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


mytuple = collections.namedtuple("mytuple", ["a", "b", "ab"])


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

        torchdynamo.testing.standard_test(self, unpack4, 2, expected_ops=5)

    def test_unpack5(self):
        def unpack5(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.shape
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torchdynamo.testing.standard_test(self, unpack5, 2, expected_ops=5)

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
        c = a + 1
        d = c + 2
        del c, a
        return b + d

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
        def fn(x, ll):
            if len(ll) == 0 and not ll and ll is not None:
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
        self.assertEqual(cnts.op_count, 8)

    @make_test
    def test_import1(x, y):
        import torch
        from torch import sub

        return sub(torch.add(x, y), y)

    @make_test
    def test_return_dict(x, y):
        z = [x + y, y, False]
        return {"x": x, "z": z, "a": x, "b": z, "c": x}

    @make_test
    def test_return_dict2(x, y):
        tmp = {"x": x}
        tmp["z"] = [x + y, y]
        tmp["y"] = y
        tmp["z"].append(False)
        return tmp

    def test_size_input(self):
        def fn(x, s):
            a, b = s
            return x + (a - b)

        v = torch.zeros(10, 20)
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnts)):
            self.assertEqual(fn(v, v.size())[0, 0], -10)
            self.assertEqual(fn(v, (10, 20))[0, 0], -10)
            self.assertEqual(fn(v, [10, 20])[0, 0], -10)
        self.assertEqual(cnts.op_count, 2)

    def test_cell_output1(self):
        out = None

        def fn(a, b):
            nonlocal out
            out = a + b * 10

        v = torch.Tensor([100])
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnts)):
            self.assertIsNone(fn(v, v))
            self.assertEqual(out[0], 1100)
        self.assertEqual(cnts.op_count, 2)

    def test_cell_output2(self):
        out = None

        def fn(a, b):
            nonlocal out
            c = unsupported(a, b)
            out = a + b * 10 + c

        v = torch.Tensor([100])
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame(cnts)):
            self.assertIsNone(fn(v, v))
            self.assertEqual(out[0], 1200)
        self.assertEqual(cnts.op_count, 3)

    @make_test
    def test_funcdef_closure(x, y):
        x = x + y + 1.0

        def inner(z):
            nonlocal x, y
            y = x + z + 20.0
            x = y + z + 10.0

        inner(2.0)
        inner(3.0)

        return x, y

    def test_return_nested_function(self):
        out = None

        def fn(a, b):
            nonlocal out
            c = a + b
            d = a + 1.0

            def fn2(f: int = 7, g: float = 9.0):
                nonlocal out
                out = a + b * 10
                return c * f - d * g

            return fn2

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnts)):
            self.assertEqual(fn(v1, v2)(1.5)[0], -459)
            self.assertEqual(out[0], 2100)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 7)

    def test_tensor_dict1(self):
        def fn(inputs):
            return inputs["a"] - inputs["b"] * 1.5

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnts)):
            self.assertEqual(fn({"a": v1, "b": v2})[0], -200)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_tensor_dict2(self):
        def fn1(inputs):
            total = torch.zeros(1)
            for k, v in inputs.items():
                total += v
            return total

        def fn2(inputs):
            total = torch.zeros(1)
            for v in inputs.values():
                total += v
            return total

        def fn3(inputs):
            total = torch.zeros(1)
            for k in inputs.keys():
                total += inputs[k]
            return total

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnts)):
            self.assertEqual(fn1({"a": v1, "b": v2})[0], 300)
            self.assertEqual(fn2({"a": v1, "b": v2})[0], 300)
            self.assertEqual(fn3({"a": v1, "b": v2})[0], 300)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 9)

    def test_dictcomp(self):
        def fn1(inputs):
            return {k: v + 1 for k, v in inputs.items()}

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnts)):
            self.assertEqual(fn1({"a": v1, "b": v2})["a"], 101)
            self.assertEqual(fn1({"a": v1, "b": v2})["b"], 201)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_listcomp(self):
        def fn2(inputs):
            return torch.sum(torch.cat([v + 1 for k, v in inputs.items()], 0))

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnts)):
            self.assertEqual(fn2({"a": v1, "b": v2}), 302)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    @make_test
    def test_module_constant(x, y):
        r = x + y
        for i in range(torchdynamo.testing.three):
            r = r / y
        return r

    @make_test
    def test_inline_softmax(x, y):
        # This is common in sme huggingface models
        return torch.nn.Softmax(dim=-1)(x + y * 2)

    @make_test
    def test_dtype_compare(a, b):
        if a.dtype == torch.float16:
            return a + 10
        if a.dtype == torch.float32:
            return a - b * 32

    @make_test
    def test_build_list_unpack(a, b):
        it1 = (x + 1 for x in (a, b))
        it2 = (x - 1 for x in (a, b))
        return torch.cat([*it1, *it2], dim=-1)

    def test_is_floating_point(self):
        def fn(a, b):
            x = a + 1.0
            if torch.is_floating_point(b):
                x = x + b
            return x + 2.0

        return torchdynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_floating_point2(self):
        def fn(a, b):
            x = a + 1.0
            if b.is_floating_point():
                x = x + b
            return x + 2.0

        return torchdynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_tensor(self):
        def fn(a, b):
            x = a + 1.0
            if torch.is_tensor(b):
                x = x + b
            return x + 2.0

        return torchdynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_numel(self):
        def fn(a):
            return a + a.numel() + torch.numel(a)

        return torchdynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=2)

    def test_pair(self):
        def fn(a):
            return (
                torch.zeros(torch.nn.modules.utils._pair(a.size()))
                + a
                + torch.ones(torch.nn.modules.utils._ntuple(3)(3)).sum()
            )

        return torchdynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=5)

    def test_tensor_item(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame(cnts)):
            self.assertEqual(fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple1(self):
        def fn(a, b):
            tmp = mytuple(a, b, a + b)
            return mytuple(tmp.a, tmp[1], tmp.ab + b)

        v1 = torch.Tensor([10])
        v2 = torch.Tensor([20])
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnts)):
            self.assertEqual(fn(v1, v2).ab, 50)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple2(self):
        def fn(packed):
            a, b, c = packed
            if hasattr(packed, "b"):
                b = packed.b + 1
            c = packed[2]
            return a + b + c

        v1 = torch.Tensor([1])
        v2 = torch.Tensor([2])
        v3 = torch.Tensor([3])
        cnts = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnts)):
            self.assertEqual(fn(mytuple(v1, v2, v3))[0], 7)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_range_input(self):
        def fn(a, rng):
            x = a
            for i in rng:
                x = x + i
            return x

        return torchdynamo.testing.standard_test(
            self, fn=functools.partial(fn, rng=range(3)), nargs=1, expected_ops=3
        )

    @make_test
    def test_tensor_len(a, b):
        return a + b + len(a) + b.__len__()

    @make_test
    def test_pop(a, b):
        ll = [a, b]
        ll.append(a + 1)
        ll.extend(
            [
                b + 2,
                a + b,
            ]
        )
        ll.pop(-1)
        ll.pop(0)
        ll.pop()
        v1, v2 = ll
        return v1 - v2

    @make_test
    def test_list_convert(a, b):
        ll = [a + 2, b]
        ll = tuple(ll)
        tmp = b + 3
        ll = list(ll)
        v1, v2 = ll
        return v1 - v2 + tmp

    @make_test
    def test_list_add(a, b):
        l1 = (a, b)
        l2 = ()  # being a LOAD_CONST in the bytecode
        l3 = l1 + l2
        return l3[0] + l3[1]

    @make_test
    def test_startswith(a, b):
        x = a + b
        if "foobar".startswith("foo") and "test" in constant3.__module__:
            x = x + 1
        return x

    @make_test
    def test_dict_ops(a, b):
        tmp = {"a": a + 1, "b": b + 2}
        v = tmp.pop("b") + tmp.get("a") + tmp.get("missing", 3) + tmp.pop("missing", 4)
        tmp.update({"d": 3})
        tmp["c"] = v + tmp["d"]
        if "c" in tmp and "missing" not in tmp:
            return tmp["c"] - tmp["a"] + len(tmp)

    def test_no_grad(self):
        def fn1(a, b):
            x = a + 1
            # redundant no_grad should get ignored
            with torch.no_grad():
                x = x + b
            x = x + 2
            return x

        def fn2(a, b):
            x = a + 1
            with torch.set_grad_enabled(False):
                x = x + b
            x = x + 2
            return x

        def fn3(a, b):
            x = a + 1
            with torch.enable_grad():
                x = x + b
            x = x + 2
            return x

        def fn4(a, b):
            x = a + 1
            with torch.set_grad_enabled(True):
                if torch.is_grad_enabled():
                    x = x + b
            x = x + 2
            return x

        with torch.no_grad():
            torchdynamo.testing.standard_test(self, fn=fn1, nargs=2, expected_ops=3)
            torchdynamo.testing.standard_test(self, fn=fn2, nargs=2, expected_ops=3)
            torchdynamo.testing.standard_test(self, fn=fn3, nargs=2, expected_ops=5)
            torchdynamo.testing.standard_test(self, fn=fn4, nargs=2, expected_ops=5)
        with torch.enable_grad():
            torchdynamo.testing.standard_test(self, fn=fn1, nargs=2, expected_ops=5)
            torchdynamo.testing.standard_test(self, fn=fn2, nargs=2, expected_ops=5)
            torchdynamo.testing.standard_test(self, fn=fn3, nargs=2, expected_ops=3)
            torchdynamo.testing.standard_test(self, fn=fn4, nargs=2, expected_ops=3)

    @make_test
    def test_min_max(a, b):
        c = a + b
        a = a.sum()
        b = b.sum()
        a = min(max(a, 0), 1)
        b = max(0, min(1, b))
        return max(a, b) - min(a, b) + c

    def test_build_tuple_unpack(self):
        def fn1(a, b, c):
            return a - b / c

        def fn2(a, b, c):
            tmp1 = (a,)
            tmp2 = (b, c)
            args = (*tmp1, *tmp2)
            return fn1(*args)

        def fn3(a, *args):
            return fn1(a, *args)

        torchdynamo.testing.standard_test(self, fn=fn2, nargs=3, expected_ops=2)
        torchdynamo.testing.standard_test(self, fn=fn3, nargs=3, expected_ops=2)
