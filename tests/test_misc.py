#!/usr/bin/env pytest
import collections
import copy
import dataclasses
import dis
import enum
import functools
import math
import random
import sys
import typing
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch
from torch.testing._internal.jit_utils import JitTestCase

import torchdynamo.testing
from torchdynamo import bytecode_transformation
from torchdynamo.testing import CompileCounter
from torchdynamo.testing import requires_static_shapes
from torchdynamo.testing import same
from torchdynamo.testing import unsupported

mytuple = collections.namedtuple("mytuple", ["a", "b", "ab"])


def my_custom_function(x):
    return x + 1


class MiscTests(torchdynamo.testing.TestCase):
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
        with torchdynamo.optimize_assert(counter):
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
        with torchdynamo.optimize_assert(counter):
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
        with torchdynamo.optimize(counter):
            self.assertRaises(AssertionError, lambda: fn(a, b, c, AssertionError))
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)

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

        torchdynamo.testing.standard_test(
            self, unpack4, 2, expected_ops=5, expected_ops_dynamic=8
        )

    def test_unpack5(self):
        def unpack5(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.shape
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torchdynamo.testing.standard_test(
            self, unpack5, 2, expected_ops=5, expected_ops_dynamic=8
        )

    def test_matmul1(self):
        def matmul_op1(a, b):
            return a @ b

        # TODO(jansel): FX doesn't support this, should add upstream support
        torchdynamo.testing.standard_test(self, matmul_op1, 2, expected_ops=1)

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
        with torchdynamo.optimize("eager"):
            r2 = fn(i)
        self.assertTrue(same(r1, r2))

    def test_empty_list(self):
        def fn(x, ll):
            if len(ll) == 0 and not ll and ll is not None:
                return x + 1

        i = torch.randn(5, 10)
        r1 = fn(i, [])
        with torchdynamo.optimize("eager"):
            r2 = fn(i, [])
            r3 = fn(i, tuple())
        self.assertTrue(same(r1, r2))
        self.assertTrue(same(r1, r3))

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
        with torchdynamo.optimize(cnts):
            v = fn(v, cfg1)  # 3
            v = fn(v, cfg2)  # 4.5
            cfg2.count = 1
            v = fn(v, cfg2)  # 5
            cfg2.val = 2.0
            v = fn(v, cfg2)  # 7
        self.assertEqual(v[0], 7)
        self.assertEqual(cnts.op_count, 8)

    def test_config_getattr_default(self):
        class Cfg:
            def __init__(self):
                self.val = 0.5
                self.count = 10

        def fn(x, cfg):
            if getattr(cfg, "just_add_7", False):
                return x + 7
            for i in range(cfg.count):
                x = x + cfg.val
            return x

        cfg1 = Cfg()
        v = torch.zeros(1)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertEqual(fn(v, cfg1)[0], 5)
            self.assertEqual(fn(v, cfg1)[0], 5)
            cfg1.just_add_7 = True
            self.assertEqual(fn(v, cfg1)[0], 7)
            self.assertEqual(fn(v, cfg1)[0], 7)
            cfg1.just_add_7 = False
            self.assertEqual(fn(v, cfg1)[0], 5)
            self.assertEqual(fn(v, cfg1)[0], 5)
        self.assertEqual(cnts.frame_count, 3)

    def test_size_input(self):
        def fn(x, s):
            a, b = s
            return x + (a - b)

        v = torch.zeros(10, 20)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
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
        with torchdynamo.optimize(cnts):
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
        with torchdynamo.optimize(cnts):
            self.assertIsNone(fn(v, v))
            self.assertEqual(out[0], 1200)
        self.assertEqual(cnts.op_count, 3)

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
        with torchdynamo.optimize(cnts):
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
        with torchdynamo.optimize(cnts):
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
        with torchdynamo.optimize(cnts):
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
        with torchdynamo.optimize(cnts):
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
        with torchdynamo.optimize(cnts):
            self.assertEqual(fn2({"a": v1, "b": v2}), 302)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

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

        return torchdynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=2, expected_ops_dynamic=4
        )

    def test_pair(self):
        def fn(a):
            return (
                torch.zeros(torch.nn.modules.utils._pair(a.size()))
                + a
                + torch.ones(torch.nn.modules.utils._ntuple(3)(3)).sum()
            )

        return torchdynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=5, expected_ops_dynamic=8
        )

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    def test_tensor_item_capture(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize((cnts)):
            self.assertEqual(fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    @patch.object(torchdynamo.config, "capture_scalar_outputs", False)
    def test_tensor_item_no_capture(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize((cnts)):
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
        with torchdynamo.optimize(cnts):
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
        with torchdynamo.optimize(cnts):
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

    def test_list_mul(self):
        def fn(count):
            head_mask = count * [None] * count
            return head_mask

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertEqual(fn(2), [None] * 4)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(cnts.op_count, 0)

    def test_user_getattr1(self):
        class MyConfig(dict):
            def __getattr__(self, name):
                return self[name]

        def fn(cfg, x, y):
            return x + y + cfg.offset

        x = torch.randn(10)
        cfg = MyConfig(offset=5)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertTrue(same(fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_user_getattr2(self):
        class MyConfig:
            defined_on_class = 1

            def __init__(self):
                self.defined_on_object = 2

            def __getattr__(self, name):
                return 3

        def fn(cfg, x):
            return x + cfg.defined_on_class - cfg.defined_on_object + cfg.not_defined

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertTrue(same(fn(cfg, x), x + 1 - 2 + 3))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_user_property(self):
        class MyConfig:
            @property
            def prop5(self):
                return 5

        def fn(cfg, x, y):
            return x + y + cfg.prop5

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertTrue(same(fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_dataclass_fields(self):
        @dataclasses.dataclass
        class MyDataClass:
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None
            d: torch.Tensor = None
            e: torch.Tensor = None

        def fn(obj):
            class_fields = dataclasses.fields(obj)
            assert len(class_fields)
            assert all(field.default is None for field in class_fields[1:])
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            assert not other_fields_are_none

            total = getattr(obj, class_fields[0].name)
            for field in class_fields[1:]:
                v = getattr(obj, field.name)
                if v is not None:
                    total += v

            return total

        obj1 = MyDataClass(torch.randn(10), torch.randn(10), torch.randn(10))
        obj2 = MyDataClass(torch.randn(10), e=torch.randn(10))
        correct1 = fn(obj1)
        correct2 = fn(obj2)

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertTrue(same(fn(obj1), correct1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertTrue(same(fn(obj2), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    @requires_static_shapes
    def test_tensor_build_list_unpack(self):
        def fn(x):
            # seen in fastNLP_Bert
            return torch.cat([*x], dim=-1)

        val = torch.randn([1, 1, 473, 768])
        correct = fn(val)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertTrue(same(fn(val), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_numpy_int_constant(self):
        def fn(x, a, b):
            return x + (a % b)

        args = [torch.randn(10), 4096, np.int64(8)]
        correct = fn(*args)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertTrue(same(fn(*args), correct))
            self.assertTrue(same(fn(*args), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_dict_mutation_side_effect(self):
        def fn(d):
            d["c"] = d["a"] + d.pop("b")
            return d

        args1 = {"a": torch.randn(10), "b": torch.randn(10)}
        args2 = dict(args1)
        assert fn(args1) is args1
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertIs(fn(args2), args2)
            self.assertTrue(same(args1, args2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_module_deepcopy(self):
        m1 = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )
        m2 = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

        def fn(m, x):
            m_copy = copy.deepcopy(m)
            return m_copy(x)

        v = torch.randn(10)
        correct1 = fn(m1, v)
        correct2 = fn(m2, v)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            for _ in range(10):
                self.assertTrue(same(fn(m1, v), correct1))
        with torchdynamo.optimize(cnts):
            for _ in range(10):
                self.assertTrue(same(fn(m2, v), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_type_copy(self):
        def fn(seq):
            a, b = seq
            return type(seq)([a + 1, b + 2, a + b])

        args1 = [torch.randn(10), torch.randn(10)]
        args2 = tuple([torch.randn(10), torch.randn(10)])
        correct1 = fn(args1)
        correct2 = fn(args2)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertTrue(same(fn(args1), correct1))
            self.assertTrue(same(fn(args2), correct2))
            self.assertIsInstance(fn(args1), list)
            self.assertIsInstance(fn(args2), tuple)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

    def test_setattr_mutation1(self):
        class MyObj:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        def fn(obj):
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            obj.c = obj.a * obj.b + 4
            obj.b = obj.a * obj.c + 5
            obj.a = obj.b * obj.c + 6
            return obj

        x1 = torch.randn(10)
        x2 = torch.randn(10)
        obj1 = MyObj(x1, x2)
        obj2 = MyObj(x1, x2)
        fn(obj2)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            self.assertIs(fn(obj1), obj1)
        self.assertTrue(same(obj1.a, obj2.a))
        self.assertTrue(same(obj1.b, obj2.b))
        self.assertTrue(same(obj1.c, obj2.c))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 12)

    def test_setattr_mutation2(self):
        class MyObj:
            def __init__(self, x):
                self.a = x + 1
                self.b = x + 2

        def fn(x):
            x = x / 3.0
            obj = MyObj(x)
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            return obj

        x1 = torch.randn(10)
        obj2 = fn(x1)

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            obj1 = fn(x1)
        self.assertTrue(same(obj1.a, obj2.a))
        self.assertTrue(same(obj1.b, obj2.b))
        self.assertTrue(same(obj1.c, obj2.c))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    def test_setattr_mutation3(self):
        # TODO(jansel): dead code eliminate the object creation
        class MyObj:
            def __init__(self, x):
                super().__init__()
                self.a = x + 1
                self.b = x + 2

        def fn(x):
            x = x / 3.0
            obj = MyObj(x)
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            return obj.a, obj.b, obj.c

        x1 = torch.randn(10)
        obj2 = fn(x1)

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            obj1 = fn(x1)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    def test_user_defined_class_name(self):
        class MyClassFoo:
            pass

        def fn1(a, b, c):
            tmp = MyClassFoo()
            if tmp.__class__.__name__ == "MyClassFoo":
                return a - b / c

        torchdynamo.testing.standard_test(self, fn=fn1, nargs=3)

    def test_manual_seed(self):
        def fn(a, b):
            x = a + b
            torch.manual_seed(9000)
            return x + 1

        torchdynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_usr_cls_staticmethod(self):
        class Foo:
            @staticmethod
            def bar(a, b):
                return a + b

        def fn(a, b):
            return Foo.bar(a, b) - 1

        torchdynamo.testing.standard_test(self, fn=fn, nargs=2)

    def test_usr_cls_classmethod(self):
        class Foo:
            @classmethod
            def bar(cls, a, b):
                return a + b

        def fn(a, b):
            return Foo.bar(a, b) - 1

        torchdynamo.testing.standard_test(self, fn=fn, nargs=2)

    def test_dunder_methods(self):
        class Foo:
            def __init__(self, val):
                super().__init__()
                self.val = val

            def __add__(self, other):
                return Foo(self.val + other.val)

            def __mul__(self, other):
                return Foo(self.val * other.val)

            def __truediv__(self, other):
                return Foo(self.val / other.val)

            def __sub__(self, other):
                return Foo(self.val - other.val)

        def fn(a, b, c):
            return Foo(a) + Foo(b) * Foo(c) / Foo(a) - Foo(b)

        torchdynamo.testing.standard_test(self, fn=fn, nargs=3, expected_ops=4)

    def test_function_annotation(self):
        class Variable:
            pass

        def fn(x):
            x = x / 3.0

            def inner(y: typing.List[Variable]):
                return x + 1

            return inner

        x1 = torch.randn(10)
        obj2 = fn(x1)([])

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize_assert(cnts):
            obj1 = fn(x1)([])
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_nested_closure(self):
        v0 = torch.randn(10)

        def fn1():
            v1 = torch.randn(10)

            def fn2(*args, **kwargs):
                assert len(args) == 1
                assert len(kwargs) == 1
                v2 = torch.randn(10) + args[0] + kwargs["b"]

                def fn3(v3=torch.randn(10)):
                    def fn4():
                        return v0 + v1 + v2 + v3 + 1

                    return fn4

                return fn3

            return fn2(1, b=2)()

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize_assert(cnts):
            tmp1 = fn1()
            tmp2 = fn1()
            self.assertTrue(tmp1().shape, (10,))
            self.assertTrue(same(tmp1(), tmp1()))
            self.assertFalse(same(tmp1(), tmp2()))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    def test_nested_closure_mutation(self):
        def fn1():
            v1 = torch.randn(10)

            def fn2():
                v2 = torch.randn(10)

                def fn3():
                    nonlocal v1, v2
                    v1 += 1
                    v2 += 2
                    return v1 + v2

                return fn3

            rv = fn2()
            rv()
            rv()
            return rv

        torch.manual_seed(9000)
        counter1 = fn1()
        result1 = [counter1(), counter1(), counter1()]

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize_assert(cnts):
            torch.manual_seed(9000)
            counter2 = fn1()
            result2 = [counter2(), counter2(), counter2()]
            result1.append(counter1())
        result2.append(counter2())

        self.assertTrue(same(result1, result2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 11)

    def test_write_to_closures_in_inlining(self):
        out = []
        for use_dynamo in [False, True]:

            def make_counter():
                x = torch.randn(10)

                def counter():
                    nonlocal x
                    x = x + 1
                    return x

                return counter

            torch.manual_seed(0)
            counter = make_counter()
            if not use_dynamo:
                out.append(counter() + counter())
            else:
                cnts = torchdynamo.testing.CompileCounter()

                @torchdynamo.optimize(cnts, nopython=True)
                def fn(counter):
                    return counter() + counter()

                out.append(fn(counter))
                self.assertEqual(cnts.frame_count, 1)
                self.assertEqual(cnts.op_count, 3)
                self.assertFalse(same(counter() + counter(), out[-1]))

        self.assertTrue(same(out[0], out[1]))

    def test_top_package_import(self):
        def fn(x):
            import torch.fx

            assert not isinstance(x, torch.fx.Proxy)
            return torch.sin(x)

        x = torch.randn(4, 5)
        ref = fn(x)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize_assert(cnts):
            res = fn(x)
        self.assertTrue(same(ref, res))

    def test_nested_optimize_decorator(self):
        cnts2 = torchdynamo.testing.CompileCounter()
        cnts3 = torchdynamo.testing.CompileCounter()

        @torchdynamo.run()
        def fn1(x):
            return torch.sin(x) * 10

        @torchdynamo.optimize(cnts2, nopython=True)
        def fn2(x):
            return fn1(x) + 1

        @torchdynamo.optimize(cnts3, nopython=True)
        def fn3(x):
            return torch.relu(fn2(x))

        fn3(torch.randn(4, 5))
        self.assertEqual(cnts2.frame_count, 0)
        self.assertEqual(cnts3.frame_count, 1)
        self.assertEqual(cnts3.op_count, 4)

    def test_nested_disable_decorator(self):
        cnts = torchdynamo.testing.CompileCounter()

        @torchdynamo.disable()
        def fn1(x):
            return torch.sin(x) * 10

        @torchdynamo.optimize(cnts)
        def fn2(x):
            x = x + 1
            x = x + 1
            x = fn1(x)  # graph break
            x = x + 1
            x = x + 1
            return x

        @torchdynamo.optimize(cnts, nopython=True)
        def fn3(x):
            return fn2(x)

        fn2(torch.randn(4, 5))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        try:
            fn3(torch.randn(4, 5))
            self.assertFalse(True)
        except torchdynamo.exc.Unsupported as e:
            self.assertIn("call torchdynamo.disable() wrapped function", str(e))

    def test_torch_size(self):
        cnts = torchdynamo.testing.CompileCounter()

        def fn(x):
            output_size = torch.Size([10, 10])
            x = x.view(*output_size)
            return (x,)

        x = torch.randn(100, requires_grad=True)
        x_clone = x.clone()
        ref = fn(x)

        with torchdynamo.optimize(cnts, nopython=True):
            res = fn(x_clone)

        self.assertTrue(same(ref, res))

    def test_torch_seed(self):
        cnts = torchdynamo.testing.CompileCounter()

        def fn(x):
            attention_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(attention_seed)
            return (x,)

        x = torch.randn(100, requires_grad=True)
        ref = fn(x)

        with torchdynamo.optimize(cnts, nopython=True):
            res = fn(x)

        self.assertTrue(same(ref, res))

    def test_is_tensor_like(self):
        cnts = torchdynamo.testing.CompileCounter()

        def f(x):
            if torch.overrides.is_tensor_like(x):
                return (x * 2,)
            return (torch.ones(10) + x,)

        x = torch.randn(10)
        ref0 = f(x)
        ref1 = f(4)
        with torchdynamo.optimize(cnts, nopython=True):
            res0 = f(x)
            res1 = f(4)
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    def test_version_ci(self):
        # temporary test to check that the ci torch version is set correctly
        self.assertTrue(hasattr(torch, "_subclasses"))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_rand(self):
        cnts = torchdynamo.testing.CompileCounter()
        device = "cuda"

        def fn():
            return torch.randn(10, device=device)

        torch.manual_seed(10)
        ref_run1 = fn()

        torch.manual_seed(10)
        ref_run2 = fn()
        self.assertTrue(same(ref_run1, ref_run2))

        torch.manual_seed(10)
        with torchdynamo.optimize(cnts, nopython=True):
            res = fn()

        self.assertTrue(same(res, ref_run1))

    def test_slice_input(self):
        cnts = torchdynamo.testing.CompileCounter()

        def getitem(a, idx):
            if isinstance(idx, slice):
                return (
                    torch.zeros(1),
                    a[idx]
                    + [
                        100,
                    ],
                )
            else:
                return (torch.zeros(1), a[idx])

        layers = list(range(10))
        ref0 = getitem(layers, slice(0, 2, 1))
        ref1 = getitem(layers, 2)
        ref2 = getitem(layers, slice(3, 8, 2))
        with torchdynamo.optimize(cnts, nopython=True):
            res0 = getitem(layers, slice(0, 2, 1))
            res1 = getitem(layers, 2)
            res2 = getitem(layers, slice(3, 8, 2))

        self.assertTrue(ref0 == res0)
        self.assertTrue(ref1 == res1)
        self.assertTrue(ref2 == res2)

    def test_grad(self):
        cnts = torchdynamo.testing.CompileCounter()

        def fn(a, b):
            out = a * b
            out.sum().backward()
            real_out = torch.sigmoid(a.grad + b)
            return real_out

        inps = [torch.randn(4, requires_grad=True) for _ in range(2)]
        for inp in inps:
            inp.grad = None
        ref = fn(*inps)

        for inp in inps:
            inp.grad = None
        with torchdynamo.optimize(cnts):
            res = fn(*inps)

        self.assertTrue(same(ref, res))

    @unittest.skipIf(sys.version_info < (3, 10), "use linetable when python >= 3.10")
    def test_linetable_writer(self):
        def fn():
            a = 10
            b = 20
            c = a + b
            f = "linetable_writer"
            return f"Test if {f} generates correct co_linetable: {c}"

        inst = dis.get_instructions(fn)
        result = bytecode_transformation.assemble(inst, fn.__code__.co_firstlineno)
        self.assertTrue(result[1] == fn.__code__.co_linetable)

    @unittest.skipIf(sys.version_info >= (3, 10), "use lnotab when python < 3.10")
    def test_lnotab_writer(self):
        def fn():
            a = 10
            b = 20
            c = a + b
            f = "lnotab_writer"
            return f"Test if {f} generates correct co_lnotab: {c}"

        inst = dis.get_instructions(fn)
        result = bytecode_transformation.assemble(inst, fn.__code__.co_firstlineno)
        self.assertTrue(result[1] == fn.__code__.co_lnotab)

    def test_python_slice(self):
        def f1(input):
            y = 0
            for i, x in enumerate(input[2:], 1):
                y = y + x
            return y

        def f2(input):
            y = 0
            for i, x in enumerate(input.shape[2:], 1):
                y = y + x
            return y

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            res1 = f1([1, 2, 3, 5])
            res2 = f2(torch.rand([2, 3, 4, 5]))

        self.assertEqual(res1, 8)
        self.assertEqual(res2, 9)

    def test_const_dict_variable_python_type(self):
        from torchdynamo.variables import ConstDictVariable

        d1 = {"a": 10, "b": 20}
        d2 = collections.OrderedDict([("x", 12), ("y", 22)])
        self.assertEqual(ConstDictVariable(d1, dict).python_type(), dict)
        self.assertEqual(
            ConstDictVariable(d2, collections.OrderedDict).python_type(),
            collections.OrderedDict,
        )

    def test_builtin_subclasses_as_method_on_class_type(self):
        class Foo:
            def __init__(name):
                self.ame_ = name

            def get_name(self):
                return "Foo " + self.name_

        class Bar(Foo):
            def __init__(name):
                self.name_ = name

            def get_name(self):
                return "Bar " + self.name_

        class Baz(Foo):
            def __init__(name):
                self.name_ = name

            def get_name(self):
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()

        counter = CompileCounter()

        @torchdynamo.optimize_assert(counter)
        def fn():
            return Foo.__subclasses__()

        subs_of_foo_optim = fn()

        self.assertEqual(len(subs_of_foo_reg), 2)
        self.assertEqual(subs_of_foo_reg, subs_of_foo_optim)

    def test_builtin_subclasses_as_method_on_var(self):
        class Foo:
            def __init__(name):
                self.name_ = name

            def get_name(self):
                return "Foo " + self.name_

        class Bar(Foo):
            def __init__(name):
                self.name_ = name

            def get_name(self):
                return "Bar " + self.name_

        class Baz(Bar):
            def __init__(name):
                self.name_ = name

            def get_name(self):
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()
        sub_of_foo_subclass_var_reg = subs_of_foo_reg[0].__subclasses__()

        sub_of_foo_subclass_var_optim = list()
        counter = CompileCounter()

        @torchdynamo.optimize_assert(counter)
        def fn():
            return Foo.__subclasses__()

        @torchdynamo.optimize_assert(counter)
        def fn_single(subs_of_foo_optim):
            return subs_of_foo_optim[0].__subclasses__()

        subs_of_foo_optim = fn()
        sub_of_foo_subclass_var_optim = fn_single(subs_of_foo_optim)

        self.assertEqual(len(sub_of_foo_subclass_var_optim), 1)
        self.assertEqual(sub_of_foo_subclass_var_optim, sub_of_foo_subclass_var_reg)

    def test_enum_no_graphbreaks(self):
        class Foo(enum.Enum):
            FOO = 0
            BAR = 1

        def fn(x, foo):
            if foo is Foo.FOO:
                x = torch.add(x, 1.0)
            x = torch.mul(x, 1.0)
            return x

        x = torch.randn(1)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts, nopython=True):
            fn(x, Foo.FOO)
        self.assertEqual(cnts.op_count, 2)

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts, nopython=True):
            fn(x, Foo.BAR)
        self.assertEqual(cnts.op_count, 1)

    def test_id_of_nn_module(self):
        class M(torch.nn.Module):
            def forward(self, x, ref_id):
                self_id = id(self)
                if self_id == ref_id:
                    x = torch.mul(x, 1.0)
                x = torch.add(x, 1.0)
                return x

        m = M().eval()
        data = torch.randn(1)
        cnts = torchdynamo.testing.CompileCounter()
        correct_ref_id = id(m)
        with torchdynamo.optimize(cnts, nopython=True):
            m(data, correct_ref_id)
        self.assertEqual(cnts.op_count, 2)

        cnts = torchdynamo.testing.CompileCounter()
        incorrect_ref_id = id(m) + 1
        with torchdynamo.optimize(cnts, nopython=True):
            m(data, incorrect_ref_id)
        self.assertEqual(cnts.op_count, 1)

    def test_inline_func_jump_on_tensor_condition(self):
        def f1(input):
            if input == 0:
                return input + 1
            else:
                return input + 2

        def f2(input):
            return f1(input)

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            res1 = f2(torch.tensor([1.0]))
            res2 = f2(torch.tensor([0.0]))

        self.assertEqual(res1, 3)
        self.assertEqual(res2, 1)

    def test_frozenset_torch_func_contains(self):
        funcs = frozenset([torch.add])

        def fn(x, func):
            if func in funcs:
                x = torch.add(x, 1.0)
            x = torch.mul(x, 1.0)
            return x

        x = torch.randn(1)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts, nopython=True):
            fn(x, torch.add)
        self.assertEqual(cnts.op_count, 2)

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts, nopython=True):
            fn(x, torch.mul)
        self.assertEqual(cnts.op_count, 1)

    @patch.object(torchdynamo.config, "fake_tensor_propagation", True)
    def test_unsupported_fake_tensor(self):
        def f(x):
            return torch.quantize_per_tensor(
                torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8
            )

        x = torch.randn(2, 2)
        with self.assertRaises(RuntimeError):
            with torchdynamo.optimize_assert(torchdynamo.testing.CompileCounter()):
                f(x)

        with patch.object(torchdynamo.config, "fake_tensor_propagation", False):
            with torchdynamo.optimize_assert(torchdynamo.testing.CompileCounter()):
                f(x)

    def test_inline_list_mutation(self):
        def f1(x):
            x.append(torch.ones(8))
            return x

        def f2():
            x = [torch.ones(6)]
            f1(x)
            return x

        res1 = f2()
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            res2 = f2()
        self.assertTrue(same(res1, res2))

    def test_inline_dict_mutation(self):
        def f1(d):
            d["c"] = d["a"] + d.pop("b")
            return d

        def f2():
            d = {"a": torch.ones(5), "b": torch.ones(5)}
            f1(d)
            return d

        res1 = f2()
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            res2 = f2()
        self.assertTrue(same(res1, res2))

    def test_recursive_inline_list_mutation(self):
        def f1(x, y):
            x.append(torch.tensor([1.1]))
            y.append(torch.tensor([1.2]))
            return x, y

        def f2(x, y):
            x.append(torch.tensor([2.1]))
            y.append(torch.tensor([2.2]))
            f1(x, y)
            return x, y

        def f3(x):
            x.append(torch.tensor([3.1]))
            y = [torch.tensor([3.2])]
            f2(x, y)
            return x, y

        def f4():
            x = [torch.tensor([4.1])]
            return f3(x)

        res1 = f4()
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            res2 = f4()
        self.assertTrue(same(res1, res2))

    def test_disallow_in_graph(self):
        cnts = torchdynamo.testing.CompileCounter()

        @torchdynamo.optimize(cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        torchdynamo.disallow_in_graph(torch.sub)
        fn(torch.randn(10))
        torchdynamo.allow_in_graph(torch.sub)

        # check for graph break on sub
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

    def test_allow_in_graph(self):
        cnts = torchdynamo.testing.CompileCounter()

        @torchdynamo.optimize(cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        torchdynamo.allow_in_graph(my_custom_function)
        fn(torch.randn(10))
        torchdynamo.disallow_in_graph(my_custom_function)

        # check for no graph break
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_sample_input(self):
        from torch.testing._internal.common_methods_invocations import SampleInput

        def fn(sample):
            if isinstance(sample.input, torch.Tensor):
                return sample.input * 2
            return torch.zeros(())

        sample = SampleInput(torch.ones(2))
        ref = fn(sample)

        with torchdynamo.optimize("eager"):
            res = fn(sample)

        self.assertTrue(same(ref, res))

    def test_release_input_memory(self):
        x = torch.rand([4])
        x_ref = weakref.ref(x)

        cnts = torchdynamo.testing.CompileCounter()

        @torchdynamo.optimize(cnts)
        def foo(x):
            return x + x

        out = foo(x)
        self.assertTrue(same(out, x + x))
        del x
        self.assertIs(x_ref(), None)

    def test_release_module_memory(self):

        mod = torch.nn.Linear(10, 10)
        x = torch.rand([10, 10])
        mod_weight_ref = weakref.ref(mod.weight)
        mod_ref = weakref.ref(mod)

        # Modules that are passed into torchdynamo optimized functions
        # will normally be held onto through the generated GraphModule,
        # which contains the modules. remove the reference in this backend
        # and test that no additional references are being held.
        class NoLeakBackend:
            def __call__(self, gm: torch.fx.GraphModule, example_inputs):
                gm.mod = None

                def foo(*args, **kwargs):
                    return (1,)

                return foo

        no_leak_backend = NoLeakBackend()

        @torchdynamo.optimize(no_leak_backend)
        def foo(mod, x):
            return mod(x)

        foo(mod, x)
        del mod
        del x
        self.assertIsNone(mod_ref(), None)
        self.assertIsNone(mod_weight_ref(), None)

    def test_update_locals_and_stack_uses_shared_cache(self):
        def fn(x):
            perm = [0, 3, 5]
            perm = [i for i in range(min(perm))] + perm
            perm.extend(i for i in range(x.dim()) if i not in perm)
            return perm

        x = torch.rand([2, 2, 2, 2, 2, 2])
        res1 = fn(x)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            res2 = fn(x)
        self.assertTrue(same(res1, res2))

    def test_dict_reconstruct_keeps_original_order(self):
        def fn():
            modules = collections.OrderedDict([("act", torch.nn.ReLU())])
            module_dict = torch.nn.ModuleDict(modules)

            next_modules = {"fc4": torch.nn.Linear(5, 6), "act3": torch.nn.Sigmoid()}
            modules.update(next_modules.items())
            module_dict.update(next_modules)
            return modules, module_dict

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            modules, module_dict = fn()

        self.assertEqual(len(module_dict), len(modules))
        for k1, m2 in zip(modules, module_dict.children()):
            self.assertTrue(modules[k1] is m2)

    def test_unspecialized_primitive_variable(self):
        # correctness check
        def fn(x, y, z):
            xy = [x + y, y, False]
            np_x = x.numpy()
            np_y = y.numpy()
            return {
                "x": x,
                "z": z,
                "a": np_y.sum(),
                "b": xy,
                "c": np_y[0][0] / 68,
                "d": np_x.sum(),
            }, x + np_y.sum() + z

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        y = torch.ones([2, 2], dtype=torch.int64)
        z = np.int64(12)
        res1 = fn(x, y, z)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            res2 = fn(x, y, z)
        self.assertTrue(same(res1, res2))

    def test_unspecialized_primitive_variable2(self):
        # no recompilations if passing on different numpy int values
        def fn(x, y):
            return {"a": x + 1, "b": y / 2}

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            for i in range(10):
                fn(x, np.int64(i))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_unspecialized_primitive_variable3(self):
        # test unspecialized primitive max/min
        def fn(x, y, z):
            return z + 1, max(x, y), min(x - 4, y)

        x = np.int64(12)
        y = 10
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        res1 = fn(x, y, z)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            res2 = fn(x, y, z)
        self.assertTrue(same(res1, res2))

    def test_unspecialized_primitive_variable4(self):
        # test random functions
        def fn(x):
            r1 = random.random()
            y = x + random.uniform(10, 20)
            r2 = random.randint(2, 18)
            return y + r1, r2

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        random.seed(1)
        res1 = fn(x)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            random.seed(1)
            res2 = fn(x)
        self.assertTrue(same(res1, res2))

    def test_side_effects_codegen_update_mutated(self):
        # codegen to update mutated variables with side effect
        # should after stack value's codegen
        def f1(x):
            alist = [x]
            alist.append(x + 1)
            alist[0].sum().item()  # graph break
            res = alist.pop()
            res.sum().item()  # graph break
            return res

        def f2(a, b):
            d = {"a": a + 1, "b": b + 2}
            x = d.pop("b")
            x.sum().item()  # graph break
            y = d["a"] + x
            y.sum().item()  # graph break
            d["c"] = y
            return d

        x = torch.rand([2, 3])
        a = torch.rand([5, 6])
        b = torch.rand([5, 6])
        res11 = f1(x)
        res21 = f2(a, b)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            res12 = f1(x)
            res22 = f2(a, b)
        self.assertTrue(same(res11, res12))
        self.assertTrue(same(res21, res22))

    def test_list_append_return_none(self):
        def fn(x):
            alist = []
            blist = alist.append(x + 1)
            return alist, blist

        x = torch.tensor([2.3])
        res = fn(x)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            res2 = fn(x)
        self.assertEqual(res, res2)

    def test_tensor_types(self):
        def fn(dtype, tensor_type):
            x = torch.empty(4, dtype=dtype)
            assert isinstance(x, tensor_type)

        with torchdynamo.optimize("eager"):
            fn(torch.float32, torch.FloatTensor)
            fn(torch.float64, torch.DoubleTensor)
            fn(torch.float16, torch.HalfTensor)
            fn(torch.bfloat16, torch.BFloat16Tensor)
            fn(torch.uint8, torch.ByteTensor)
            fn(torch.int8, torch.CharTensor)
            fn(torch.int64, torch.LongTensor)
            fn(torch.int, torch.IntTensor)
            fn(torch.int16, torch.ShortTensor)
            fn(torch.bool, torch.BoolTensor)

    def test_nan(self):
        def f(x, n):
            return x * 2 + n

        x = torch.randn(4)
        n = float("nan")

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            f(x, n)
            f(x, n)
        self.assertEqual(cnts.frame_count, 1)

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    def test_item(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = torch.max(x)
                return z.int().item()

        with torchdynamo.optimize("eager", nopython=True):
            x = torch.tensor([[10.6763, 11.7445, -2.2369]])
            model = MyMod()
            y = model(x)

        self.assertEqual(y, 11)

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    def test_item_changes(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = torch.max(x)
                return z.int().item()

        with torchdynamo.optimize("eager", nopython=True):
            x = torch.tensor([[10.6763, 11.7445, -2.2369]])
            model = MyMod()
            y = model(x)
            z = model(torch.tensor([[y - 5, y + 10, y + 50]]))

        self.assertEqual(y, 11)
        self.assertEqual(z, 61)

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    def test_item_changes_new_shape(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = torch.max(x)
                return z.int().item()

        with torchdynamo.optimize("eager", nopython=True):
            x = torch.tensor([[10.6763, 11.7445, -2.2369]])
            model = MyMod()
            y = model(x)
            z = model(torch.tensor([[y - 5, y + 50], [y + 5, y - 50]]))

        self.assertEqual(y, 11)
        self.assertEqual(z, 61)

    def test_cross_entropy_loss_fancy_ctor(self):
        output = None
        rand_5 = torch.randn(5)
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        with torchdynamo.optimize("eager", nopython=True):
            loss = torch.nn.CrossEntropyLoss(
                weight=rand_5, reduce=False, label_smoothing=0.5
            )
            input = rand_3_5
            dynamo_output = loss(input, target)

        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_simple_ctor(self):
        output = None
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        with torchdynamo.optimize("eager", nopython=True):
            loss = torch.nn.CrossEntropyLoss()
            input = rand_3_5
            dynamo_output = loss(input, target)

        loss = torch.nn.CrossEntropyLoss()
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_large_reduction_list(self):
        dtype = torch.float32
        device = "cpu"

        def check_sum_all(tensor: torch.Tensor) -> None:
            pylist = tensor.reshape(-1).tolist()
            self.assertTrue(same(tensor.sum(), torch.tensor(sum(pylist))))

        check_sum_all(torch.randn(200000, dtype=dtype, device=device))

    @patch.object(torchdynamo.config, "raise_on_backend_error", True)
    def test_raise_on_backend_error(self):
        def my_compiler(gm, _):
            raise RuntimeError("duck!")

        @torchdynamo.optimize(my_compiler)
        def fn(a, b):
            return a + b / (a - b)

        self.assertRaises(
            torchdynamo.exc.BackendCompilerFailed,
            lambda: fn(torch.randn(10), torch.randn(10)),
        )

    def test_named_parameters(self):
        n_embd = 768
        block_size = 128
        vocab_size = 65
        embd_pdrop = 0.1

        class MyModel2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)

            def forward(self, x):
                return x

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)
                self.submod2 = MyModel2()

            def forward(self, x):
                return x

        # Regular
        params = []
        mod = MyModel()
        actual_params = list(mod.named_parameters())

        with torchdynamo.optimize("eager", nopython=True):
            params = list(mod.named_parameters())

        self.assertEqual(len(actual_params), len(params))
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            self.assertTrue(torch.allclose(v_a, v))

        # Prefix
        params = []
        mod = MyModel()
        actual_params = list(mod.named_parameters(prefix="foo"))

        with torchdynamo.optimize("eager", nopython=True):
            params = list(mod.named_parameters(prefix="foo"))

        self.assertEqual(len(actual_params), len(params))
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            self.assertTrue(torch.allclose(v_a, v))

    def test_module_complex_iter(self):
        n_embd = 768
        block_size = 128
        vocab_size = 65
        embd_pdrop = 0.1

        class FakeGPT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)
                self.ln_f = torch.nn.LayerNorm(n_embd)
                self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)

                self.block_size = block_size
                self.names = []

            def forward(self, idx, targets=None):
                from torch.nn import functional as F

                b, t = idx.size()
                assert (
                    t <= self.block_size
                ), "Cannot forward, model block size is exhausted."

                # forward the GPT model
                token_embeddings = self.tok_emb(
                    idx
                )  # each index maps to a (learnable) vector
                position_embeddings = self.pos_emb[
                    :, :t, :
                ]  # each position maps to a (learnable) vector
                x = self.drop(token_embeddings + position_embeddings)
                x = self.blocks(x)
                x = self.ln_f(x)
                logits = self.head(x)

                # if we are given some desired targets also calculate the loss
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )

                return logits, loss

            def foo(self, memo=None, prefix="", remove_duplicate=False):
                for mn, m in self.named_modules(
                    memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
                ):
                    for pn, p in self.named_parameters():
                        fpn = "%s.%s" % (mn, pn) if mn else pn
                        self.names.append(fpn)

        # Test plain recurse
        model_a = FakeGPT()
        model_a.foo()
        a_names = model_a.names

        model_b = FakeGPT()
        with torchdynamo.optimize("eager", nopython=True):
            model_b.foo()

        self.assertEqual(a_names, model_b.names)

        # Test with prefix
        model_a = FakeGPT()
        model_a.foo(prefix="abc")
        a_names = model_a.names

        model_b = FakeGPT()
        with torchdynamo.optimize("eager", nopython=True):
            model_b.foo(prefix="abc")

        self.assertEqual(a_names, model_b.names)


class TestTracer(JitTestCase):
    def test_jit_save(self):
        def fn():
            class Foo(torch.nn.Module):
                def __init__(self):
                    super(Foo, self).__init__()
                    self.a = 3

                @torch.jit.export
                def __getstate__(self):
                    return (3, self.training)

                @torch.jit.export
                def __setstate__(self, state):
                    self.a = state[0]
                    self.training = state[1]

                def forward(self, x):
                    return x + self.a

            f = Foo()

            return torch.jit.trace(f, (torch.rand(3, 4),))

        fn()
        with torchdynamo.optimize("eager"):
            fn()
