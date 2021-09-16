#!/usr/bin/env pytest
import unittest

import torch

import torchdynamo.testing
from torchdynamo.testing import unsupported

globalmod = torch.nn.ReLU()


def indirectly_unsupported(a, b):
    c = a + b
    return unsupported(a, c)


class SubGraphTests(torchdynamo.testing.TestCase):
    def _common(self, fn, frame_count, op_count):
        torchdynamo.reset()
        v1 = torch.ones(10)
        v2 = torch.ones(10) * -2.0
        correct1 = fn(v1, v2)
        correct2 = fn(v2, v1)
        cnt = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize_assert(cnt):
            r1 = fn(v1, v2)
            r2 = fn(v2, v1)
        self.assertTrue(torchdynamo.testing.same(r1, correct1))
        self.assertTrue(torchdynamo.testing.same(r2, correct2))
        self.assertEqual(cnt.frame_count, frame_count)
        self.assertEqual(cnt.op_count, op_count)

    def test_control_flow1(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            if c1.sum() > c2.sum():
                return c1
            else:
                return c2

        self._common(fn, 1, 5)

    def test_control_flow2(self):
        def fn(a, b):
            if a.sum() > b.sum():
                return 1
            else:
                return 2

        self._common(fn, 1, 3)

    def test_control_flow3(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            m = globalmod
            if c1.sum() > c2.sum():
                return m(c1)
            else:
                return m(c2)

        self._common(fn, 1, 5)

    def test_capi_call1(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            return unsupported(c1, c2)

        self._common(fn, 1, 2)

    def test_capi_call2(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            return a - (b - unsupported(c1, c2))

        self._common(fn, 1, 2)

    def test_capi_call3(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            return torchdynamo.testing.unsupported(c1, c2)

        self._common(fn, 1, 2)

    def test_indirect_unsupported1(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            return indirectly_unsupported(c1, c2)

        self._common(fn, 2, 3)

    @unittest.skip("todo")
    def test_indirect_unsupported2(self):
        def fn(a, b):
            local_const = 1
            c1 = a - b
            c2 = b - a
            return local_const + indirectly_unsupported(c1, c2)

        self._common(fn, 2, 3)

    @unittest.skip("todo")
    def test_indirect_unsupported3(self):
        def fn(a, b):
            args = [a - b, b - a]
            return indirectly_unsupported(*args)

        self._common(fn, 2, 3)
