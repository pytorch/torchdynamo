#!/usr/bin/env pytest
import torch

import torchdynamo.testing

globalmod = torch.nn.ReLU()


class SubGraphTests(torchdynamo.testing.TestCase):
    def test_control_flow1(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            if c1.sum() > c2.sum():
                return c1
            else:
                return c2

        v1 = torch.ones(10)
        v2 = torch.ones(10) * -2.0
        correct = fn(v1, v2)
        cnt = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize_assert(cnt):
            r1 = fn(v1, v2)
            r2 = fn(v2, v1)
        self.assertTrue(torchdynamo.testing.same(r1, correct))
        self.assertTrue(torchdynamo.testing.same(r2, correct))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 5)

    def test_control_flow2(self):
        def fn(a, b):
            if a.sum() > b.sum():
                return 1
            else:
                return 2

        v1 = torch.ones(10)
        v2 = torch.ones(10) * -2.0
        cnt = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize_assert(cnt):
            r1 = fn(v1, v2)
            r2 = fn(v2, v1)
        self.assertEqual(r1, 1)
        self.assertEqual(r2, 2)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_control_flow3(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            m = globalmod
            if c1.sum() > c2.sum():
                return m(c1)
            else:
                return m(c2)

        v1 = torch.ones(10)
        v2 = torch.ones(10) * -2.0
        correct = fn(v1, v2)
        cnt = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize_assert(cnt):
            r1 = fn(v1, v2)
            r2 = fn(v2, v1)
        self.assertTrue(torchdynamo.testing.same(r1, correct))
        self.assertTrue(torchdynamo.testing.same(r2, correct))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 5)
