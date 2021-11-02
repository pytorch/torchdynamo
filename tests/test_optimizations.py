#!/usr/bin/env pytest
import torch

import torchdynamo
from torchdynamo.optimizations.normalize import Inplacifier
from torchdynamo.optimizations.normalize import normalize
from torchdynamo.optimizations.inference import user_compiler
from torchdynamo.testing import same


class Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class TestOptimizations(torchdynamo.testing.TestCase):
    def test_inplacifier(self):
        gm = torch.fx.symbolic_trace(Seq())
        normalize(gm)
        Inplacifier(gm).inplacify()
        gm.recompile()
        code = gm.code.replace(" ", "")
        self.assertIn("inplace=True", code)
        self.assertIn("out=linear_1", code)

    def test_example_inputs(self):
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            nonlocal r1
            r1 = graph(*example_inputs)
            return graph.forward

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        with torchdynamo.optimize_assert(compiler_fn):
            r3 = fn(a, (b, c), d)

        self.assertIsNotNone(r1)
        self.assertTrue(same(r1, r2))
        self.assertTrue(same(r1, r3))

    def test_export(self):
        s = Seq()
        i = torch.randn(10)
        r1 = s(i)
        with torchdynamo.optimize_assert(user_compiler):
            r2 = s(i)
        self.assertTrue(same(r1, r2))
