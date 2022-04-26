#!/usr/bin/env pytest
import importlib
import unittest

import torch

import torchdynamo
from torchdynamo.optimizations import backends
from torchdynamo.optimizations.inference import offline_autotuner
from torchdynamo.optimizations.normalize import Inplacifier
from torchdynamo.optimizations.normalize import normalize
from torchdynamo.testing import same


def has_onnxruntime():
    try:
        importlib.import_module("onnxruntime")
        return True
    except ImportError:
        return False


def has_ipex():
    try:
        importlib.import_module("intel_extension_for_pytorch")
        return True
    except ImportError:
        return False


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


class Conv_Bn_Relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_Bn_Relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


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
            r1 = graph(*example_inputs)[0]
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

    @unittest.skipIf(not has_onnxruntime(), "requires onnxruntime")
    def test_export(self):
        s = Seq()
        i = torch.randn(10)
        r1 = s(i)
        with torchdynamo.optimize_assert(offline_autotuner):
            r2 = s(i)
        self.assertTrue(same(r1, r2))

    @unittest.skipIf(not has_ipex(), "requires ipex")
    def test_ipex_fp32(self):
        model = Conv_Bn_Relu(3, 32, kernel_size=3, stride=1)
        model = model.to(memory_format=torch.channels_last)
        model = model.eval()
        input = torch.randn(8, 3, 64, 64).contiguous(memory_format=torch.channels_last)
        r1 = model(input)
        with torchdynamo.optimize(backends.ipex_fp32), torch.no_grad():
            r2 = model(input)
        self.assertTrue(same(r1, r2))
        self.assertEqual(r2.dtype, torch.float32)

    @unittest.skipIf(not has_ipex(), "requires ipex")
    def test_ipex_bf16(self):
        model = Conv_Bn_Relu(3, 32, kernel_size=3, stride=1)
        model = model.to(memory_format=torch.channels_last)
        model = model.eval()
        input = torch.randn(8, 3, 64, 64).contiguous(memory_format=torch.channels_last)
        r1 = model(input)
        with torchdynamo.optimize(
            backends.ipex_bf16
        ), torch.no_grad(), torch.cpu.amp.autocast():
            r2 = model(input)
        self.assertTrue(same(r1, r2.float(), tol=0.1))
        self.assertEqual(r2.dtype, torch.bfloat16)
