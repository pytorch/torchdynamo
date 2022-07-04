#!/usr/bin/env pytest
import importlib
import operator
import unittest
from unittest.mock import patch

import torch

import torchdynamo
import torchdynamo.config as config
from torchdynamo.optimizations import backends
from torchdynamo.optimizations.inference import fixed_strategy1
from torchdynamo.optimizations.inference import offline_autotuner
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


def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


def transform(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # operator.add)
        if node.op == "call_function":
            # The target attribute is the function
            # that call_function calls.
            if node.target == operator.mul:
                node.target = operator.add

    gm.graph.lint()  # Does some checks to make sure the
    # Graph is well-formed.

    gm.recompile()
    return gm


class TestVerifyCorrectness(torchdynamo.testing.TestCase):
    @patch.object(config, "verify_correctness", True)
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

    @patch.object(config, "verify_correctness", True)
    def test_fixed_strategy1(self):
        s = Seq()
        i = torch.randn(10)
        r1 = s(i)
        with torchdynamo.optimize(fixed_strategy1):
            r2 = s(i)
        self.assertTrue(same(r1, r2))

    @patch.object(config, "verify_correctness", True)
    def test_nnc(self):
        s = Seq()
        i = torch.randn(10)
        r1 = s(i)
        with torchdynamo.optimize("nnc"):
            r2 = s(i)
        self.assertTrue(same(r1, r2))

    @patch.object(config, "verify_correctness", True)
    def test_incorrect_verify_true(self):
        """
        Even the bad optimization return a graph that
        is not functionally equal to the original graph;
        When config.verify_correctness=True, it will
        check the correctness of outputs and fallback using
        the original graph
        """
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            return transform(gm).forward

        r1 = toy_example(i1, i2)
        with torchdynamo.optimize(incorrect_compile_fn):
            r2 = toy_example(i1, i2)
        self.assertTrue(same(r1, r2))

    @patch.object(config, "verify_correctness", False)
    def test_incorrect_verify_false(self):
        """
        The bad optimization return a graph that
        is not functionally equal to the original graph;
        When config.verify_correctness=False, wrong outputs
        will return
        """
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            return transform(gm).forward

        r1 = toy_example(i1, i2)
        with torchdynamo.optimize(incorrect_compile_fn):
            r2 = toy_example(i1, i2)
        self.assertTrue(not same(r1, r2))

    @unittest.skipIf(not has_onnxruntime(), "requires onnxruntime")
    @patch.object(config, "verify_correctness", True)
    def test_export(self):
        s = Seq()
        i = torch.randn(10)
        r1 = s(i)
        with torchdynamo.optimize_assert(offline_autotuner):
            r2 = s(i)
        self.assertTrue(same(r1, r2))

    @unittest.skipIf(not has_ipex(), "requires ipex")
    @patch.object(config, "verify_correctness", True)
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
    @patch.object(config, "verify_correctness", True)
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
