#!/usr/bin/env pytest
import dataclasses
import importlib
import unittest

import torch
from torch import fx
from torchinductor import config
from torchinductor.lowering import GraphLowering

from torchdynamo.optimizations.python_key import python_key_normalize
from torchdynamo.testing import same

HAS_CUDA = False
if torch.cuda.is_available():
    try:
        importlib.import_module("triton")
        HAS_CUDA = True
    except ImportError:
        pass


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_debug = config.debug
        config.debug = True

    @classmethod
    def tearDownClass(cls):
        config.debug = cls.prior_debug


@dataclasses.dataclass
class InputGen:
    n: int
    device: str

    def dense(self):
        return torch.randn((self.n, self.n), device=self.device)

    def transposed(self):
        return self.dense().transpose(0, 1)

    def strided(self):
        return torch.randn((self.n * 2, self.n * 3), device=self.device)[
            self.n :, self.n :: 2
        ]

    def broadcast1(self):
        return torch.randn((self.n,), device=self.device)

    def broadcast2(self):
        return torch.randn((1, self.n, 1), device=self.device)

    def broadcast3(self):
        return torch.randn((1,), device=self.device)

    def double(self):
        return torch.randn((self.n, self.n), device=self.device, dtype=torch.double)

    def int(self):
        return torch.arange(self.n, device=self.device, dtype=torch.int32)


def check_model(self: TestCase, model, example_inputs):
    gm, wrap = python_key_normalize(fx.symbolic_trace(model), example_inputs)
    gm.graph.print_tabular()
    graph = GraphLowering(gm)
    wrap(graph.run)(*example_inputs)

    compiled_fn = graph.compile_to_fn()
    actual = wrap(compiled_fn)(*example_inputs)
    correct = model(*example_inputs)
    self.assertTrue(same(actual, correct))


def check_model_cuda(self: TestCase, model, example_inputs):
    if hasattr(model, "to"):
        model = model.to("cuda")
    example_inputs = tuple(x.to("cuda") for x in example_inputs)
    check_model(self, model, example_inputs)


class SweepInputs2:
    input_gen_types1 = [
        "dense",
        "transposed",
        "strided",
        "broadcast1",
        "broadcast2",
        "broadcast3",
        "double",
        "int",
    ]
    input_gen_types2 = input_gen_types1
    gen = None

    @staticmethod
    def kernel(a, b):
        return (a + b,)

    @classmethod
    def gen_template(cls, name1, name2):
        def test(self):
            check_model(
                self,
                cls.kernel,
                (
                    getattr(cls.gen, name1)(),
                    getattr(cls.gen, name2)(),
                ),
            )

        test.__name__ = f"test_{cls.gen.device}_{name1}_{name2}"
        setattr(cls, test.__name__, test)

    @classmethod
    def populate(cls):
        for name1 in cls.input_gen_types1:
            for name2 in cls.input_gen_types2:
                cls.gen_template(name1, name2)


class SweepInputsCpuTest(SweepInputs2, TestCase):
    gen = InputGen(10, "cpu")


SweepInputsCpuTest.populate()


class CommonTemplate:
    @classmethod
    def install(my_cls, other_cls, suffix):
        for name, value in my_cls.__dict__.items():
            if name.startswith("test_"):
                setattr(other_cls, f"{name}_{suffix}", value)

    def test_add_const_int(self):
        def fn(a):
            return (a + 1,)

        self.common(fn, (torch.randn(32),))

    def test_add_const_float(self):
        def fn(a):
            return (a + 1.5,)

        self.common(fn, (torch.randn(32),))

    def test_abs(self):
        def fn(a):
            return (a / (torch.abs(a) + 1),)

        self.common(fn, (torch.randn(17),))


class CpuTests(TestCase):
    common = check_model


CommonTemplate.install(CpuTests, "cpu")

if HAS_CUDA:

    class SweepInputsCudaTest(SweepInputs2, TestCase):
        gen = InputGen(10, "cuda")

    SweepInputsCudaTest.populate()

    class GpuTests(TestCase):
        common = check_model_cuda

    CommonTemplate.install(GpuTests, "cuda")
