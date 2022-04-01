#!/usr/bin/env pytest
import dataclasses
import unittest

import torch
from torch import fx
from torchinductor.lowering import GraphLowering

from torchdynamo.optimizations.python_key import python_key_normalize
from torchdynamo.testing import same

HAS_CUDA = False
if torch.cuda.is_available():
    try:
        import triton

        HAS_CUDA = True
    except ImportError:
        pass


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
        return torch.randn((self.n, 1), device=self.device)

    def broadcast3(self):
        return torch.randn((1,), device=self.device)

    def double(self):
        return torch.randn((self.n, self.n), device=self.device, dtype=torch.double)

    def int(self):
        return torch.arange(self.n, device=self.device, dtype=torch.int32)


def check_model(self: unittest.TestCase, model, example_inputs):
    gm, wrap = python_key_normalize(fx.symbolic_trace(model), example_inputs)
    graph = GraphLowering(gm)
    wrap(graph.run)(*example_inputs)

    compiled_fn = graph.compile_to_fn()
    actual = wrap(compiled_fn)(*example_inputs)
    correct = model(*example_inputs)
    self.assertTrue(same(actual, correct))


class SweepInputs2:
    input_gen_types = [
        "dense",
        "transposed",
        "strided",
        "broadcast1",
        "broadcast2",
        "broadcast3",
        "double",
        "int",
    ]
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
        for name1 in cls.input_gen_types:
            for name2 in cls.input_gen_types:
                cls.gen_template(name1, name2)


class SweepInputsCpuTest(SweepInputs2, unittest.TestCase):
    gen = InputGen(10, "cpu")


SweepInputsCpuTest.populate()

if HAS_CUDA:

    class SweepInputsCudaTest(SweepInputs2, unittest.TestCase):
        gen = InputGen(10, "cuda")

    SweepInputsCudaTest.populate()
