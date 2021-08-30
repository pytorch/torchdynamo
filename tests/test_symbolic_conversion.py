#!/usr/bin/env pytest
import inspect

from torchdynamo import optimizer
import unittest
import torch
from torchdynamo.testing import same
from torchdynamo.symbolic_convert import convert_frame_assert


def add(a, b):
    return a + b


def make_test(fn):
    def test_fn(self):
        args = [torch.randn(10) for _ in inspect.signature(fn).parameters]
        correct = fn(*args)
        with optimizer.context(convert_frame_assert):
            val1 = fn(*args)
            val2 = fn(*args)
        self.assertTrue(same(val1, correct))
        self.assertTrue(same(val2, correct))

    return test_fn


class SymblicConversionTests(unittest.TestCase):
    test_add = make_test(add)
