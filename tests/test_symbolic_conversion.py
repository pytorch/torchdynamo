#!/usr/bin/env pytest
import inspect
import unittest

import torch
from torch import sub

from torchdynamo import eval_frame
from torchdynamo.symbolic_convert import convert_frame_assert
from torchdynamo.testing import same

d = torch.ones(10, 10)


def add(a, b):
    return a + b


def constant(a, b, c):
    return a - b * c + 1.0


def globalvar(a, b):
    return a - b + d


def globalfn(a, b):
    return sub(a, b)


def make_test(fn):
    def test_fn(self):
        args = [torch.randn(10, 10) for _ in inspect.signature(fn).parameters]
        correct = fn(*args)
        with eval_frame.context(convert_frame_assert):
            val1 = fn(*args)
            val2 = fn(*args)
        self.assertTrue(same(val1, correct))
        self.assertTrue(same(val2, correct))

    return test_fn


class SymblicConversionTests(unittest.TestCase):
    test_add = make_test(add)
    test_constant = make_test(constant)
    test_globalvar = make_test(globalvar)
    # test_globalfn = make_test(globalfn)
