import torch

import torchdynamo
from torchdynamo.testing import CompileCounter


class SkipNonTensorTests(torchdynamo.testing.TestCase):
    def test_add_tensor1(self):
        def fn(a, b):
            return a + b

        counter = CompileCounter()
        with torchdynamo.optimize_assert(counter):

            x = torch.randn(4)
            y = 5
            fn(x, y)

        assert counter.op_count == 1

    def test_add_tensor2(self):
        def fn(a, b):
            return torch.add(a, b)

        counter = CompileCounter()
        with torchdynamo.optimize_assert(counter):

            x = torch.randn(4)
            y = 5
            fn(x, y)

        assert counter.op_count == 1

    def test_add_tensor_list(self):
        def fn(lst):
            return lst[0] + lst[1]

        counter = CompileCounter()
        with torchdynamo.optimize_assert(counter):
            x = torch.randn(4)
            y = 5
            fn([x, y])

        assert counter.op_count == 1

    def test_add_tensor_dict(self):
        def fn(dt):
            return dt["a"] + dt["b"]

        counter = CompileCounter()
        with torchdynamo.optimize_assert(counter):
            x = torch.randn(4)
            y = 5
            fn({"a": x, "b": y})

        assert counter.op_count == 1

    def test_add_skip(self):
        def fn(a, b):
            return a + b

        counter = CompileCounter()
        with torchdynamo.optimize_assert(counter):
            x = 4
            y = 5
            fn(x, y)

        assert counter.op_count == 0
