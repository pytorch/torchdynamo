import torchdynamo
from torchdynamo.testing import CompileCounter


class SkipNonTensorTests(torchdynamo.testing.TestCase):
    def test_add_tensor(self):
        def fn(a, b):
            return a + b

        counter = CompileCounter()
        with torchdynamo.optimize_assert(counter):
            import torch

            x = torch.randn(4)
            y = 5
            fn(x, y)

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
