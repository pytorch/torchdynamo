import unittest

import torch

import torchdynamo
import torchdynamo.testing
from torchdynamo.testing import same


def func(a):
    return a.softmax(dim=1) + a.sum()


@torchdynamo.optimize()
def func_dynamo(a):
    return a.softmax(dim=1) + a.sum()


class TestMultipleProcess(torchdynamo.testing.TestCase):
    def test_multiple(self):
        a = torch.rand([100, 100])
        expected_result = func(a)
        dynamo_result = func_dynamo(a)

        self.assertTrue(same(expected_result, dynamo_result))


if __name__ == "__main__":
    unittest.main()
