#!/usr/bin/env pytest

import inspect
import unittest

import torch

import torchdynamo
import torchdynamo.testing

input = torch.ones([10, 10])
model = torch.nn.Sequential(*[torch.nn.Linear(10, 10) for _ in range(2)])
model(input).sum().backward()


def make_test(optim_cls, exp_frame_cnt=1, **kwargs):
    opt = optim_cls(model.parameters(), **kwargs)

    def test_fn(self):
        nonlocal opt
        counter = torchdynamo.testing.CompileCounter()

        with torchdynamo.optimize(counter):
            opt.step()

        self.assertEqual(counter.frame_count, exp_frame_cnt)

    return test_fn


class OptimizerTests(torchdynamo.testing.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # needed until pytorch assertion is changed to enable Adam
        # to be called with capturable=True
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torchdynamo.config, "capture_scalar_outputs", True
            )
        )
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torchdynamo.config, "raise_on_assertion_error", True
            )
        )

    test_sgd = make_test(torch.optim.SGD, lr=0.01)


# exclude SGD because it doesn't have proper default args
exclude = set(["SGD", "Optimizer", "SparseAdam"])
optimizers = [
    opt
    for opt in torch.optim.__dict__.values()
    if inspect.isclass(opt)
    and issubclass(opt, torch.optim.Optimizer)
    and opt.__name__ not in exclude
]


for opt in optimizers:
    setattr(OptimizerTests, "test_" + opt.__name__.lower(), make_test(opt))
