from torchdynamo.utils import counters
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
import unittest

import torch

import torchdynamo
import torchdynamo.config
import torchdynamo.testing

torchdynamo.config.debug = False


class RecompileUxTests(torchdynamo.testing.TestCase):
    # TODO(whc) dynamo actualy recompiles one more time than the cache limit
    cache_limit = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(torchdynamo.config, "cache_size_limit",
                                       cls.cache_limit))

    def test_loop_torture(self):
        def loop_torture(input, iters):
            out = input
            # randint itself causes one graph break
            for i in range(iters):
                # out += input should be another graph
                # what about the for loop?
                out += input
            return out

        for i in range(10):
            x = torch.randn(3)
            iters = torch.randint(low=0, high=1000, size=())
            with torchdynamo.optimize("eager"):
                out = loop_torture(x, iters)

        # Currently, we recompile each time,
        # We'd probably like to bail out quickly and warn
        self.assertEqual(counters["frames"]["total"], 2 + self.cache_limit)
        self.assertEqual(counters["frames"]["ok"], 1 + self.cache_limit)

    def test_dynamic_input(self):
        def model(input):
            return input + input

        for i in range(10):
            bsz = torch.randint(low=0, high=1000, size=())
            x = torch.randn((bsz, 3, 4))
            with torchdynamo.optimize("eager"):
                out = model(x)

        print(counters)
        self.assertEqual(counters["frames"]["ok"], self.cache_limit)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_nvfuser_guards(self):
        # we may want to model dynamo's guards sufficiently after nvfuser's ProfilingExecutor guards
        # such that we ensure dynamo is in charge of all the recompilations at the top level,
        # and we could thus simplfy the underlying torchscript executor
        def func(a, b, c):
            return a + b * c

        a = torch.rand(3, 4, 5, device='cuda')
        b = torch.rand(3, 4, 5, device='cuda')
        b_v = torch.rand(3, 5, 4, device='cuda').view(3, 4, 5)
        b_p = torch.rand(3, 5, 4, device='cuda').permute(0, 2, 1)
        c = torch.rand(3, 4, 5, device='cuda')

        with torchdynamo.optimize("eager"):
            func(a, b, c)  # warmup
            self.assertEqual(counters["frames"]["total"], 1)
            self.assertEqual(counters["frames"]["ok"], 1)
            
            func(a, b, c)  # no guard fail or recompile
            self.assertEqual(counters["frames"]["total"], 1)
            self.assertEqual(counters["frames"]["ok"], 1)

            func(a, b_v, c)  # a view should not cause nvfuser recompile
            self.assertEqual(counters["frames"]["total"], 1)
            self.assertEqual(counters["frames"]["ok"], 1)

            func(a, b_p, c)  # a permutation should cause recompile
            self.assertEqual(counters["frames"]["total"], 2)
            self.assertEqual(counters["frames"]["ok"], 1)

            func(torch.rand(5), torch.rand(5), torch.rand(5))
            self.assertEqual(counters["frames"]["total"], 2)
            self.assertEqual(counters["frames"]["ok"], 1)

