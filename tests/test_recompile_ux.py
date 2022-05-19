from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List

import torch

import torchdynamo
import torchdynamo.config
import torchdynamo.testing

torchdynamo.config.debug = False
from torchdynamo.utils import counters

class RecompileUxTests(torchdynamo.testing.TestCase):
    # TODO(whc) dynamo actualy recompiles one more time than the cache limit
    max_recompiles = 2

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torchdynamo.config.cache_size_limit = max(1, RecompileUxTests.max_recompiles - 1)

    def test_loop_torture(self):
        def loop_torture(input):
            out = input
            for i in range(torch.randint(low=0, high=1000, size=())):
                out += input
            return out

        for i in range(10):
            x = torch.randn(3)
            with torchdynamo.optimize("eager"):
                out = loop_torture(x)

        # Currently, we recompile each time,
        # We'd probably like to bail out quickly and warn
        self.assertEqual(counters["frames"]["total"], RecompileUxTests.max_recompiles)
        self.assertEqual(
            torchdynamo.utils.counters["frames"]["total"],
            torchdynamo.utils.counters["frames"]["ok"],
        )

    def test_dynamic_input(self):
        def model(input):
            return input + input

        for i in range(10):
            bsz = torch.randint(low=0, high=1000, size=())
            x = torch.randn((bsz, 3, 4))
            with torchdynamo.optimize("eager"):
                out = model(x)
        
        print(counters)
        self.assertEqual(counters["frames"]["total"], TortureTests.max_recompiles)
        self.assertEqual(
            torchdynamo.utils.counters["frames"]["total"],
            torchdynamo.utils.counters["frames"]["ok"],
        )

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
        
        torchdynamo.config.cache_size_limit = 5
        
        with torchdynamo.optimize("eager"):
            func(a, b, c) # warmup
            self.assertEqual(counters["frames"]["total"], 1)
            func(a, b, c) # no guard fail
            self.assertEqual(counters["frames"]["total"], 1)
            func(a, b_v, c) # should NOT fail?
            self.assertEqual(counters["frames"]["total"], 1)
            func(a, b_p, c) # should fail
            self.assertEqual(counters["frames"]["total"], 2)
            func(torch.rand(5), torch.rand(5), torch.rand(5))
            self.assertEqual(counters["frames"]["total"], 2)
            self.assertEqual(
                torchdynamo.utils.counters["frames"]["total"],
                torchdynamo.utils.counters["frames"]["ok"],
            )
        
        torchdynamo.config.cache_size_limit = RecompileUxTests.max_recompiles - 1