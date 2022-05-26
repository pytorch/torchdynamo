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
            unittest.mock.patch.object(
                torchdynamo.config, "cache_size_limit", cls.cache_limit
            )
        )

    def test_loop_torture(self):
        def loop_torture(input, iters):
            out = input
            # randint itself causes one graph break
            for _ in range(iters):
                out += input
            return out

        compile_counter = torchdynamo.testing.CompileCounter()
        for _ in range(10):
            x = torch.randn(3)
            iters = torch.randint(low=0, high=1000, size=())
            with torchdynamo.optimize(compile_counter):
                loop_torture(x, iters)

        # Currently, we recompile each time,
        # We'd probably like to bail out quickly and warn
        # TODO(whc) these checks fail on py37.  Why?
        # self.assertEqual(counters["frames"]["total"], 2 + self.cache_limit)
        # self.assertEqual(counters["frames"]["ok"], 1 + self.cache_limit)

        # compile_counter only sees frames that were fed to the backend compiler,
        # which is a subset of counters["frames"]["ok"] -- probably becuase
        # counters["frames"]["ok"] includes frames not containing torch ops?
        self.assertEqual(compile_counter.frame_count, self.cache_limit)

    def test_dynamic_input(self):
        def model(input):
            return input + input

        expected_recompiles = 2
        compile_counter = torchdynamo.testing.CompileCounter()
        with unittest.mock.patch.object(
            torchdynamo.config, "cache_size_limit", expected_recompiles
        ):
            with self.assertLogs(level="WARNING") as logs:
                for _ in range(10):
                    bsz = torch.randint(low=0, high=1000, size=())
                    x = torch.randn((bsz, 3, 4))
                    with torchdynamo.optimize(compile_counter):
                        model(x)

        self.assertEqual(compile_counter.frame_count, expected_recompiles)
        self.assertEqual(len(logs.records), 1)
        print(logs.records[0])
        self.assertTrue(
            logs.records[0]
            .getMessage()
            .startswith("torchdynamo hit recompilation cache limit")
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_nvfuser_guards(self):
        # we may want to model dynamo's guards sufficiently after nvfuser's ProfilingExecutor guards
        # such that we ensure dynamo is in charge of all the recompilations at the top level,
        # and we could thus simplfy the underlying torchscript executor
        def func(a, b, c):
            return a + b * c

        a = torch.rand(3, 4, 5, device="cuda")
        b = torch.rand(3, 4, 5, device="cuda")
        b_v = torch.rand(3, 5, 4, device="cuda").view(3, 4, 5)
        b_p = torch.rand(3, 5, 4, device="cuda").permute(0, 2, 1)
        c = torch.rand(3, 4, 5, device="cuda")
        compile_counter = torchdynamo.testing.CompileCounter()

        with unittest.mock.patch.object(torchdynamo.config, "cache_size_limit", 2):
            with torchdynamo.optimize(compile_counter):
                func(a, b, c)  # warmup
                self.assertEqual(compile_counter.frame_count, 1)

                func(a, b, c)  # no guard fail or recompile
                self.assertEqual(compile_counter.frame_count, 1)

                func(a, b_v, c)  # a view should not cause nvfuser recompile
                self.assertEqual(compile_counter.frame_count, 1)

                func(a, b_p, c)  # a permutation should cause recompile
                self.assertEqual(compile_counter.frame_count, 2)
