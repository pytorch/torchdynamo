#!/usr/bin/env python
import argparse
import gc
import sys
import time

import numpy as np
import tabulate
import torch
import torchinductor
from torch import fx
from torch.cuda import synchronize
from torchinductor.lowering import GraphLowering

from torchdynamo.optimizations.python_key import python_key_normalize
from torchdynamo.testing import same

try:
    import tests.test_torchinductor as tti
except ImportError:
    tti = None


def timed(model, example_inputs, times=1):
    synchronize()
    gc.collect()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    # GC the result after timing
    assert result is not None
    return t1 - t0


def compute_speedups(args, models, example_inputs):
    expected = models[0](*example_inputs)
    for model in models[1:]:
        assert same(model(*example_inputs), expected)

    timings = np.zeros((args.repeat, len(models)), np.float64)
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        for m, model in enumerate(models):
            timings[rep, m] = timed(model, example_inputs)
    median = np.median(timings, axis=0)
    return (median[0] / median[1:]).tolist()


def microbenchmark(args, model, example_inputs):
    gm, wrap = python_key_normalize(fx.symbolic_trace(model), example_inputs)
    graph = GraphLowering(gm)
    wrap(graph.run)(*example_inputs)
    compiled_fn = graph.compile_to_fn()
    return compute_speedups(
        args,
        [model, torch.jit.trace(model, example_inputs), wrap(compiled_fn)],
        example_inputs,
    )


class MyModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
        )

    def forward(self, input):
        # return (self.model(input) + 1,)
        return (self.model(input),)


class MyModel2(torch.nn.Module):
    def forward(self, x, y):
        # return x / (torch.abs(x) + 1.0),
        return (x + y,)


class MicroBenchmarks:
    @staticmethod
    def add(a, b):
        return (a + b,)

    @staticmethod
    def scale(x, m, d):
        return ((x - m) / d,)

    @staticmethod
    def abs_norm(x):
        return (x / (torch.abs(x) + 1),)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude", "-x", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument("--devices", "-d", action="append", help="cpu or cuda")
    parser.add_argument("--size", "-s", action="append", help="cpu or cuda")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    parser.add_argument(
        "--threads", "-t", type=int, help="number of threads to use for eager"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="enable verbose debug printouts"
    )
    parser.add_argument(
        "--nvfuser", action="store_true", help="enable nvfuser globally"
    )
    args = parser.parse_args()

    # defaults
    args.devices = args.devices or ["cpu", "cuda"]
    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]
    args.size = args.size or [64, 256, 1024, 4096]

    if args.nvfuser:
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
    else:
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        if torch.cuda.is_available():
            torch._C._jit_set_nvfuser_enabled(False)

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.verbose:
        torchinductor.config.debug = True

    rows = []
    for model in (MicroBenchmarks.abs_norm,):
        for device in args.devices:
            for n in args.size:
                sys.stdout.write(f"{model.__name__:10} {device:4} {n:5} ")
                sys.stdout.flush()
                result = microbenchmark(
                    args,
                    model,
                    (torch.rand((n, n), device=device),),
                )
                rows.append([model.__name__, device, str(n)] + result)
                print(" ".join(f"{v:.2f}x" for v in result))

    print(
        tabulate.tabulate(
            rows,
            headers=[
                "model",
                "dev",
                "n",
                "ts",
                "inductor",
            ],
        )
    )


if __name__ == "__main__":
    main()
