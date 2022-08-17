#!/usr/bin/env python
import click
import numpy as np
import torch
import triton
from operator_inp_utils import OperatorInputsLoader

import torchinductor
from torchdynamo.optimizations.backends import cudagraphs_inner
from torchdynamo.testing import same
from torchinductor.compile_fx import compile_fx
from torchinductor.utils import gen_gm_and_inputs

aten = torch.ops.aten


def compute_speedups(repeats, models, example_inputs):
    expected = models[0](*example_inputs)
    for model in models[1:]:
        actual = model(*example_inputs)
        assert same(actual, expected), expected[0] - actual[0]

    timings = np.zeros((repeats, len(models)), np.float64)
    for rep in range(repeats):
        # interleave the runs to handle frequency scaling and load changes
        for m, model in enumerate(models):
            # do_bench() clears L2 cache to hide the latency of CPU launch time
            # along with cuda synchronization
            median_ms, _, _ = triton.testing.do_bench(lambda: model(*example_inputs))
            timings[rep, m] = median_ms
    return np.median(timings, axis=0)


def microbenchmark(target, args, kwargs, dtype):
    gm, gm_args = gen_gm_and_inputs(target, args, kwargs)
    compiled_fn = compile_fx(gm, gm_args)
    cudagraphs_eager = cudagraphs_inner(gm, gm_args, copy_outputs=False)
    cudagraphs_jit = cudagraphs_inner(
        torch.jit.trace(gm, gm_args), gm_args, copy_outputs=False
    )

    repeats = 3
    medians = compute_speedups(
        repeats,
        [cudagraphs_eager, cudagraphs_jit, compiled_fn],
        gm_args,
    )

    print(f"Perf for {target} {dtype} w/cudagraphs")
    print(f"JIT NVFuser speedup over aten {medians[0]/medians[1]}")
    print(f"Inductor speedup over aten {medians[1]/medians[2]}")


@click.command()
@click.option(
    "--suite",
    help="suite to load inps from: options: timm, huggingface, torchbench",
    default="torchbench",
)
@click.option("--op", help="operator overload to benchmark")
@click.option("--dtype", help="dtype to benchmark")
def benchmark(suite, op, dtype):
    assert suite in ("timm", "huggingface", "torchbench"), f"got {suite}"
    if suite == "timm":
        loader = OperatorInputsLoader.get_timm_loader()
    elif suite == "huggingface":
        loader = OperatorInputsLoader.get_huggingface_loader()
    else:
        loader = OperatorInputsLoader.get_torchbench_loader()

    assert dtype in ("float16", "float32"), f"got {dtype}"
    dtype = torch.float16 if dtype == "float16" else torch.float32

    operator = eval(op)

    # TODO - add more testing options / sweep over multiple inputs
    # sweep for underperforming kernels

    inp_gen = loader.get_inputs_for_operator(operator, dtype=torch.float)
    for _ in range(1):
        args, kwargs = next(inp_gen)
        microbenchmark(operator, args, kwargs, dtype)


if __name__ == "__main__":
    # TODO: getting error w/aot_autograd in compile fx
    torchinductor.config.aot_autograd = False
    benchmark()
