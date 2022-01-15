#!/usr/bin/env python
import argparse
import collections
import copy
import csv
import functools
import gc
import io
import itertools
import logging
import os
import re
import sys
import textwrap
import time
import warnings
from os.path import abspath
from os.path import exists

import numpy as np
import torch
from scipy.stats import gmean
from scipy.stats import ttest_ind

import torchdynamo.utils
from torchdynamo.optimizations import backends
from torchdynamo.optimizations.inference import user_compiler
from torchdynamo.profiler import fx_insert_profiling
from torchdynamo.profiler import ProfileMetrics
from torchdynamo.profiler import Profiler
from torchdynamo.testing import dummy_fx_compile
from torchdynamo.testing import format_speedup
from torchdynamo.testing import same
import torchdynamo

os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
torchbench_dir = abspath(
    "../torchbench" if exists("../torchbench") else "../torchbenchmark"
)
assert os.path.exists(torchbench_dir)
os.chdir(torchbench_dir)
sys.path.append(torchbench_dir)
log = logging.getLogger(__name__)
SKIP = {
    # non-deterministic output / cant check correctness
    "pyhpc_turbulent_kinetic_energy",
}
current_name = ""
current_device = ""


class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def null_print(*args):
    pass


def synchronize():
    pass


def short_name(name, limit=20):
    """Truncate a model name to limit chars"""
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."


def global_fixes():
    from fastNLP.core import logger
    from pycocotools import coco

    # silence some spam
    coco.print = null_print
    logger.setLevel(logging.WARNING)

    # TODO(jansel): remove when https://github.com/pytorch/pytorch/pull/70459 lands
    torch.nn.modules.utils._pair.__name__ = "_pair"


def iter_models(args):
    from torchbenchmark import list_models  # noqa

    global_fixes()

    for benchmark_cls in list_models():
        if (
            not re.search("|".join(args.filter), benchmark_cls.name, re.I)
            or re.search("|".join(args.exclude), benchmark_cls.name, re.I)
            or benchmark_cls.name in SKIP
        ):
            continue
        for device in args.devices:
            try:
                benchmark = benchmark_cls(device=device, jit=False)
                model, example_inputs = benchmark.get_module()
                model.eval()
                gc.collect()
                global current_name, current_device
                current_device = device
                current_name = short_name(benchmark.name)
                yield device, current_name, model, example_inputs
            except NotImplementedError:
                pass
            except Exception:
                log.exception(f"misconfigured model {benchmark_cls.name}")


def timed(model, example_inputs, times=1):
    synchronize()
    gc.collect()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0


class Stats:
    totals = collections.defaultdict(collections.Counter)

    @classmethod
    def reset_counters(cls):
        for k, v in torchdynamo.utils.counters.items():
            cls.totals[k].update(v)
        ok = torchdynamo.utils.counters["frames"]["ok"]
        total = torchdynamo.utils.counters["frames"]["total"]
        torchdynamo.utils.counters.clear()
        return ok, total

    @classmethod
    def print_summary(cls):
        for k, v in sorted(cls.totals.items()):
            lines = "\n  ".join(map(str, v.most_common(50)))
            print(f"STATS {k}\n  {lines}")


def coverage_experiment(coverage_results, model, example_inputs):
    profiler = Profiler()
    with profiler.prof, torchdynamo.run():
        model(*example_inputs)
    coverage_result = profiler.results()
    coverage_results.append(coverage_result.percent())
    return coverage_result


def speedup_experiment(speedups, args, model, example_inputs):
    timings = np.zeros((args.repeat, 2), np.float64)
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        _, timings[rep, 0] = timed(model, example_inputs)
        with torchdynamo.run():
            _, timings[rep, 1] = timed(model, example_inputs)
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    speedups.append(speedup)
    output_csv(
        "speedups.csv",
        ("dev", "name", "speedup"),
    ).writerow([current_device, current_name, f"{speedup:.4f}"])
    return format_speedup(speedup, pvalue)


@functools.lru_cache(1)
def output_csv(name, headers):
    output = csv.writer(
        io.TextIOWrapper(
            open(os.path.join(torchdynamo.config.base_dir, name), "wb", buffering=0),
            "utf-8",
            write_through=True,
        ),
        lineterminator="\n",
    )
    output.writerow(headers)
    return output


def baselines(models, example_inputs, args, speedups, filaname="baselines"):
    models = list(models)
    for idx, (name, model) in enumerate(models):
        if idx == 0:
            result0, _ = timed(model, example_inputs)
        elif model is not None:
            try:
                result, _ = timed(model, example_inputs)
                if same(result0, result):
                    continue
                print(name, "is INCORRECT")
            except Exception:
                log.exception("error checking %s", name)
            models[idx] = (name, None)
    timings = np.zeros((args.repeat, len(models)), np.float64)
    timings.fill(1.0e10)
    for rep in range(args.repeat):
        for idx, (name, model) in enumerate(models):
            if model is not None:
                _, timings[rep, idx] = timed(model, example_inputs)
    pvalue = [
        ttest_ind(timings[:, 0], timings[:, i]).pvalue
        for i in range(1, timings.shape[1])
    ]
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1:]
    for idx, (name, model) in enumerate(models[1:]):
        if model is None:
            speedup[idx] = 0.0
    speedups.append(speedup)
    result = " ".join(
        [
            format_speedup(s, p, m is not None)
            for s, p, m in zip(speedup, pvalue, [m for n, m in models[1:]])
        ]
    )
    output_csv(
        f"{filaname}.csv",
        ("dev", "name") + tuple(n for n, m in models[1:]),
    ).writerow([current_device, current_name] + [f"{x:.4f}" for x in speedup])
    return result


def try_script(model, example_inputs):
    try:
        return torch.jit.script(model)
    except Exception:
        return None


def speedup_experiment_ts(speedups, args, model, example_inputs):
    return baselines(
        [
            ("eager", model),
            ("ts", try_script(model, example_inputs)),
            (
                "ofi",
                backends.ofi(try_script(model, example_inputs), example_inputs),
            ),
            # ("nnc", backends.nnc(try_script(model, example_inputs), example_inputs)),
            # ("nvfuser", backends.nvfuser(try_script(model, example_inputs), example_inputs)),
        ],
        example_inputs,
        args,
        speedups,
        "baseline_ts",
    )


def speedup_experiment_sr(speedups, args, model, example_inputs):
    if current_name not in ("opacus_cifar10", "timm_nfnet", "hf_T5"):
        sr = backends.static_runtime(try_script(model, example_inputs), example_inputs)
    else:
        # segfaults on these models
        sr = None
    return baselines(
        [
            ("eager", model),
            (
                "sr",
                sr,
            ),
        ],
        example_inputs,
        args,
        speedups,
        "baseline_sr",
    )


def speedup_experiment_onnx(speedups, args, model, example_inputs):
    if current_device == "cpu":
        m_onnxrt = backends.onnxrt_cpu(
            try_script(model, example_inputs), example_inputs
        )
    else:
        m_onnxrt = backends.onnxrt_cuda(
            try_script(model, example_inputs), example_inputs
        )

    if current_name != "timm_resnest":
        m_onnx2tf = backends.onnx2tf(try_script(model, example_inputs), example_inputs)
    else:
        # this one takes 8+ hours to finish
        m_onnx2tf = None

    return baselines(
        [
            ("eager", model),
            ("onnxrt", m_onnxrt),
            ("onnx2tf", m_onnx2tf),
        ],
        example_inputs,
        args,
        speedups,
        "baseline_onnx",
    )


def speedup_experiment_trt(speedups, args, model, example_inputs):
    m_onnx2trt = backends.onnx2tensorrt(
        try_script(model, example_inputs), example_inputs
    )

    m_torch2trt = backends.torch2trt(model, example_inputs)

    if current_name != "opacus_cifar10":
        m_fx2trt = backends.fx2trt(model, example_inputs)
    else:
        # fx2trt infinite loops on one model
        m_fx2trt = None

    return baselines(
        [
            ("eager", model),
            ("onnx2trt", m_onnx2trt),
            ("torch2trt", m_torch2trt),
            ("fx2trt", m_fx2trt),
        ],
        example_inputs,
        args,
        speedups,
        "baseline_trt",
    )


def null_experiment(model, example_inputs):
    return []


def pick_grad(name):
    if name in ("maml",):
        return torch.enable_grad()
    else:
        return torch.no_grad()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", "-k", action="append", help="filter benchmarks")
    parser.add_argument("--exclude", "-x", action="append", help="filter benchmarks")
    parser.add_argument("--devices", "-d", action="append", help="cpu or cuda")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    parser.add_argument("--threads", "-t", type=int, help="number of threads to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="show errors")
    parser.add_argument(
        "--no-skip", action="store_true", help="run models that don't fx cleanly"
    )
    parser.add_argument("--overhead", action="store_true", help="measure overheads")
    parser.add_argument(
        "--speedup", action="store_true", help="measure speedup with default passes"
    )
    parser.add_argument(
        "--speedup-ts",
        action="store_true",
    )
    parser.add_argument(
        "--speedup-sr",
        action="store_true",
    )
    parser.add_argument(
        "--speedup-onnx",
        action="store_true",
    )
    parser.add_argument(
        "--speedup-trt",
        action="store_true",
    )
    parser.add_argument(
        "--nvfuser",
        action="store_true",
    )
    parser.add_argument(
        "--nothing", action="store_true", help="just check the benchmark works"
    )
    parser.add_argument(
        "--nops", action="store_true", help="check bytecode rewriting works"
    )
    parser.add_argument("--minimum-call-count", type=int)
    args = parser.parse_args()

    # defaults
    args.devices = args.devices or ["cpu"]
    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]

    if args.devices != ["cpu"] and torch.cuda.is_available():
        global synchronize
        synchronize = torch.cuda.synchronize

    if args.no_skip:
        SKIP.clear()

    if args.nvfuser:
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
    else:
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.verbose:
        torchdynamo.config.debug = True

    coverage_results = []
    speedups = []
    experiment = null_experiment
    optimize_ctx = NullContext()

    if args.overhead:
        optimize_ctx = torchdynamo.optimize(dummy_fx_compile)
        experiment = functools.partial(speedup_experiment, speedups, args)
    elif args.speedup:
        optimize_ctx = torchdynamo.optimize(user_compiler)
        experiment = functools.partial(speedup_experiment, speedups, args)
    elif args.speedup_ts:
        experiment = functools.partial(speedup_experiment_ts, speedups, args)
    elif args.speedup_sr:
        experiment = functools.partial(speedup_experiment_sr, speedups, args)
    elif args.speedup_onnx:
        experiment = functools.partial(speedup_experiment_onnx, speedups, args)
    elif args.speedup_trt:
        experiment = functools.partial(speedup_experiment_trt, speedups, args)
    elif args.nothing:
        pass
    elif args.nops:
        optimize_ctx = torchdynamo.eval_frame.optimize(
            torchdynamo.testing.debug_insert_nops
        )
    else:
        optimize_ctx = torchdynamo.optimize(fx_insert_profiling)
        experiment = functools.partial(coverage_experiment, coverage_results)

    if args.minimum_call_count:
        torchdynamo.config.minimum_call_count = args.minimum_call_count

    for device, name, model, example_inputs in iter_models(args):
        with pick_grad(name):
            sys.stdout.write(f"{current_device:4} {current_name:20} ")
            sys.stdout.flush()
            for submod in itertools.chain([model], model.modules()):
                assert not torchdynamo.utils.is_jit_model(submod)
            torch.manual_seed(1337)
            correct_result = copy.deepcopy(model)(*example_inputs)
            torch.manual_seed(1337)
            torchdynamo.reset()
            try:
                with optimize_ctx:
                    new_result = model(*example_inputs)
            except Exception:
                logging.exception("unhandled error")
                print("ERROR")
                continue
            if current_name == "pyhpc_turbulent_k...":
                # This model has non-deterministic output so we cant
                # check correctness.
                # TODO(jansel): submit upstream fix for this
                pass
            elif not same(correct_result, new_result):
                print("INCORRECT")
                continue
            ok, total = Stats.reset_counters()
            results = []

            # run one more time to see if we reached a fixed point
            with optimize_ctx:
                model(*example_inputs)
            _, frames_second_pass = Stats.reset_counters()  # should be 0
            results.append(f"{ok:3}/{total:3} frames (+{frames_second_pass:2}),")

            results.append(experiment(model, example_inputs))
            print(" ".join(map(str, results)))
            del model, example_inputs, correct_result, new_result
            torchdynamo.reset()
            gc.collect()

    Stats.print_summary()
    if coverage_results:
        print(
            "\nMEAN COVERAGE:",
            functools.reduce(ProfileMetrics.__add__, coverage_results)
            / len(coverage_results),
        )
    if speedups:
        print(
            textwrap.dedent(
                f"""
                MEAN SPEEDUP {np.mean(speedups, axis=0)}
                GEOMEAN SPEEDUP {gmean(speedups, axis=0)}"""
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
