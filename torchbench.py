#!/usr/bin/env python
import argparse
import collections
import copy
import functools
import gc
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
from scipy.stats import ttest_ind, gmean

import torchdynamo
from torchdynamo import symbolic_convert
from torchdynamo.profiler import Profiler, fx_insert_profiling, ProfileMetrics
from torchdynamo.testing import same

os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
torchbench_dir = abspath(
    "../torchbench" if exists("../torchbench") else "../torchbenchmark"
)
assert os.path.exists(torchbench_dir)
os.chdir(torchbench_dir)
sys.path.append(torchbench_dir)
log = logging.getLogger(__name__)
SKIP = {
    # torchbench `get_model()` is broken these:
    "albert",
    "demucs",
    "hf_T5",
    "hf_Reformer",
    "hf_Longformer",
    "hf_GPT2",
    "hf_DistilBert",
    "hf_BigBird",
    "hf_Bert",
    "hf_Bart",
    "nvidia_deeprecommender",
    "hf_Albert",
    # TODO: need to debug a crash in this one on debug_insert_nops
    "pyhpc_isoneutral_mixing",
}
current_name = ""
current_device = ""


def nothing():
    pass


def synchronize():
    global synchronize
    if torch.cuda.is_available():
        synchronize = torch.cuda.synchronize
        synchronize()
    else:
        synchronize = nothing


def short_name(name, limit=20):
    """Truncate a model name to limit chars"""
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."


def iter_models(args):
    from torchbenchmark import list_models  # noqa

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


def format_speedup(speedup, pvalue, pvalue_threshold=0.1):
    if pvalue > pvalue_threshold:
        return f"{speedup:.3f}x SAME"
    return f"{speedup:.3f}x p={pvalue:.2f}"


def print_row(results):
    print(f"{current_device:4} {current_name:20} " + " ".join(map(str, results)))


class Stats:
    totals = collections.defaultdict(collections.Counter)

    @classmethod
    def reset_counters(cls):
        for k, v in symbolic_convert.counters.items():
            cls.totals[k].update(v)
        ok = symbolic_convert.counters["frames"]["ok"]
        total = symbolic_convert.counters["frames"]["total"]
        symbolic_convert.counters.clear()
        return ok, total

    @classmethod
    def print_summary(cls):
        for k, v in sorted(cls.totals.items()):
            lines = "\n  ".join(map(str, v.most_common(20)))
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
    result = format_speedup(speedup, pvalue)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", "-b", default="eager", help="baseline to normalize to"
    )
    parser.add_argument("--filter", "-k", action="append", help="filter benchmarks")
    parser.add_argument("--exclude", "-x", action="append", help="filter benchmarks")
    parser.add_argument("--devices", "-d", action="append", help="cpu or cuda")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    parser.add_argument("--threads", "-t", type=int, help="number of threads to use")
    parser.add_argument(
        "--min-measure-sec",
        type=float,
        default=0.1,
        help="floor of how long a timing run can take (introduces multiple calls to hit this)",
    )
    parser.add_argument(
        "--cpu-fusion", action="store_true", help="enable can_fuse_on_cpu"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="show errors")
    parser.add_argument(
        "--no-skip", action="store_true", help="run models that don't fx cleanly"
    )
    parser.add_argument(
        "--overhead", "-s", action="store_true", help="measure overheads"
    )
    args = parser.parse_args()

    args.verbose = True

    # defaults
    args.devices = args.devices or ["cpu"]
    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]

    if args.devices == ["cpu"]:
        global synchronize
        synchronize = nothing

    if args.no_skip:
        SKIP.clear()

    if args.cpu_fusion:
        torch._C._jit_override_can_fuse_on_cpu(True)

    if args.threads:
        torch.set_num_threads(args.threads)

    coverage_results = []
    speedups = []

    if args.overhead:
        optimize_ctx = torchdynamo.optimize(lambda gm: gm.forward)
        experiment = functools.partial(speedup_experiment, speedups, args)
    else:
        optimize_ctx = torchdynamo.optimize(fx_insert_profiling)
        experiment = functools.partial(coverage_experiment, coverage_results)

    for device, name, model, example_inputs in iter_models(args):
        torch.manual_seed(1337)
        correct_result = copy.deepcopy(model)(*example_inputs)
        torch.manual_seed(1337)
        with optimize_ctx:
            new_result = model(*example_inputs)
        if not same(correct_result, new_result):
            print_row(["INCORRECT"])
            continue
        ok, total = Stats.reset_counters()
        results = []

        # run one more time to see if we reached a fixed point
        with optimize_ctx:
            model(*example_inputs)
        _, frames_second_pass = Stats.reset_counters()  # should be 0
        results.append(f"{ok:3}/{total:3} frames (+{frames_second_pass:2}),")

        results.append(experiment(model, example_inputs))
        print_row(results)

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
        MEAN SPEEDUP {np.mean(speedups):.3f}x
        GEOMEAN SPEEDUP {gmean(speedups):.3f}x"""
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
