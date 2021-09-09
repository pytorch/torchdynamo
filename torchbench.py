#!/usr/bin/env python
import argparse
import collections
import copy
import gc
import logging
import os
import re
import sys
import time
import warnings
from os.path import abspath
from os.path import exists

import numpy as np
import torch
from scipy.stats import ttest_ind

import torchdynamo
from torchdynamo import symbolic_convert
from torchdynamo.profiler import Profiler, fx_insert_profiling, ProfileResult
from torchdynamo.testing import same

os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
torchbench_dir = abspath("../torchbench" if exists("../torchbench") else "../torchbenchmark")
assert os.path.exists(torchbench_dir)
os.chdir(torchbench_dir)
sys.path.append(torchbench_dir)
log = logging.getLogger(__name__)
SKIP = {
    # torchbench `get_model()` is broken these:
    "albert", "demucs", "hf_T5", "hf_Reformer", "hf_Longformer",
    "hf_GPT2", "hf_DistilBert", "hf_BigBird", "hf_Bert", "hf_Bart",
    "nvidia_deeprecommender", "hf_Albert",
    # TODO: need to debug a crash in this one on debug_insert_nops
    "pyhpc_isoneutral_mixing",
}
current_name = ""


def synchronize():
    global synchronize
    if torch.cuda.is_available():
        synchronize = torch.cuda.synchronize
        synchronize()
    else:
        synchronize = lambda: None


def short_name(name, limit=20):
    """ Truncate a model name to limit chars"""
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."


def iter_models(args):
    from torchbenchmark import list_models  # noqa
    for benchmark_cls in list_models():
        if (not re.search("|".join(args.filter), benchmark_cls.name, re.I) or
                re.search("|".join(args.exclude), benchmark_cls.name, re.I) or
                benchmark_cls.name in SKIP):
            continue
        for device in args.devices:
            try:
                benchmark = benchmark_cls(device=device, jit=False)
                model, example_inputs = benchmark.get_module()
                model.eval()
                gc.collect()
                global current_name
                current_name = short_name(benchmark.name)
                # print(current_name)
                yield device, current_name, model, example_inputs
            except NotImplementedError:
                pass
            except Exception:
                log.exception(f"misconfigured model {benchmark_cls.name}")


def timed(model, example_inputs, times=1):
    torch.manual_seed(1337)
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0


def measure_speedups(models, example_inputs, times, repeat):
    timings = np.zeros((repeat, len(models)), np.float64)
    for rep in range(repeat):
        # interleave the runs to handle frequency scaling and load changes
        for i in range(len(models)):
            if models[i] is not None:
                _, timings[rep, i] = timed(models[i], example_inputs, times)

    pvalues = [ttest_ind(timings[:, 0], timings[:, i])[1] for i in range(1, len(models))]
    timings = np.median(timings, axis=0)
    return timings[0] / timings[1:], pvalues


class ExperimentResult(object):
    pvalue_threshold = 0.1

    def __init__(self, model, ok):
        self.model = model
        self.ok = ok

    def format_speedup(self, speedup, pvalue):
        if self.ok == "OK":
            if pvalue > self.pvalue_threshold:
                return f"{speedup:.3f}x SAME"
            return f"{speedup:.3f}x p={pvalue:.2f}"
        return self.ok


def print_row(device, name, results, sec="sec"):
    print(f"{device:4} {name:20} " + " ".join(results))  # + f" -- {sec or -1:.1f}")


def insert_profiling(model, example_inputs):
    return torchdynamo.context(fx_insert_profiling)(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", "-b", default="eager",
                        help="baseline to normalize to")
    parser.add_argument("--filter", "-k", action="append",
                        help="filter benchmarks")
    parser.add_argument("--exclude", "-x", action="append",
                        help="filter benchmarks")
    parser.add_argument("--devices", "-d", action="append",
                        help="cpu or cuda")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warmup runs to do")
    parser.add_argument("--repeat", "-n", type=int, default=1,
                        help="number of timing runs")
    parser.add_argument("--threads", "-t", type=int,
                        help="number of threads to use")
    parser.add_argument("--min-measure-sec", type=float, default=0.1,
                        help="floor of how long a timing run can take (introduces multiple calls to hit this)")
    parser.add_argument("--cpu-fusion", action="store_true",
                        help="enable can_fuse_on_cpu")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="show errors")
    parser.add_argument("--no-skip", action="store_true",
                        help="run models that don't fx cleanly")
    args = parser.parse_args()

    args.verbose = True

    # defaults
    args.devices = args.devices or ["cpu"]
    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]

    if args.no_skip:
        SKIP.clear()

    if args.cpu_fusion:
        torch._C._jit_override_can_fuse_on_cpu(True)

    if args.threads:
        torch.set_num_threads(args.threads)

    totals = collections.defaultdict(collections.Counter)

    def reset_counters():
        for k, v in symbolic_convert.counters.items():
            totals[k].update(v)
        symbolic_convert.counters.clear()

    def check_correctness(fn):
        torch.manual_seed(1337)
        try:
            alt_model = fn(copy.deepcopy(original_model), example_inputs)
            if same(result, alt_model(*example_inputs)):
                return alt_model, "OK"
            return None, "INCORRECT"
        except Exception:
            if args.verbose:
                log.exception("error running fn.__name__")
            return None, "ERROR"  # f"{type(e).__name__}: {str(e)[:40]}"

    prof_totals = ProfileResult()
    for device, name, original_model, example_inputs in iter_models(args):
        try:
            t0 = time.time()
            model = copy.deepcopy(original_model)
            result, sec = timed(model, example_inputs, args.warmup)

            reset_counters()
            fn_model, fn_ok = check_correctness(insert_profiling)
            results = [fn_ok]

            ok = symbolic_convert.counters["frames"]["ok"]
            total = symbolic_convert.counters["frames"]["total"]
            reset_counters()

            profiler = Profiler()
            with profiler.prof:
                fn_model(*example_inputs)
            prof_results = profiler.results()
            prof_totals += prof_results

            frames_second_pass = symbolic_convert.counters["frames"]["total"]
            reset_counters()

            results.extend([
                f"{ok:2}/{total:2} frames (+{frames_second_pass:2}),",
                str(prof_results)
            ])

            print_row(device, name, results, time.time() - t0)

            # speedups, pvalues = measure_speedups([model] + [x.model for x in experiments],
            #                                      example_inputs,
            #                                      max(1, int(args.min_measure_sec / sec)),
            #                                      args.repeat)
            # if all(x.ok == "OK" for x in experiments):
            #     all_speedups.append(speedups)

            # print_row(device, name,
            #           [e.format_speedup(s, p)
            #            for e, s, p in zip(experiments, speedups, pvalues)], f"{time.time() - t0:.1f}s")
        except Exception:
            log.exception(f"ERROR from {name}")

    for k, v in sorted(totals.items()):
        lines = '\n  '.join(map(str, v.most_common(20)))
        print(f"STATS {k}\n  {lines}")

    print()
    print("PROFILE:", prof_totals)

    # print_row("", "GEOMEAN", map("{:.3f}x".format, gmean(np.vstack(all_speedups), axis=0)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
