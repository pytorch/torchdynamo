#!/usr/bin/env python
import collections
import copy
import gc
import importlib
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from towhee.compiler import jit_compile

import torchdynamo
from torchdynamo.testing import same

# We are primarily interested in tf32 datatype
torch.backends.cuda.matmul.allow_tf32 = True


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

    @classmethod
    def aot_summary(cls):
        return [cls.totals["aot_autograd"]["total"], cls.totals["aot_autograd"]["ok"]]


def load_model(args):
    module = importlib.import_module(f"torchbenchmark.models.{args.model_name}")
    benchmark_cls = getattr(module, "Model", None)
    if not hasattr(benchmark_cls, "name"):
        benchmark_cls.name = args.model_name
    batch_size = None

    benchmark = benchmark_cls(
        test="eval", device=args.device, jit=False, batch_size=batch_size
    )
    model, example_inputs = benchmark.get_module()

    model.eval()
    gc.collect()
    return benchmark.name, model, example_inputs


def get_tolerance():
    return 1e-4


def timed(model, example_inputs, repeat=1):
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    # Dont collect outputs to correctly measure timing
    for _ in range(repeat):
        model(*example_inputs)
    t1 = time.perf_counter()
    return t1 - t0


def run_one_model(
    args,
    model,
    example_inputs,
    optimize_ctx,
):
    t0 = time.perf_counter()
    tolerance = get_tolerance()
    with torch.no_grad():
        sys.stdout.write(f"{args.device} {args.model_name} ")
        sys.stdout.flush()

        torch.manual_seed(1337)
        correct_result = copy.deepcopy(model)(
            *torchdynamo.utils.clone_inputs(example_inputs)
        )
        torchdynamo.reset()

        try:
            with optimize_ctx:
                new_result = model(*example_inputs)
        except Exception:
            logging.exception("unhandled error")
            print("ERROR")
            return sys.exit(-1)
        if not same(correct_result, new_result, False, tolerance):
            print("INCORRECT")
        ok, total = Stats.reset_counters()
        results = []

        # run one more time to see if we reached a fixed point
        with optimize_ctx:
            model(*example_inputs)
        _, frames_second_pass = Stats.reset_counters()  # should be 0

        if frames_second_pass > 0:
            with optimize_ctx:
                model(*example_inputs)
            _, frames_third_pass = Stats.reset_counters()  # should be 0
        else:
            frames_third_pass = 0

        results.append(
            f"{ok:3}/{total:3} +{frames_third_pass} frames {time.perf_counter()-t0:3.0f}s"
        )

        report = []
        for _ in range(args.round):
            inputs = example_input
            t = timed(model, inputs, args.repeat)
            report.append(
                dict(
                    name=args.model_name,
                    device=args.device,
                    experiment="no_jit",
                    time=t,
                )
            )
        for _ in range(args.round):
            inputs = example_input
            with torchdynamo.run():
                t = timed(model, inputs, args.repeat)
            report.append(
                dict(name=args.model_name, device=args.device, experiment="jit", time=t)
            )
        return pd.DataFrame(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("towhee torch benchmark")
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-m", "--model_name", type=str)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-b", "--backend", type=str, default="nebullvm")
    parser.add_argument("-r", "--repeat", type=int, default=5)
    parser.add_argument("-R", "--round", type=int, default=5)

    args = parser.parse_args()

    original_dir = os.path.abspath(os.getcwd())
    torchbench_dir = os.path.abspath(args.path)
    os.chdir(torchbench_dir)
    sys.path.append(torchbench_dir)

    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")

    if isinstance(args.model_name, str):
        if "," in args.model_name:
            args.model_name = args.model_name.split(",")
        else:
            args.model_name = [args.model_name]
    if args.model_name is None:
        from torchbenchmark import _list_model_paths

        args.model_name = [
            os.path.basename(model_path) for model_path in _list_model_paths()
        ]

    model_list = copy.deepcopy(args.model_name)
    print(f"trying model list: {model_list}")
    for model_name in model_list:
        print(f"trying model: {model_name}")
        args.model_name = model_name

        optimize_ctx = jit_compile(backend=args.backend, perf_loss_ths=0.001)
        name, model, example_input = load_model(args)
        print(name, model, example_input)
        # import ipdb
        # ipdb.set_trace()
        print(run_one_model(args, model, example_input, optimize_ctx))
