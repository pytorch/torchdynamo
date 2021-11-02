#!/usr/bin/env python
import argparse
import gc
import importlib
import itertools
import json
import logging
import math
import operator
import os
import queue
import re
import time
import warnings
import multiprocessing
from collections import OrderedDict

import numpy as np
import torch
from scipy.stats import ttest_ind
from tabulate import tabulate

from torchdynamo import config
from torchdynamo.optimizations.backends import onnxrt
from torchdynamo.optimizations.backends import optimize_for_inference
from torchdynamo.optimizations.backends import static_runtime
from torchdynamo.optimizations.backends import torchscript
from torchdynamo.optimizations.backends import tvm_compile
from torchdynamo.testing import same, format_speedup


def synchronize():
    pass


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", "-d", action="append", help="cpu or cuda")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    parser.add_argument("--threads", "-t", type=int, help="number of threads to use")
    parser.add_argument("--name")
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
    )
    args = parser.parse_args()

    # defaults
    args.devices = args.devices or ["cpu"]

    if args.devices != ["cpu"] and torch.cuda.is_available():
        global synchronize
        synchronize = torch.cuda.synchronize

    # if args.cpu_fusion:
    torch._C._jit_override_can_fuse_on_cpu(True)

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.name:
        name = re.sub(r"^[./]*subgraphs[./]*", "", args.name)
        return run(args, name)

    headers = None
    rows = []

    for i, name in zip(
        range(args.limit) if args.limit else itertools.count(),
        sorted(os.listdir(os.path.join(config.base_dir, "subgraphs"))),
    ):
        if name.startswith("g"):
            print()
            print("BEGIN", name, i)
            res = run_subproc(args, name)
            if res is not None:
                headers = list(res[0].keys())
                rows.append(list(res[0].values()))

    print()
    print(tabulate(rows, headers=headers))


def run_subproc(args, name):
    q = multiprocessing.Queue(1)
    p = multiprocessing.Process(target=run_pipe, args=(args, name, q))
    p.start()
    try:
        while True:
            is_alive = p.is_alive()
            try:
                return q.get(True, 1)
            except queue.Empty:
                if not is_alive:
                    return None
    except KeyboardInterrupt:
        p.kill()
        raise
    except Exception:
        logging.exception(name)
    finally:
        p.join(timeout=1)


def run_pipe(args, name: str, res: multiprocessing.Queue):
    try:
        res.put(run(args, name))
    except Exception:
        logging.exception(name)
        res.put(None)
    finally:
        res.close()


def run(args, name):
    pymod = importlib.import_module(f"subgraphs.{name}")
    pymod.module._operator_iadd = operator.iadd
    pymod.module._operator_imul = operator.imul
    pymod.module._operator_itruediv = operator.itruediv
    pymod.module.math_sqrt = math.sqrt
    model0 = pymod.FxModule()
    model_dir = os.path.join(config.base_dir, "subgraphs", name)
    example_inputs = torch.load(os.path.join(model_dir, "example_inputs.pt"))
    correct, sec = timed(model0, example_inputs, 1)

    model1 = torchscript(model0, example_inputs)

    models = [
        ("eager", model0),
        ("torchscript", model1),
        ("freezing", optimize_for_inference(model1, example_inputs)),
        ("static_runtime", static_runtime(model1, example_inputs)),
        ("onnxrt", onnxrt(model1, example_inputs)),
        # ("tvm", tvm_compile(model1, example_inputs)),
        (
            "ansor128",
            tvm_compile(
                model1, example_inputs, os.path.join(model_dir, "ansor128"), trials=128
            ),
        ),
        # ("ansor1024", tvm_compile(
        #     model1, example_inputs, os.path.join(model_dir, "ansor1024"), trials=1024
        # )),
        # ("ansor10k", tvm_compile(
        #     model1, example_inputs, os.path.join(model_dir, "ansor10k), trials=10000
        # )),
    ]

    is_corrects = [x is not None for _, x in models]
    timings = np.zeros((args.repeat, len(models)), np.float64)
    timings.fill(1.0e10)
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        for i, (n, m) in enumerate(models):
            if is_corrects[i]:
                result, timings[rep, i] = timed(m, example_inputs)
                try:
                    assert same(result, correct)
                except AssertionError:
                    is_corrects[i] = False

    pvalues = [
        ttest_ind(timings[:, 0], timings[:, i]).pvalue for i in range(1, len(models))
    ]
    median = np.median(timings, axis=0)
    speedups = [median[0] / median[i] for i in range(1, len(models))]
    results = [
        format_speedup(s, p, c) for s, p, c in zip(speedups, pvalues, is_corrects[1:])
    ]
    row = [name, f"{sec:.4f}"] + results
    names = [k for k, v in models]
    headers = ["name", "sec"] + names[1:]
    print(tabulate([row], headers=headers))
    perf = {k: float(v) for k, v, c in zip(names, median, is_corrects) if c}
    with open(os.path.join(model_dir, "perf.json"), "w") as fd:
        json.dump(perf, fd)
    print({k: v for k, v, c in zip(names, median, is_corrects) if c})
    return OrderedDict(zip(headers, row)), perf


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    with torch.no_grad():
        main()
