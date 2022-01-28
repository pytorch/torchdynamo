#!/usr/bin/env python
import argparse
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import torch
from scipy.stats import ttest_ind
from tabulate import tabulate

import torchdynamo.optimizations.backends
from torchdynamo import config
from torchdynamo.optimizations.backends import BACKENDS
from torchdynamo.optimizations.subgraph import SubGraph
from torchdynamo.testing import format_speedup
from torchdynamo.testing import same

synchronize = torch.cuda.synchronize


def nothing():
    pass


def timed(model, example_inputs, times=1):
    if not torch.cuda.is_available():
        global synchronize
        synchronize = nothing

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
    parser.add_argument(
        "--repeat", "-n", type=int, default=20, help="number of timing runs"
    )
    parser.add_argument("--threads", "-t", type=int, help="number of threads to use")
    parser.add_argument("--name")
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--silent", "-q", action="store_true")
    parser.add_argument("--max-age", type=int, default=48)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--skip", action="append", help="dont use a backend")
    parser.add_argument(
        "--nvfuser",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
    )
    args = parser.parse_args()

    if args.silent:
        torchdynamo.optimizations.backends.log.setLevel(logging.FATAL)

    if args.threads:
        torch.set_num_threads(args.threads)

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

    if args.name:
        name = re.sub(r"^[./]*subgraphs[./]*", "", args.name)
        return run(args, name)
    elif args.stats:
        rows = []
        for name in list_subgraphs(args):
            with open(
                os.path.join(config.base_dir, "subgraphs", name, "stats.json")
            ) as f:
                keys, values = json.load(f)
            rows.append(OrderedDict(zip(keys, values)))
        headers = OrderedDict()
        for row in rows:
            headers.update(row)
        headers = list(headers.keys())
        rows = [[row.get(k, "") for k in headers] for row in rows]
        print(tabulate(rows, headers=headers))
    else:
        for i, name in enumerate(list_subgraphs(args)):
            path = os.path.join(config.base_dir, "subgraphs", name)
            has_perf = os.path.exists(os.path.join(path, "perf.json"))
            try:
                age = time.time() - float(open(os.path.join(path, "timestamp")).read())
            except OSError:
                age = float("inf")
            if has_perf and (args.new or age > args.max_age * 3600):
                continue

            print()
            print("BEGIN", name, i)
            cmd = [sys.executable] + list(sys.argv) + ["--name", name]

            try:
                subprocess.check_call(cmd)
            except Exception:
                try:
                    skip = open(os.path.join(path, "running")).read().strip()
                    print("ERROR skipping", skip)
                    cmd += ["--skip", skip]
                    subprocess.check_call(cmd)
                except Exception:
                    logging.exception("failure from %s", name)


def list_subgraphs(args):
    limit = args.limit or float("inf")
    for name in sorted(os.listdir(os.path.join(config.base_dir, "subgraphs"))):
        if name.startswith("g"):
            yield name
            limit -= 1
            if limit <= 0:
                return


def run(args, graph_name):
    subgraph = SubGraph.load(graph_name)

    # these next lines control which backends to try
    # for a list see torchdynamo/optimizations/backends.py
    if subgraph.is_cuda:
        backend_names = [
            "eager",
            "nnc",
            "nvfuser",
            "ofi",
            "cudagraphs",
            "onnxrt_cuda",
            "tensorrt",
            "onnx2tf",
        ]
    else:
        backend_names = ["eager", "ts", "ofi", "onnxrt_cpu", "onnx2tf"]

    skip = set(args.skip or [])
    backend_names = [x for x in backend_names if x not in skip]

    models = []
    for name in backend_names:
        with open(subgraph.filename("running"), "w") as fd:
            fd.write(name)
        models.append((name, BACKENDS[name](subgraph)))
    os.unlink(subgraph.filename("running"))

    example_inputs = subgraph.example_inputs
    example_outputs = subgraph.example_outputs

    is_corrects = [x is not None for _, x in models]
    timings = np.zeros((args.repeat, len(models)), np.float64)
    timings.fill(1.0e10)
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        for i, (n, m) in enumerate(models):
            if is_corrects[i]:
                try:
                    result, timings[rep, i] = timed(m, example_inputs)
                    assert same(result, example_outputs)
                except AssertionError:
                    logging.exception(f"incorrect while running {n}")
                    is_corrects[i] = False
                except Exception:
                    logging.exception(f"error while running {n}")
                    is_corrects[i] = False

    assert is_corrects[0]

    pvalues = [
        ttest_ind(timings[:, 0], timings[:, i]).pvalue for i in range(1, len(models))
    ]
    median = np.median(timings, axis=0)
    speedups = [median[0] / median[i] for i in range(1, len(models))]
    results = [
        format_speedup(s, p, c) for s, p, c in zip(speedups, pvalues, is_corrects[1:])
    ]
    sec = float(np.mean(timings[:, 0]))
    row = [graph_name, f"{sec:.4f}"] + results
    names = [k for k, v in models]
    headers = ["name", "sec"] + names[1:]
    print(tabulate([row], headers=headers))
    perf = {k: float(v) for k, v, c in zip(names, median, is_corrects) if c}
    with open(subgraph.filename("perf.json"), "w") as fd:
        json.dump(perf, fd)
    with open(subgraph.filename("stats.json"), "w") as fd:
        json.dump([headers, row], fd)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    with torch.no_grad():
        main()
