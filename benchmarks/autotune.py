#!/usr/bin/env python
import argparse
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
from tabulate import tabulate

import torchdynamo.optimizations.backends
from torchdynamo import config
from torchdynamo.optimizations.backends import BACKENDS
from torchdynamo.optimizations.subgraph import SubGraph
from torchdynamo.testing import same
from torchdynamo.utils import clone_inputs
from torchdynamo.utils import timed


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
            "cudagraphs",
            "ts",
            "cudagraphs_ts",
            "ofi",
            "cudagraphs_ts_ofi",
            "tensorrt",
            "nnc" if args.nvfuser else "nvfuser",
        ]
    else:
        backend_names = ["eager", "ts", "ofi", "onnxrt_cpu", "onnx2tf"]

    skip = set(args.skip or [])
    backend_names = [x for x in backend_names if x not in skip]

    example_inputs = torch.load(
        os.path.join(config.base_dir, "subgraphs", graph_name, "example_inputs.pt")
    )
    example_outputs = torch.load(
        os.path.join(config.base_dir, "subgraphs", graph_name, "example_outputs.pt")
    )

    models = []
    for name in backend_names:
        with open(subgraph.filename("running"), "w") as fd:
            fd.write(name)
        try:
            compiled_model = BACKENDS[name](subgraph)
            if compiled_model is None:
                continue
            subgraph.restore()
            result = compiled_model(*clone_inputs(example_inputs))
            assert same(result, example_outputs)
            models.append((name, compiled_model))
        except AssertionError:
            logging.exception(f"incorrect while running {name}")
        except Exception:
            logging.exception(f"error while running {name}")
    os.unlink(subgraph.filename("running"))
    del example_outputs

    timings = np.zeros((args.repeat, len(models)), np.float64)
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        for i, (n, m) in enumerate(models):
            result, timings[rep, i] = timed(m, example_inputs)

    median = np.median(timings, axis=0)
    names = [k for k, v in models]
    sec = float(np.mean(timings[:, 0]))
    headers = ["name", "sec"]
    row = [graph_name, f"{sec:.4f}"]
    if models[0][0] == "eager" and len(models) > 1:
        speedups = [median[0] / median[i] for i in range(1, len(models))]
        row.extend(f"{x:.2}x" for x in speedups)
        headers.extend(names[1:])
    else:
        row.extend(f"{x:.2} sec" for x in median)
        headers.extend(names)

    print(tabulate([row], headers=headers))

    perf = {k: float(v) for k, v in zip(names, median)}
    with open(subgraph.filename("perf.json"), "w") as fd:
        json.dump(perf, fd)
    with open(subgraph.filename("stats.json"), "w") as fd:
        json.dump([headers, row], fd)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    with torch.no_grad():
        main()
