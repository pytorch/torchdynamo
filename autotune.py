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

import torchdynamo.optimizations.backends
from torchdynamo import config
from torchdynamo.optimizations.backends import cudagraphs
from torchdynamo.optimizations.backends import fx2trt
from torchdynamo.optimizations.backends import ipex
from torchdynamo.optimizations.backends import is_jit_model
from torchdynamo.optimizations.backends import onnx2trt
from torchdynamo.optimizations.backends import onnxrt
from torchdynamo.optimizations.backends import optimize_for_inference
from torchdynamo.optimizations.backends import static_runtime
from torchdynamo.optimizations.backends import taso
from torchdynamo.optimizations.backends import torch2trt
from torchdynamo.optimizations.backends import torchscript
from torchdynamo.optimizations.backends import tvm_compile
from torchdynamo.testing import format_speedup
from torchdynamo.testing import same

ANSOR = False
TASO = False
STATIC_RUNTIME = False
IPEX = False
TVM = False


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
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--silent", "-q", action="store_true")
    parser.add_argument("--ansor-sec", type=float)
    parser.add_argument("--max-age", type=int, default=24)
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
    )
    args = parser.parse_args()

    # defaults
    args.devices = args.devices or ["cpu"]

    if args.silent:
        torchdynamo.optimizations.backends.log.setLevel(logging.FATAL)

    if args.devices != ["cpu"] and torch.cuda.is_available():
        global synchronize
        synchronize = torch.cuda.synchronize

    torch._C._jit_override_can_fuse_on_cpu(True)

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.name:
        name = re.sub(r"^[./]*subgraphs[./]*", "", args.name)
        return run(args, name, False)

    rows = []

    for i, name in zip(
        range(args.limit) if args.limit else itertools.count(),
        sorted(os.listdir(os.path.join(config.base_dir, "subgraphs"))),
    ):
        if name.startswith("g"):
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
            res = run_subproc(args, name, False)
            if res is not None:
                rows.append(res)
            else:
                print("No result, trying safe_mode")
                # often static runtime segfaults, so run without it
                res = run_subproc(args, name, True)
                if res is not None:
                    rows.append(res)
                else:
                    print("Safe mode failed")

    if rows:
        headers = OrderedDict()
        for row in rows:
            headers.update(row)
        headers = list(headers.keys())
        rows = [[row.get(k, "") for k in headers] for row in rows]
        print(tabulate(rows, headers=headers))


def run_subproc(args, name, safe_mode):
    q = multiprocessing.Queue(1)
    p = multiprocessing.Process(target=run_pipe, args=(args, name, safe_mode, q))
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


def run_pipe(args, name: str, safe_mode, res: multiprocessing.Queue):
    try:
        res.put(run(args, name, safe_mode))
    except Exception:
        logging.exception(name)
        res.put(None)
    finally:
        res.close()


def autotune_ansor(model1, example_inputs, model_dir, args):
    run_ansor = False
    if os.path.exists(os.path.join(model_dir, "perf.json")):
        perf = json.loads(open(os.path.join(model_dir, "perf.json")).read())
        if os.path.exists(os.path.join(model_dir, "ansor20k")) or (
            args.ansor_sec and perf.get("eager", 0) > args.ansor_sec
        ):
            run_ansor = True

    if run_ansor:
        return (
            "ansor20k",
            tvm_compile(
                model1,
                example_inputs,
                os.path.join(model_dir, "ansor20k"),
                trials=20000,
            ),
        )
    else:
        return ("ansor20k", None)


def load_module_fx(name):
    pymod = importlib.import_module(f"subgraphs.{name}")
    # TODO(jansel): upstream these fixes to to_folder()
    pymod.module._operator_iadd = operator.iadd
    pymod.module._operator_imul = operator.imul
    pymod.module._operator_itruediv = operator.itruediv
    pymod.module._operator_setitem = operator.setitem
    pymod.module.math_sqrt = math.sqrt
    pymod.module.device = torch.device
    pymod.module.inf = float("inf")
    return pymod.FxModule()


def load_module_jit(name):
    filename = os.path.join(config.base_dir, "subgraphs", name, "model.ts")
    if not os.path.exists(filename):
        return None
    model = torch.jit.load(filename)
    assert is_jit_model(model)
    return model


def get_options_cpu(model_fx, model_jit, example_inputs, model_dir, safe_mode):
    models = [
        ("eager", model_fx),
        ("torchscript", model_jit),
        ("onnxrt", onnxrt(model_jit, example_inputs, os.path.join(model_dir, "onnx"))),
        ("freezing", optimize_for_inference(model_jit, example_inputs)),
    ]
    if safe_mode:
        return models
    if TVM:
        models.append(("tvm", tvm_compile(model_jit, example_inputs)))
    if IPEX:
        models.append(("ipex", ipex(model_jit, example_inputs)))
    if STATIC_RUNTIME:
        # Static runtime is crashy, don't run it in safe mode
        models.append(("static_runtime", static_runtime(model_jit, example_inputs)))
    if ANSOR:
        models.append(autotune_ansor(model_jit, example_inputs, model_dir))
    if TASO:
        models.append(
            (
                "taso",
                taso(
                    example_inputs,
                    os.path.join(model_dir, "onnx"),
                    os.path.join(model_dir, "taso"),
                ),
            )
        )
    return models


def get_options_gpu(model_fx, model_jit, example_inputs, model_dir, safe_model):
    models = [
        ("eager", model_fx),
        ("torchscript", model_jit),
        ("freezing", optimize_for_inference(model_jit, example_inputs)),
        ("fx2trt", fx2trt(model_fx, example_inputs)),
        ("torch2trt", torch2trt(model_fx, example_inputs)),
        ("onnx2trt", onnx2trt(model_jit, example_inputs)),
        ("cudagraphs", cudagraphs(model_fx, example_inputs)),
    ]
    return models


def run(args, name, safe_mode):
    model_dir = os.path.join(config.base_dir, "subgraphs", name)
    example_inputs = torch.load(os.path.join(model_dir, "example_inputs.pt"))
    example_outputs = torch.load(os.path.join(model_dir, "example_outputs.pt"))
    metadata = json.loads(open(os.path.join(model_dir, "metadata.json")).read())
    model_fx = load_module_fx(name)
    model_jit = load_module_jit(name)
    if metadata["is_cuda"]:
        model_fx = model_fx.cuda()
        model_jit = model_jit.cuda()
        get_options = get_options_gpu
    else:
        get_options = get_options_cpu

    if model_jit is None:
        model_jit = torchscript(model_fx, example_inputs)
    if not same(example_outputs, model_fx(*example_inputs)):
        logging.warning("FX graph is incorrect")
        assert model_jit and same(example_outputs, model_jit(*example_inputs))
        model_fx = model_jit

    models = get_options(model_fx, model_jit, example_inputs, model_dir, safe_mode)

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
    row = [name, f"{sec:.4f}"] + results
    names = [k for k, v in models]
    headers = ["name", "sec"] + names[1:]
    print(tabulate([row], headers=headers))
    perf = {k: float(v) for k, v, c in zip(names, median, is_corrects) if c}
    with open(os.path.join(model_dir, "perf.json"), "w") as fd:
        json.dump(perf, fd)
    return OrderedDict(zip(headers, row))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    with torch.no_grad():
        main()
