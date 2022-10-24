"""
NOTICE: torchinductor has moved to torch._inductor

This file is a shim to redirect to the new location

For more info see:
https://github.com/pytorch/torchdynamo/issues/681
"""
import importlib
import sys

import torch._inductor


def _populate():
    for name in (
        "codecache",
        "codegen",
        "compile_fx",
        "config",
        "cuda_properties",
        "debug",
        "decomposition",
        "dependencies",
        "exc",
        "graph",
        "ir",
        "lowering",
        "metrics",
        "overrides",
        "scheduler",
        "sizevars",
        "triton_ops",
        "utils",
        "virtualized",
    ):
        try:
            globals()[name] = sys.modules[
                f"torchinductor.{name}"
            ] = importlib.import_module(f"torch._inductor.{name}")
        except ImportError:
            pass

    for name, val in torch._inductor.__dict__.items():
        if not name.startswith("_"):
            globals()[name] = val


_populate()
