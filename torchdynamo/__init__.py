"""
NOTICE: torchdynamo has moved to torch._dynamo

This file is a shim to redirect to the new location

For more info see:
https://github.com/pytorch/torchdynamo/issues/681
"""
import importlib
import sys

import torch._dynamo


def _populate():
    for name in (
        "allowed_functions",
        "bytecode_analysis",
        "bytecode_transformation",
        "codegen",
        "config",
        "convert_frame",
        "debug_utils",
        "eval_frame",
        "exc",
        "guards",
        "logging",
        "mutation_guard",
        "optimizations",
        "output_graph",
        "profiler",
        "replay_record",
        "resume_execution",
        "side_effects",
        "skipfiles",
        "source",
        "symbolic_convert",
        "test_case",
        "testing",
        "utils",
        "variables",
    ):
        try:
            globals()[name] = sys.modules[
                f"torchdynamo.{name}"
            ] = importlib.import_module(f"torch._dynamo.{name}")
        except ImportError:
            pass

    for name, val in torch._dynamo.__dict__.items():
        if not name.startswith("_"):
            globals()[name] = val


_populate()
