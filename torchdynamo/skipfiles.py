import _weakrefset
import abc
import copy
import enum
import importlib
import inspect
import multiprocessing
import operator
import os
import random
import re
import selectors
import threading
import types
import typing
import unittest

import torch

SKIP_DIRS = [
    # torch.*
    os.path.dirname(torch.__file__) + "/",
    # torchdynamo.*
    os.path.dirname(__file__) + "/",
    "<frozen importlib",
] + [
    # skip some standard libs
    re.sub(r"__init__.py$", "", m.__file__)
    for m in (
        abc,
        copy,
        enum,
        importlib,
        inspect,
        multiprocessing,
        operator,
        os,
        random,
        re,
        selectors,
        threading,
        types,
        typing,
        unittest,
        _weakrefset,
    )
]

# skip common third party libs
for _name in (
    "numpy",
    "onnx",
    "tvm",
    "onnxruntime",
    "tqdm",
    "pandas",
    "sklearn",
    "intel_extension_for_pytorch",
):
    try:
        SKIP_DIRS.append(os.path.dirname(importlib.import_module(_name).__file__) + "/")
    except ImportError:
        pass

SKIP_DIRS_RE = re.compile(f"^({'|'.join(map(re.escape, SKIP_DIRS))})")


def check(filename):
    """Should skip this file?"""
    return bool(SKIP_DIRS_RE.match(filename))
