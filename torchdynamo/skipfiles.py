import enum
import importlib
import inspect
import multiprocessing
import operator
import os
import re
import selectors
import threading
import types
import typing
import _weakrefset

import torch

SKIP_DIRS = [
    # torch.*
    os.path.dirname(torch.__file__) + "/",
    # torchdynamo.*
    os.path.dirname(__file__) + "/",
] + [
    # skip some standard libs
    re.sub(r"__init__.py$", "", m.__file__)
    for m in (
        os,
        enum,
        inspect,
        re,
        operator,
        types,
        typing,
        threading,
        multiprocessing,
        _weakrefset,
        selectors,
    )
]

# skip common third party libs
for _name in ("numpy", "onnx", "tvm", "onnxruntime", "tqdm"):
    try:
        SKIP_DIRS.append(os.path.dirname(importlib.import_module(_name).__file__) + "/")
    except ImportError:
        pass


def check(filename):
    """Should skip this file?"""
    return any(filename.startswith(d) for d in SKIP_DIRS)
