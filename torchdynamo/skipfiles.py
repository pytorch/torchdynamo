import abc
import collections
import contextlib
import copy
import copyreg
import dataclasses
import enum
import functools
import importlib
import inspect
import linecache
import multiprocessing
import operator
import os
import random
import re
import selectors
import threading
import traceback
import types
import typing
import unittest
import weakref

import _collections_abc
import _weakrefset
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
        collections,
        contextlib,
        copy,
        copyreg,
        dataclasses,
        enum,
        functools,
        importlib,
        inspect,
        linecache,
        multiprocessing,
        operator,
        os,
        random,
        re,
        selectors,
        threading,
        traceback,
        types,
        typing,
        unittest,
        weakref,
        _collections_abc,
        _weakrefset,
    )
]
SKIP_DIRS_RE = None  # set in add() below


def add(module: types.ModuleType):
    assert isinstance(module, types.ModuleType)
    global SKIP_DIRS_RE
    SKIP_DIRS.append(os.path.dirname(module.__file__) + "/")
    SKIP_DIRS_RE = re.compile(f"^({'|'.join(map(re.escape, SKIP_DIRS))})")


def check(filename):
    """Should skip this file?"""
    if filename is None:
        return True
    return bool(SKIP_DIRS_RE.match(filename))


# skip common third party libs
for _name in (
    "intel_extension_for_pytorch",
    "functorch",
    "numpy",
    "omegaconf",
    "onnx",
    "onnxruntime",
    "onnx_tf",
    "pandas",
    "sklearn",
    "tensorflow",
    "tensorrt",
    "torch2trt",
    "tree",
    "tqdm",
    "tvm",
):
    try:
        add(importlib.import_module(_name))
    except ImportError:
        pass
