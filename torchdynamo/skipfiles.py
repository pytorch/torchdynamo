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
import logging
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
    "<__array_function__ internals>",
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
        logging,
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
FILENAME_ALLOWLIST = {
    torch.nn.Sequential.__init__.__code__.co_filename,
}


def add(module: types.ModuleType):
    assert isinstance(module, types.ModuleType)
    global SKIP_DIRS_RE
    name = module.__file__
    if name is None:
        return
    if name.endswith("__init__.py"):
        name = os.path.dirname(name) + "/"
    SKIP_DIRS.append(name)
    SKIP_DIRS_RE = re.compile(f"^({'|'.join(map(re.escape, SKIP_DIRS))})")


def check(filename, allow_torch=False):
    """Should skip this file?"""
    if filename is None:
        return True
    if filename in FILENAME_ALLOWLIST:
        return False
    if allow_torch and is_torch(filename):
        return False
    return bool(SKIP_DIRS_RE.match(filename))


# skip common third party libs
for _name in (
    "functorch",
    "intel_extension_for_pytorch",
    "networkx",
    "numpy",
    "omegaconf",
    "onnx",
    "onnxruntime",
    "onnx_tf",
    "pandas",
    "sklearn",
    "tabulate",
    "tensorflow",
    "tensorrt",
    "torch2trt",
    "tqdm",
    "tree",
    "tvm",
):
    try:
        add(importlib.import_module(_name))
    except (ImportError, TypeError):
        pass


def is_torch_inline_allowed(filename):
    return filename.startswith(
        os.path.dirname(torch.nn.__file__)
    ) or filename.startswith(os.path.dirname(torch.distributions.__file__))


def is_torch(filename):
    return filename.startswith(os.path.dirname(torch.__file__))
