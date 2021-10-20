import enum
import inspect
import operator
import os
import re
import types
import typing

import torch

SKIP_DIRS = [
    # torch.*
    os.path.dirname(torch.__file__) + "/",
    # torchdynamo.*
    os.path.dirname(__file__) + "/",
] + [
    # skip some standard libs
    re.sub(r"__init__.py$", "", m.__file__)
    for m in (enum, inspect, re, operator, types, typing)
]


def check(filename):
    """Should skip this file?"""
    return any(filename.startswith(d) for d in SKIP_DIRS)
