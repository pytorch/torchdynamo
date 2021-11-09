import collections
import functools
import os
import weakref

import torch
from torch import fx

counters = collections.defaultdict(collections.Counter)


def count_calls(g: fx.Graph):
    c = 0
    for n in g.nodes:
        if "call" in n.op:
            c += 1
    return c


def identity(x):
    return x


class ExactWeakKeyDictionary:
    """Similar to weakref.WeakKeyDictionary, but use `is`/`id` rather than `==` to compare equality"""

    def __init__(self):
        self.values = dict()
        self.refs = dict()

    def __getitem__(self, key):
        return self.values[id(key)]

    def __contains__(self, key):
        return id(key) in self.values

    def __setitem__(self, key, value):
        idx = id(key)
        if idx not in self.refs:
            self.refs[idx] = weakref.ref(key, lambda ref: self._remove_id(idx))
        self.values[idx] = value

    def _remove_id(self, idx):
        if idx in self.refs:
            del self.values[idx]
            del self.refs[idx]

    def clear(self):
        self.values.clear()
        self.refs.clear()


def istype(obj, allowed_types):
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types
    return type(obj) is allowed_types


def istensor(obj):
    """Check of obj is a tensor"""
    return istype(obj, (torch.Tensor, torch.nn.Parameter))


@functools.lru_cache(None)
def print_once(msg):
    print(msg)


def make_cell(val):
    """Some black magic to create a cell object that usually only exists in a closure"""
    x = val

    def f():
        return x

    assert len(f.__closure__) == 1
    return f.__closure__[0]


class Unsupported(RuntimeError):
    pass


def unimplemented(msg: str):
    counters["unimplemented"][msg] += 1
    assert msg != os.environ.get("BREAK", False)
    raise Unsupported(msg)


def warning(msg: str):
    assert msg != os.environ.get("BREAK", False)
    counters["warnings"][msg] += 1
