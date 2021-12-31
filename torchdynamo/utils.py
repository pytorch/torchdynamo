import collections
import dataclasses
import functools
import inspect
import logging
import os
import weakref
from typing import Any
from typing import Dict

import torch
from torch import fx

log = logging.getLogger(__name__)
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
    counters["warnings"][msg] += 1
    assert msg != os.environ.get("BREAK", False)


def proxy_args_kwargs(args, kwargs):
    try:
        proxy_args = tuple(arg.as_proxy() for arg in args)
        proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return proxy_args, proxy_kwargs
    except NotImplementedError:
        from .variable_tracker import typestr

        raise unimplemented(
            f"call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}"
        )


@dataclasses.dataclass
class CleanupHook:
    """Remove a global variable when hook is called"""

    scope: Dict[str, Any]
    name: str

    def __call__(self, *args):
        CleanupManager.count -= 1
        del self.scope[self.name]

    @staticmethod
    def create(scope, name, val):
        assert name not in scope
        CleanupManager.count += 1
        scope[name] = val
        return CleanupHook(scope, name)


class CleanupManager(ExactWeakKeyDictionary):
    count = 0

    def _remove_id(self, idx):
        for hook in self.values[idx]:
            hook()
        super()._remove_id(idx)


CleanupManager.instance = CleanupManager()


def clone_inputs(example_inputs):
    res = list(example_inputs)
    for i in range(len(res)):
        if isinstance(res[i], torch.Tensor):
            res[i] = res[i].clone().detach()
    return res


def is_jit_model(model0):
    return isinstance(
        model0,
        (
            torch.jit._trace.TopLevelTracedModule,
            torch.jit._script.RecursiveScriptModule,
            torch.jit.ScriptFunction,
            torch.jit.ScriptModule,
        ),
    )


def torchscript(model0, example_inputs, verbose=True):
    if is_jit_model(model0):
        return model0
    try:
        model1 = torch.jit.trace(model0, example_inputs)
    except Exception:
        if verbose:
            log.exception("jit trace error")
        try:
            model1 = torch.jit.script(model0)
        except Exception:
            model1 = None
    return model1


def getfile(obj):
    try:
        return inspect.getfile(obj)
    except TypeError:
        return None
