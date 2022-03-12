import builtins
import collections
import copy
import functools
import itertools
import math
import operator
import types
import warnings
from functools import lru_cache

import numpy
import torch


@lru_cache(None)
def _disallowed_function_ids():
    remove = [
        True,
        False,
        None,
        collections.OrderedDict,
        copy.copy,
        copy.deepcopy,
        torch.autocast_decrement_nesting,
        torch.autocast_increment_nesting,
        torch.clear_autocast_cache,
        torch.distributions.constraints.is_dependent,
        torch.distributions.normal.Normal,
        torch.inference_mode,
        torch.set_anomaly_enabled,
        torch.set_autocast_cache_enabled,
        torch.set_autocast_cpu_dtype,
        torch.set_autocast_cpu_enabled,
        torch.set_autocast_enabled,
        torch.set_autocast_gpu_dtype,
        warnings.warn,
    ]
    return {id(x) for x in remove}


@lru_cache(None)
def _allowed_function_ids():
    """
    Walk torch.* and get the ids of all the stuff in it
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    torch.distributions.Distribution.set_default_validate_args(False)
    torch_object_ids = dict()

    def _find_torch_objects(module):
        if module.__name__.startswith("torch.distributions"):
            return
        torch_object_ids[id(module)] = module.__name__
        for name, obj in list(module.__dict__.items()):
            if id(obj) not in torch_object_ids:
                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith("torch."):
                        torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                        _find_torch_objects(obj)
                else:
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"

    _find_torch_objects(torch)
    _find_torch_objects(math)

    for idx in _disallowed_function_ids():
        if idx in torch_object_ids:
            del torch_object_ids[idx]

    return torch_object_ids


def is_allowed(obj):
    """Is this safe to trace like torch.add ?"""
    return id(obj) in _allowed_function_ids()


def is_disallowed(obj):
    """Is this safe to trace like torch.add ?"""
    return id(obj) in _disallowed_function_ids()


@lru_cache(None)
def _builtin_function_ids():
    rv = {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and callable(v)
    }
    rv.update(
        {
            id(v): f"operator.{k}"
            for k, v in operator.__dict__.items()
            if not k.startswith("_") and callable(v)
        }
    )
    rv.update(
        {id(v): f"functools.{v.__name__}" for v in (itertools.chain, itertools.islice)}
    )
    rv[id(functools.reduce)] = "functools.reduce"
    return rv


def is_builtin(obj):
    return id(obj) in _builtin_function_ids()


@lru_cache(None)
def _numpy_function_ids():
    rv = dict()
    for mod in (numpy, numpy.random):
        rv.update(
            {
                id(v): f"{mod.__name__}.{k}"
                for k, v in mod.__dict__.items()
                if callable(v)
                and (getattr(v, "__module__", None) or mod.__name__) == mod.__name__
            }
        )
    return rv


def is_numpy(obj):
    return isinstance(obj, numpy.ndarray) or id(obj) in _numpy_function_ids()
