import builtins
import collections
import math
import types
import warnings
from functools import lru_cache

import torch


@lru_cache(None)
def _disallowed_function_ids():
    remove = [
        True,
        False,
        None,
        collections.OrderedDict,
        torch.autocast_decrement_nesting,
        torch.autocast_increment_nesting,
        torch.clear_autocast_cache,
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
    torch_object_ids = dict()

    def _find_torch_objects(module):
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
    return {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and callable(v)
    }


def is_builtin(obj):
    return id(obj) in _builtin_function_ids()
