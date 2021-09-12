import types
from functools import lru_cache

import torch


@lru_cache(None)
def _allowed_function_ids():
    torch_object_ids = set()

    def _find_torch_objects(module):
        torch_object_ids.add(id(module))
        for name, obj in list(module.__dict__.items()):
            if id(obj) not in torch_object_ids:
                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith("torch."):
                        torch_object_ids.add(id(obj))
                        _find_torch_objects(obj)
                else:
                    torch_object_ids.add(id(obj))

    _find_torch_objects(torch)

    return torch_object_ids


def is_allowed(obj):
    """Is this safe to trace like torch.add ?"""
    return id(obj) in _allowed_function_ids()
