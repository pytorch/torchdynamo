import functools
import operator
import time
from typing import Dict

import numpy as np
import sympy
import torch
from torch.cuda import synchronize
from torch.fx.immutable_collections import immutable_dict
from torch.fx.immutable_collections import immutable_list

VarRanges = Dict[sympy.Expr, sympy.Expr]


@functools.lru_cache(None)
def has_triton():
    try:
        import triton

        return triton is not None
    except (ImportError, ModuleNotFoundError):
        return False


@functools.lru_cache(None)
def has_torchvision_roi_align():
    try:
        from torchvision.ops import roi_align  # noqa

        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except (ImportError, ModuleNotFoundError):
        return False


@functools.lru_cache(None)
def has_triton_libdevice():
    try:
        from triton.language import libdevice

        return libdevice is not None
    except (ImportError, ModuleNotFoundError):
        return False


def conditional_product(*args):
    return functools.reduce(operator.mul, [x for x in args if x])


def sympy_product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def unique(it):
    return {id(x): x for x in it}.values()


def timed(model, example_inputs, times=1):
    synchronize()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    # GC the result after timing
    assert result is not None
    return t1 - t0


def print_performance(fn, args=(), times=10, repeat=10, baseline=1.0):
    timings = [timed(fn, args, times) for _ in range(repeat)]
    took = np.median(timings)
    print(f"{took/baseline:.6f}")
    return took


immutable_dict.__hash__ = lambda self: hash(tuple(self.items()))
immutable_list.__hash__ = lambda self: hash(tuple(self))


def freeze_inputs(f):
    """
    Useful for wrapping lists in tuples for caching purposes
    """

    def freeze_value(x):
        if isinstance(x, list):
            return immutable_list(x)
        if isinstance(x, dict):
            return immutable_dict(x)
        return x

    @functools.wraps(f)
    def wrapped(*args):
        args = [freeze_value(x) for x in args]
        return f(*args)

    wrapped.cache_info = f.cache_info
    return wrapped
