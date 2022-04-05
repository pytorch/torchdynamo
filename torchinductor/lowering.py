import functools
import itertools
from typing import List

import sympy
import torch
import torch.fx

from .ir import ExpandView
from .ir import Reduction
from .ir import TensorBox
from .ir import UnrealizedBuffer
from .virtualized_ops import ops

lowerings = {}
aten = torch.ops.aten


def _register_lowering(aten_fn, decomp_fn, broadcast, type_promote):
    """
    Add a lowering to lowerings dict

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promote: True to apply type promotion to tensor inputs
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        args = list(args)
        # Only look at args that are Tensors
        indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
        # kwargs tensors not supported yet
        assert not any(isinstance(x, TensorBox) for x in kwargs.values())

        if type_promote and indices:
            dtype = functools.reduce(
                torch.promote_types, [args[i].get_dtype() for i in indices]
            )
            for i in indices:
                args[i] = to_dtype(args[i], dtype)

        if broadcast and indices:
            for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
                args[i] = x

        return decomp_fn(*args, **kwargs)

    lowerings[aten_fn] = wrapped
    return wrapped


def register_lowering(aten_fn, broadcast=False, type_promote=True):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        _register_lowering, aten_fn, broadcast=broadcast, type_promote=type_promote
    )


def broadcast_symbolic_shapes(a, b):
    """
    Broadcasting logic based on symbolic shapes.

    We give the shapes 0 and 1 concrete values, while all other shapes
    are symbolic sympy formulas.
    """
    output = []
    for a, b in itertools.zip_longest(
        reversed(a), reversed(b), fillvalue=sympy.Integer(1)
    ):
        if b == 1:
            output.append(a)
        elif a == 1:
            output.append(b)
        else:
            guard_shape_equal(a, b)
            if len(str(b)) < len(str(a)):
                output.append(b)  # prefer shorter formula
            else:
                output.append(a)
    return tuple(reversed(output))


def guard_shape_equal(a, b):
    if a != b:
        assert False
        pass  # TODO(jansel): implement guarding


def make_pointwise(fn, override_dtype=None, override_device=None):
    def inner(*inputs: List[TensorBox]):
        loaders = [x.make_loader() for x in inputs]
        return UnrealizedBuffer.create(
            inputs[0].get_size(),
            lambda index: fn(*[load(index) for load in loaders]),
            device=override_device or inputs[0].get_device(),
            dtype=override_dtype or inputs[0].get_dtype(),
        )

    return inner


def to_dtype(x: TensorBox, dtype: torch.dtype):
    if x.get_dtype() == dtype:
        return x

    def _to_dtype(x):
        return ops.to_dtype(x, dtype)

    return make_pointwise(_to_dtype, override_dtype=dtype)(x)


def register_pointwise(aten_fn, name=None, broadcast=True, type_promote=True):
    """A pointwise function that maps ops.{name} to inputs"""
    name = name or aten_fn.__name__

    @register_lowering(aten_fn, broadcast=broadcast, type_promote=type_promote)
    @make_pointwise
    def fn(*args, **kwargs):
        return getattr(ops, name)(*args, **kwargs)

    return fn


@register_lowering(aten.broadcast_tensors, broadcast=False, type_promote=False)
def broadcast_tensors(*inputs):
    target = functools.reduce(
        broadcast_symbolic_shapes, [x.get_size() for x in inputs], ()
    )
    outputs = []
    for x in inputs:
        sizes = x.get_size()
        if len(sizes) != len(target) or any(
            ((a == 1 and b != 1) or (a != 1 and b == 1)) for a, b in zip(sizes, target)
        ):
            x = expand(x, target)
        outputs.append(x)
    return outputs


@register_lowering(aten.expand, type_promote=False, broadcast=False)
def expand(x, sizes):
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    return TensorBox(ExpandView(x.data, tuple(sizes)))


def make_reduction(reduction_type: str):
    def inner(x, axis):
        axis = list(axis)
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] += len(axis)
            assert 0 <= axis[i] < len(axis)
        assert len(set(axis)) == len(axis), "reduction axis not unique"
        axis = set(axis)

        kept_sizes = []
        kept_idx = []
        reduced_sizes = []
        reduced_idx = []
        size = x.get_size()
        for i in range(len(size)):
            if i in axis:
                reduced_idx.append(i)
                reduced_sizes.append(size[i])
            else:
                kept_idx.append(i)
                kept_sizes.append(size[i])

        def loader(index, reduction_index):
            assert len(index) == len(kept_idx)
            assert len(reduction_index) == len(reduced_idx)
            new_index = [None] * len(index)
            for idx, var in itertools.chain(
                zip(kept_idx, index), zip(reduced_idx, reduction_index)
            ):
                new_index[idx] = var
            return inner_loader(new_index)

        inner_loader = x.make_loader()
        return Reduction.create(
            kept_sizes,
            loader,
            device=x.get_device(),
            dtype=x.get_dtype(),
            reduction_size=reduced_sizes,
            reduction_type=reduction_type,
        )

    return inner


register_lowering(aten.sum)(make_reduction("sum"))
register_pointwise(aten.add)
register_pointwise(aten.div)
register_pointwise(aten.abs)
register_pointwise(aten.sub)
register_pointwise(aten.mul)
register_pointwise(aten.maximum)
register_pointwise(aten.minimum)
