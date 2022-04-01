import collections
import functools
import itertools
from itertools import chain
from typing import List

import sympy
import torch
import torch.fx
from sympy import Integer

from . import config
from .codegen import ScheduleCodeGen
from .ir import Constant
from .ir import ExpandView
from .ir import FixedLayout
from .ir import InputBuffer
from .ir import TensorBox
from .ir import UnrealizedBuffer
from .shapes import SizeVarAllocator
from .virtualized import prim

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
        return prim.to_dtype(x, dtype)

    return make_pointwise(_to_dtype, override_dtype=dtype)(x)


def register_pointwise(aten_fn, name=None, broadcast=True, type_promote=True):
    """A pointwise function that maps prim.{name} to inputs"""
    name = name or aten_fn.__name__

    @register_lowering(aten_fn, broadcast=broadcast, type_promote=type_promote)
    @make_pointwise
    def fn(*args, **kwargs):
        return getattr(prim, name)(*args, **kwargs)

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


register_pointwise(aten.add)


class GraphLowering(torch.fx.Interpreter):
    def symbolic_sizes_strides(self, ex: torch.Tensor):
        """
        Support dynamic shapes and dynamic strides by assigning variables
        to each dimension.  We duck-shape tensors, so if two tensors
        have the same size they get assigned the same symbolic variable.
        """
        size = [self.sizevars[i] for i in ex.size()]
        stride = [None] * len(size)
        for i, val in enumerate(ex.stride()):
            if val in (0, 1):
                stride[i] = Integer(val)

        while any(x is None for x in stride):
            candidates = {
                ex.size(i) * ex.stride(i): size[i] * stride[i]
                for i in range(len(size))
                if stride[i] is not None
            }
            for i in chain(reversed(range(len(stride))), range(len(stride))):
                if stride[i] is None and ex.stride(i) in candidates:
                    stride[i] = candidates[ex.stride(i)]
                    candidates[ex.size(i) * ex.stride(i)] = size[i] * stride[i]
            if any(x is None for x in stride):
                # bind the smallest unbound stride to a new variable
                val, i = sorted(
                    [(ex.stride(i), i) for i in range(len(stride)) if stride[i] is None]
                )[0]
                stride[i] = self.sizevars[val]

        # print(f"{ex.size()} = {size}, {ex.stride()} = {stride}")
        return size, stride

    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)
        self.sizevars = SizeVarAllocator("s")
        self.graph_inputs = collections.OrderedDict()
        self.graph_outputs = None
        self.device = None

    def placeholder(self, target, args, kwargs):
        example: torch.Tensor = super().placeholder(target, args, kwargs)
        if self.device is None:
            self.device = example.device
        assert example.device == self.device
        sizes, strides = self.symbolic_sizes_strides(example)
        # TODO(jansel): handle input aliasing
        data = InputBuffer(
            FixedLayout(example.device, example.dtype, sizes, strides),
            len(self.graph_inputs),
            target,
        )
        tensor = TensorBox.create(data)
        self.graph_inputs[target] = tensor
        return tensor

    def call_function(self, target, args, kwargs):
        return lowerings[target](*args, **kwargs)

    def get_attr(self, target, args, kwargs):
        # this is a constant
        value = getattr(self.module, target)
        assert value.shape == ()
        return Constant(value.item(), value.dtype, value.device)

    def call_module(self, target, args, kwargs):
        assert False

    def output(self, target, args, kwargs):
        result = super().output(target, args, kwargs)
        assert isinstance(result, tuple)
        self.graph_outputs = collections.OrderedDict(
            (f"out{i}", r) for i, r in enumerate(result)
        )

    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)
        num_users = len(set(n.users))
        if num_users > 1:
            # TODO(jansel): introduce a store vs inline choice
            print(n, set(n.users))
            result.mark_reuse(n.users)
        return result

    def codegen(self):
        from .codegen import CppPointwiseKernel
        from .codegen import TritonPointwiseKernel

        backends = {"cpu": CppPointwiseKernel, "cuda": TritonPointwiseKernel}
        backend_cls = backends[self.device.type]

        with backend_cls() as kernel:
            for name, node in self.graph_outputs.items():
                node.codegen(kernel, name)

        schedule = ScheduleCodeGen(self)
        schedule.define_kernel("kernel0", kernel)
        schedule.call_kernel("kernel0", kernel)
        return schedule.generate()

    def compile_to_fn(self):
        from .codecache import PyCodeCache

        code = self.codegen()
        if config.debug:
            print(code)
        return PyCodeCache.load(code).call
