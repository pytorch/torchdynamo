import collections
import functools
import itertools
import operator
import textwrap
from itertools import chain
from typing import List

import sympy
import torch
import torch.fx
from sympy import Integer

from .codegen import CppPointwiseKernel
from .codegen import TritonPointwiseKernel
from .ir import FixedLayout
from .ir import InputBuffer
from .ir import TensorBox
from .ir import UnrealizedBuffer
from .shapes import SizeVarAllocator
from .virtualized import prim

lowerings = {}
aten = torch.ops.aten


def _register_lowering(aten_fn, decomp_fn, broadcast=False, type_promote=True):
    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        assert not any(isinstance(x, TensorBox) for x in kwargs.values())
        args = list(args)
        tensor_args = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]

        if type_promote and tensor_args:
            dtype = functools.reduce(
                torch.promote_types, [args[i].get_dtype() for i in tensor_args]
            )
            for i in tensor_args:
                args[i] = to_dtype(args[i], dtype)

        if broadcast and tensor_args:
            for i, x in zip(
                tensor_args, broadcast_tensors(*[args[i] for i in tensor_args])
            ):
                args[i] = x

        return decomp_fn(*args, **kwargs)

    lowerings[aten_fn] = wrapped
    return wrapped


def register_lowering(aten_fn, broadcast=True, type_promote=True):
    return functools.partial(
        _register_lowering, aten_fn, broadcast=broadcast, type_promote=type_promote
    )


def broadcast_shapes(a, b):
    output = []
    for a, b in itertools.zip_longest(
        reversed(a), reversed(b), fillvalue=sympy.Integer(1)
    ):
        if a == 1:
            output.append(b)
        elif b == 1:
            output.append(a)
        elif len(str(b)) < len(str(a)):
            output.append(b)
        else:
            output.append(a)
    return tuple(reversed(output))


@register_lowering(aten.broadcast_tensors, broadcast=False, type_promote=False)
def broadcast_tensors(*inputs):
    target = functools.reduce(broadcast_shapes, [x.get_size() for x in inputs], ())
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
    assert False


def to_dtype(x: TensorBox, dtype: torch.dtype):
    assert x.get_dtype() == dtype, "TODO(jansel): type promotion"
    return x


def register_pointwise(aten_fn, name=None):
    name = name or aten_fn.__name__

    @register_lowering(aten_fn, broadcast=True, type_promote=True)
    def inner(*inputs: List[TensorBox]):
        loaders = [x.make_loader() for x in inputs]
        return UnrealizedBuffer.create(
            inputs[0].get_size(),
            lambda index: getattr(prim, name)(*[load(index) for load in loaders]),
            dtype=inputs[0].get_dtype(),
        )

    return inner


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
        self.graph_outputs = []

    def placeholder(self, target, args, kwargs):
        example: torch.Tensor = super().placeholder(target, args, kwargs)
        sizes, strides = self.symbolic_sizes_strides(example)
        # TODO(jansel): handle input aliasing
        data = InputBuffer(
            FixedLayout(example.dtype, sizes, strides),
            len(self.graph_inputs),
            target,
        )
        tensor = TensorBox.create(data)
        self.graph_inputs[target] = tensor
        return tensor

    def call_function(self, target, args, kwargs):
        return lowerings[target](*args, **kwargs)

    def get_attr(self, target, args, kwargs):
        assert False

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
        if num_users > 0:
            # TODO(jansel): introduce a store vs inline choice
            result.mark_reuse(n.users)
        return result

    def codegen(self, cls):
        with cls() as kernel:
            for name, node in self.graph_outputs.items():
                node.codegen(kernel, name)

        return kernel.generate(self)

    def cpp(self):
        return self.codegen(CppPointwiseKernel)

    def triton(self):
        return self.codegen(TritonPointwiseKernel)
