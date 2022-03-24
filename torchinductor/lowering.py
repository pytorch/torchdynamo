import collections
import operator
import textwrap
from itertools import chain
from typing import List

import torch
import torch.fx
from sympy import Integer

from .ir import FixedLayout
from .ir import InputBuffer
from .ir import Loops
from .ir import TensorBox
from .shapes import SizeVarAllocator
from .virtualized import prim

lowerings = {}
aten = torch.ops.aten


def register_lowering(aten_fn, broadcast=True, type_promote=True):
    def _reg(decomp_fn):
        lowerings[aten_fn] = decomp_fn
        return decomp_fn

    return _reg


def register_pointwise(aten_fn, name=None):
    name = name or aten_fn.__name__

    @register_lowering(aten_fn, broadcast=True, type_promote=True)
    def inner(*inputs: List[TensorBox]):
        loaders = [x.make_loader() for x in inputs]
        return TensorBox.create(
            Loops(
                inputs[0].get_ranges(),
                lambda index: getattr(prim, name)(*[load(index) for load in loaders]),
            )
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
        self.graph_outputs = super().output(target, args, kwargs)
        return self.graph_outputs

    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)
        num_users = len(set(n.users))
        if num_users > 0:
            result.mark_reuse(n.users)
        return result

    def cpp(self):
        args = ", ".join(self.graph_inputs.keys())
        code = "\n".join([x.cpp() for x in self.graph_outputs])
        return textwrap.dedent(
            """
        void kernel({args}) {{
        {code}
        }}
        """
        ).format(args=args, code=code)
