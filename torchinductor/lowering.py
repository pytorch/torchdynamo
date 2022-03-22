import collections
import dataclasses
from itertools import chain
from typing import Dict
from typing import List
from typing import Optional

import torch.fx
from sympy import Expr
from sympy import Integer
from sympy import Symbol


class SizeVarAllocator(object):
    def __init__(self, prefix="s", zero_one_const=True):
        super().__init__()
        self.prefix = prefix
        self.val_to_var: Dict[int, Expr] = {0: Integer(0), 1: Integer(1)}
        self.var_to_val: Dict[Expr, int] = collections.OrderedDict()
        if not zero_one_const:
            self.val_to_var.clear()

    def __getitem__(self, val):
        if val in self.val_to_var:
            return self.val_to_var[val]
        var = Symbol(f"{self.prefix}{len(self.var_to_val)}")
        self.val_to_var[val] = var
        self.var_to_val[var] = val
        return var


class Layout:
    pass


@dataclasses.dataclass
class FixedLayout(Layout):
    """A Tensor layout we cannot change"""
    stride: List[Expr]
    offset: Expr = Integer(0)


class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""

    pass


class IRNode(object):
    pass



@dataclasses.dataclass
class VData(IRNode):
    dtype: torch.dtype


@dataclasses.dataclass
class VInputData(VData):
    index: int
    name: str
    layout: Layout

@dataclasses.dataclass
class VReusedData(VData):
    inner: VData

@dataclasses.dataclass
class VStorage(IRNode):
    data: VData


@dataclasses.dataclass
class VTensor(IRNode):
    storage: VStorage
    size: List[Expr]

    def mark_reuse(self, users):
        self.storage.data = VReusedData(self.storage.data)


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
                val, i = sorted([(ex.stride(i), i)
                                 for i in range(len(stride))
                                 if stride[i] is None
                                 ])[0]
                stride[i] = self.sizevars[val]

        print(f"{ex.size()} = {size}, {ex.stride()} = {stride}")
        return size, stride

    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)
        self.sizevars = SizeVarAllocator("s")
        self.graph_inputs = []

    def placeholder(self, target, args, kwargs):
        example: torch.Tensor = super().placeholder(target, args, kwargs)
        # TODO(jansel): handle input aliasing
        data = VInputData(example.dtype, len(self.graph_inputs), target)
        storage = VStorage(data)
        sizes, strides = self.symbolic_sizes_strides(example)
        tensor = VTensor(storage, FixedLayout(strides), sizes)
        self.graph_inputs.append(tensor)
        return tensor

    def call_function(self, target, args, kwargs):
        pass

    def get_attr(self, target, args, kwargs):
        assert False

    def call_module(self, target, args, kwargs):
        assert False

    def output(self, target, args, kwargs):
        assert False

    def run_node(self, n : torch.fx.Node):
        result = super().run_node(n)
        num_users = len(set(n.users))
        if num_users > 0:
            result.mark_reuse(n.users)
        return result



