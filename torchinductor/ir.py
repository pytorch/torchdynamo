import dataclasses
import textwrap
from typing import Callable
from typing import List

import torch
from sympy import Expr
from sympy import Integer
from sympy import Symbol

from .virtualized import prim

class IRNode(object):
    pass


@dataclasses.dataclass
class Layout(IRNode):
    dtype: torch.dtype
    size: List[Expr]


@dataclasses.dataclass
class FixedLayout(Layout):
    """A Tensor layout we cannot change"""

    stride: List[Expr]
    offset: Expr = Integer(0)

    def make_indexer(self):
        """A closure containing math to read a given element"""
        stride = list(self.stride)
        offset = self.offset

        def indexer(index):
            assert len(index) == len(stride)
            return sum(l * r for l, r in zip(index, stride)) + offset

        return indexer


class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""

    pass


@dataclasses.dataclass
class Loops(IRNode):
    ranges: List[Expr]
    inner_fn: Callable

    def cpp(self):
        loop_headers = []
        vars = [Symbol(f"i{n}") for n in range(len(self.ranges))]
        for var, size in zip(vars, self.ranges):
            loop_headers.append(f"for(int {var}=0; {var}<{size}; ++{var})")
        inner = self.inner_fn(vars)
        return textwrap.dedent(
            """
        {}
        {{
        {}
        }}
        """
        ).format("\n".join(loop_headers), inner)

class UnrealizedBuffer(Loops):
    pass



@dataclasses.dataclass
class Buffer(IRNode):
    layout: Layout

    def get_ranges(self):
        return self.layout.size

    def make_loader(self):
        indexer = self.layout.make_indexer()

        def loader(index):
            return prim.load(self.name, indexer(index))

        return loader


@dataclasses.dataclass
class InputBuffer(Buffer):
    index: int
    name: str


@dataclasses.dataclass
class ComputedBuffer(Buffer):
    contents = IRNode



@dataclasses.dataclass
class MutableBox(IRNode):
    data: IRNode

    def __getattr__(self, name):
        fn = getattr(self.data, name)
        if callable(fn):
            return fn
        raise AttributeError(f"{type(self.data).__name__}.{name} not callable")


class TensorBox(MutableBox):
    @staticmethod
    def create(data):
        return TensorBox(StorageBox(data))

    def mark_reuse(self, users):
        pass


class StorageBox(MutableBox):
    pass


@dataclasses.dataclass
class View(IRNode):
    storage: StorageBox
