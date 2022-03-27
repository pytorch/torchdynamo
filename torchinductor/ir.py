import dataclasses
from typing import Callable
from typing import List

import sympy
import torch
from sympy import Expr
from sympy import Integer

from .codegen import PointwiseKernel
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

    @staticmethod
    def default_strides(sizes):
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.Integer(1)]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    @staticmethod
    def indexer_from_sizes(sizes):
        return FixedLayout(
            None, sizes, FixedLayout.default_strides(sizes)
        ).make_indexer()


class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""

    pass


@dataclasses.dataclass
class Loops(IRNode):
    ranges: List[Expr]
    inner_fn: Callable

    @classmethod
    def create(cls, *args, **kwargs):
        return TensorBox.create(cls(*args, **kwargs))


@dataclasses.dataclass
class UnrealizedBuffer(Loops):
    dtype: torch.dtype

    def get_size(self):
        return self.ranges

    def get_dtype(self):
        return self.dtype


@dataclasses.dataclass
class BaseView(IRNode):
    data: IRNode


@dataclasses.dataclass
class ExpandView(BaseView):
    size: List[Expr]

    def get_size(self):
        return self.size

    def make_indexer(self):
        target = self.get_size()
        actual = self.data.get_size()
        skip = len(target) - len(actual)
        inner = self.data.make_indexer()

        def indexer(index):
            index = list(index[skip:])
            assert len(index) == len(actual)
            for i in len(actual):
                if actual[i] == 1:
                    index[i] = sympy.Integer(0)
            return inner(index)

        return indexer


@dataclasses.dataclass
class Buffer(IRNode):
    layout: Layout

    def get_size(self):
        return self.layout.size

    def get_dtype(self):
        return self.layout.dtype

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

    def codegen(self, kernel: PointwiseKernel, output_name):
        vars = kernel.set_ranges(self.get_size())
        indexer = FixedLayout.indexer_from_sizes(self.get_size())
        prim.store(output_name, indexer(vars), self.inner_fn(vars))


class StorageBox(MutableBox):
    pass


@dataclasses.dataclass
class View(IRNode):
    storage: StorageBox
