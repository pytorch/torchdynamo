import dataclasses
from typing import Any
from typing import Callable
from typing import List

import sympy
import torch
from sympy import Expr
from sympy import Integer

from .virtualized_ops import ops


class IRNode(object):
    pass


@dataclasses.dataclass
class Layout(IRNode):
    device: torch.device
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
            return sum(a * b for a, b in zip(index, stride)) + offset

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
            None, None, sizes, FixedLayout.default_strides(sizes)
        ).make_indexer()


class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""

    pass


@dataclasses.dataclass
class Loops(IRNode):
    ranges: List[Expr]
    inner_fn: Callable

    def get_size(self):
        return self.ranges

    @classmethod
    def create(cls, *args, **kwargs):
        return TensorBox.create(cls(*args, **kwargs))


@dataclasses.dataclass
class TypedLoops(Loops):
    device: torch.device
    dtype: torch.dtype

    def get_dtype(self):
        return self.dtype

    def get_device(self):
        return self.device

    def get_stride(self):
        return FixedLayout.default_strides(self.get_size())


class UnrealizedBuffer(TypedLoops):
    def make_loader(self):
        return self.inner_fn

    def get_reduction_size(self):
        return []

    def get_reduction_type(self):
        return None

    def store_output(self, output_name, vars):
        indexer = FixedLayout.indexer_from_sizes(self.get_size())
        return ops.store(output_name, indexer(vars), self.inner_fn(vars))


@dataclasses.dataclass
class Reduction(TypedLoops):
    reduction_size: List[Expr]
    reduction_type: str

    def get_reduction_size(self):
        return self.reduction_size

    def get_reduction_type(self):
        return self.reduction_type

    def store_reduction(self, output_name, vars, reduction_vars):
        indexer = FixedLayout.indexer_from_sizes(self.get_size())
        return ops.store(
            output_name, indexer(vars), self.inner_fn(vars, reduction_vars)
        )


@dataclasses.dataclass
class BaseView(IRNode):
    data: IRNode


@dataclasses.dataclass
class ExpandView(BaseView):
    size: List[Expr]

    def get_size(self):
        return self.size

    def get_dtype(self):
        return self.data.get_dtype()

    def get_device(self):
        return self.data.get_device()

    def make_loader(self):
        target = self.get_size()
        actual = self.data.get_size()
        skip = len(target) - len(actual)
        inner = self.data.make_loader()

        def load(index):
            index = list(index[skip:])
            assert len(index) == len(actual)
            for i in range(len(actual)):
                if actual[i] == 1:
                    # zero out broadcast dimension
                    index[i] = sympy.Integer(0)
            return inner(index)

        return load


@dataclasses.dataclass
class Constant(IRNode):
    value: Any
    dtype: torch.dtype
    device: torch.device

    def get_size(self):
        return ()

    def get_dtype(self):
        return self.dtype

    def get_device(self):
        return self.device

    def make_loader(self):
        def loader(index):
            return ops.constant(self.value, self.dtype)

        return loader

    def mark_reuse(self, users):
        pass


@dataclasses.dataclass
class Buffer(IRNode):
    layout: Layout

    def get_size(self):
        return self.layout.size

    def get_dtype(self):
        return self.layout.dtype

    def get_device(self):
        return self.layout.device

    def make_loader(self):
        indexer = self.layout.make_indexer()

        def loader(index):
            return ops.load(self.name, indexer(index))

        return loader


@dataclasses.dataclass
class InputBuffer(Buffer):
    index: int
    name: str

    def get_stride(self):
        return self.layout.stride


@dataclasses.dataclass
class MutableBox(IRNode):
    """
    TensorBox / StorageBox allow in-place mutation of Tensors
    """

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
