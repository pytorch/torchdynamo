import dataclasses
import functools
import itertools
import logging
import textwrap
from collections import OrderedDict
from functools import partial
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from unittest.mock import patch

import numpy
import sympy
import torch.fx
import torch.utils._pytree as pytree
from sympy import Expr
from sympy import Integer

from . import config
from . import dependencies
from .codegen.common import _simplify_loops
from .dependencies import extract_read_writes
from .utils import sympy_product
from .virtualized import V
from .virtualized import ops

log = logging.getLogger(__name__)
indent = functools.partial(textwrap.indent, prefix="  ")


def inverse_reorder(order):
    inv_order = dict(zip(order, range(len(order))))

    def reindex(index):
        assert len(index) == len(inv_order)
        return [index[inv_order[i]] for i in range(len(index))]

    return reindex


def fuse_reindexing(reindex1, reindex2):
    def reindex(index):
        return reindex1(reindex2(index))

    return reindex


class ModularIndexing(sympy.Function):
    """
    ModularIndexing(a, b, c) => (a // b) % c
    """

    nargs = (3,)

    @classmethod
    def eval(cls, base, divisor, modulus):
        if base == 0 or modulus == 1:
            return sympy.Integer(0)

        if (
            isinstance(base, sympy.Integer)
            and isinstance(divisor, sympy.Integer)
            and isinstance(modulus, sympy.Integer)
        ):
            return (base // divisor) % modulus

        if divisor != 1:
            gcd = sympy.gcd(base, divisor)
            if gcd != 1:
                return ModularIndexing(base / gcd, divisor / gcd, modulus)

        if isinstance(base, sympy.Add):
            new_terms = []
            for term in base.args:
                if sympy.gcd(term, modulus * divisor) != modulus * divisor:
                    new_terms.append(term)
            if len(new_terms) != len(base.args):
                return ModularIndexing(sum(new_terms), divisor, modulus)


class IndexingDiv(sympy.Function):
    """
    a // b used in indexing where we need to be careful about simplification.
    We don't use sympy.FloorDiv to bypass some simplification rules.
    """

    nargs = (2,)

    @classmethod
    def eval(cls, base, divisor):
        if base == 0:
            return sympy.Integer(0)
        if divisor == 1:
            return base
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return base // divisor
        gcd = sympy.gcd(base, divisor)
        if gcd != 1:
            return IndexingDiv(base / gcd, divisor / gcd)


class CleanDiv(IndexingDiv):
    """
    Div where we can assume no rounding.
    This is to enable future optimizations.
    """

    pass


def is_triton(device):
    # TODO(jansel): a config check once we have multi-backend
    return device.type == "cuda"


class IRNode(object):
    def str_helper(self, lines):
        lines = indent(",\n".join(map(str, lines)))
        return f"{type(self).__name__}(\n{lines}\n)"

    def is_user_of(self, name):
        return any(name == dep.name for dep in self.get_reads())

    def get_numel(self):
        return sympy_product(self.get_size())


@dataclasses.dataclass
class Loops(IRNode):
    device: torch.device
    dtype: torch.dtype
    inner_fn: Callable
    ranges: List[Expr]

    def __str__(self, names=("ranges",)):
        return self.str_helper(
            [
                f"'{self.device.type}'",
                str(self.dtype),
                self.inner_fn_str(),
            ]
            + [f"{name}={getattr(self, name)}" for name in names]
        )

    __repr__ = __str__

    def get_dtype(self):
        return self.dtype

    def get_device(self):
        return self.device

    def get_size(self):
        return self.ranges

    def is_extern(self):
        return False

    @classmethod
    def create(cls, *args, **kwargs):
        return TensorBox.create(cls(*args, **kwargs))

    @staticmethod
    def _index(ranges, prefix="i"):
        return [
            sympy.Integer(0) if s == 1 else sympy.Symbol(f"{prefix}{n}")
            for n, s in enumerate(ranges)
        ]

    def inner_fn_str(self):
        try:
            with V.set_ops_handler(V.MockHandler()), patch.object(
                FlexibleLayout, "allow_indexing", True
            ):
                return self.inner_fn(self._index(self.ranges))
        except Exception as e:
            return f"inner_fn(): {e}"

    def is_zero_elements(self):
        return any(r == 0 for r in self.ranges)

    def get_reads(self):
        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.get_reduction_type():
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                    self.get_reduction_size(),
                ).reads
            else:
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                ).reads


class Pointwise(Loops):
    def make_loader(self):
        return self.inner_fn

    def get_reduction_size(self):
        return []

    def get_reduction_type(self):
        return None

    def store_output(self, output_name, indexer, vars):
        return ops.store(output_name, indexer(vars), self.inner_fn(vars))

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(device, self.dtype, loader, self.ranges)


@dataclasses.dataclass
class Scatter(Pointwise):
    output_indexer: Callable[[List[Expr]], Expr]
    scatter_mode: str = None

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Scatter(
            device,
            self.dtype,
            loader,
            self.ranges,
            self.output_indexer,
            self.scatter_mode,
        )

    def store_output(self, output_name, indexer, vars):
        return ops.store(
            output_name,
            indexer(self.output_indexer(vars)),
            self.inner_fn(vars),
            mode=self.scatter_mode,
        )


@dataclasses.dataclass
class Reduction(Loops):
    reduction_ranges: List[Expr]
    reduction_type: str

    def __str__(self):
        return Loops.__str__(
            self, names=("ranges", "reduction_ranges", "reduction_type")
        )

    __repr__ = __str__

    def get_reduction_size(self):
        return self.reduction_ranges

    def get_reduction_type(self):
        return self.reduction_type

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        return ops.reduction(
            output_name,
            self.dtype,
            self.reduction_type,
            indexer(vars),
            self.inner_fn(vars, reduction_vars),
        )

    def index_length(self):
        return len(self.ranges) + len(self.reduction_ranges)

    def inner_fn_str(self):
        try:
            with V.set_ops_handler(V.MockHandler()), patch.object(
                FlexibleLayout, "allow_indexing", True
            ):
                return self.inner_fn(
                    self._index(self.ranges), self._index(self.reduction_ranges, "r")
                )
        except Exception as e:
            return f"inner_fn(): {e}"

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Reduction(
            device,
            self.dtype,
            loader,
            self.ranges,
            self.reduction_ranges,
            self.reduction_type,
        )

    @classmethod
    def create(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fn: Callable,
        ranges: List[Expr],
        reduction_ranges: List[Expr],
        reduction_type: str,
    ):
        reduction_numel = sympy_product(reduction_ranges)
        if reduction_numel == 1:
            # this reduction is actually a pointwise op
            def fn(index):
                reduction_index = [sympy.Integer(0) for _ in reduction_ranges]
                return inner_fn(index, reduction_index)

            return Pointwise.create(device, dtype, fn, ranges)

        if is_triton(device):
            reduction_numel_hint = V.graph.sizevars.size_hint(reduction_numel)
            numel_hint = V.graph.sizevars.size_hint(sympy_product(ranges))
            if reduction_numel_hint > 8192 and numel_hint == 1:
                # triton doesn't support reduce to single element well, so break it up
                split = 128
                return cls.create_multilayer(
                    device,
                    dtype,
                    inner_fn,
                    ranges,
                    reduction_ranges,
                    reduction_type,
                    split,
                )

        return TensorBox.create(
            Reduction(
                device,
                dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
            )
        )

    @staticmethod
    def default_value(reduction_type, dtype):
        return {"sum": 0, "max": float("-inf"), "min": float("inf"), "any": 0}[
            reduction_type
        ]

    @classmethod
    def create_multilayer(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fn: Callable,
        ranges: List[Expr],
        reduction_ranges: List[Expr],
        reduction_type: str,
        split,
    ):
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        reduction_numel = sympy_product(reduction_ranges)

        # TODO(jansel): convert this to dynamic shapes
        # TODO(jansel): realize the reduction so we can do dynamic indexing
        reduction_ranges = [
            sympy.Integer(V.graph.sizevars.guard_static_shape(s))
            for s in reduction_ranges
        ]
        reduction_numel = sympy.Integer(
            V.graph.sizevars.guard_static_shape(reduction_numel)
        )

        if V.graph.sizevars.size_hint(reduction_numel) % split == 0:
            need_mask = False
        else:
            need_mask = True

        split = sympy.Integer(split)
        block_size = IndexingDiv(reduction_numel + (split - 1), split)

        reindex = View.dynamic_reshape_indexer(reduction_ranges, [reduction_numel])

        def wrapper_fn(index, reduction_index):
            (reduction_index,) = reduction_index
            *new_index, reduction_block = index
            indices = block_size * reduction_block + reduction_index

            def body():
                return inner_fn(new_index, reindex([indices]))

            if need_mask:
                mask = ops.lt(
                    ops.index_expr(indices, torch.int32),
                    ops.index_expr(reduction_numel, torch.int32),
                )
                return ops.masked(mask, body, cls.default_value(reduction_type, dtype))
            else:
                return body()

        # triton will automatically compute reductions in fp32 if reducing over fp16/bf16
        # within the kernel. keep the intermediate in fp32 so as to keep the whole reduction
        # in fp32 and not reduce precision by breaking up the kernel into multiple layers
        intermediate_dtype = (
            dtype if dtype not in (torch.float16, torch.bfloat16) else torch.float
        )
        intermediate = Reduction.create(
            device,
            intermediate_dtype,
            wrapper_fn,
            [*ranges, split],
            [block_size],
            reduction_type,
        )
        intermediate.realize()
        intermediate_loader = intermediate.make_loader()

        def intermediate_fn(index, reduction_index):
            return intermediate_loader([*index, *reduction_index])

        return TensorBox.create(
            Reduction(
                device,
                dtype,
                intermediate_fn,
                ranges,
                [split],
                reduction_type,
            )
        )


def is_storage_and_layout(x):
    try:
        as_storage_and_layout(x, freeze=False)
        return True
    except NotImplementedError:
        return False


def is_contiguous_storage_and_layout(x):
    try:
        buffer, layout = as_storage_and_layout(x, freeze=False)
        return layout.is_contiguous()
    except NotImplementedError:
        return False


def as_storage_and_layout(x, freeze=True, want_contiguous=False):
    """Try to simplify x into a StorageBox and a Layout"""
    if isinstance(x, TensorBox):
        return as_storage_and_layout(
            x.data, freeze=freeze, want_contiguous=want_contiguous
        )
    if isinstance(x, StorageBox) and isinstance(x.data, Buffer):
        if freeze:
            if want_contiguous:
                x.data.freeze_layout()
            else:
                x.data.decide_layout()
        return x, x.data.layout
    if isinstance(x, ReinterpretView):
        buffer, _ = as_storage_and_layout(
            x.data, freeze=freeze, want_contiguous=want_contiguous
        )
        return buffer, x.layout
    raise NotImplementedError


as_contiguous_storage_and_layout = functools.partial(
    as_storage_and_layout, want_contiguous=True
)


@dataclasses.dataclass
class BaseView(IRNode):
    data: IRNode

    def get_dtype(self):
        return self.data.get_dtype()

    def get_device(self):
        return self.data.get_device()

    def get_name(self):
        return self.data.get_name()

    def mark_reuse(self, users):
        return self.data.mark_reuse(users)

    def realize(self):
        return self.data.realize()

    def get_storage_numel(self):
        return self.data.get_storage_numel()

    def is_extern(self):
        return self.data.is_extern()

    def get_reads(self):
        with patch.object(FlexibleLayout, "allow_indexing", True):
            return extract_read_writes(
                self.make_loader(),
                self.get_size(),
            ).reads


@dataclasses.dataclass
class ExpandView(BaseView):
    size: List[Expr]

    @staticmethod
    def _normalize_size(x, new_size):
        """Replace `-1` with correct sizes"""
        new_size = list(map(sympy.expand, new_size))
        old_size = x.get_size()
        old_size = [None] * (len(new_size) - len(old_size)) + list(old_size)
        assert len(new_size) == len(old_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                assert old_size[i] is not None
                new_size[i] = old_size[i]
        return new_size

    @classmethod
    def create(cls, x, new_size):
        new_size = cls._normalize_size(x, new_size)

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            skip = len(new_size) - len(old_layout.size)
            assert skip >= 0
            new_stride = [sympy.Integer(0)] * skip
            for stride, size in zip(old_layout.stride, old_layout.size):
                new_stride.append(stride if size != 1 else sympy.Integer(0))
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                list(new_size),
                new_stride,
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        return ExpandView(x, new_size)

    def get_size(self):
        return self.size

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
class PermuteView(BaseView):
    dims: List[Expr]

    @classmethod
    def create(cls, x, dims):
        assert set(dims) == set(range(len(dims)))

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                [old_layout.size[i] for i in dims],
                [old_layout.stride[i] for i in dims],
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        return PermuteView(x, dims)

    def get_size(self):
        assert set(self.dims) == set(range(len(self.dims)))
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def make_loader(self):
        inner = self.data.make_loader()
        inv = {j: i for i, j in enumerate(self.dims)}
        inv = [inv[i] for i in range(len(self.dims))]
        assert set(inv) == set(range(len(self.dims)))

        def load(index):
            index = [index[i] for i in inv]
            return inner(index)

        return load


class SqueezeView(BaseView):
    @classmethod
    def create(cls, x):

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_size = []
            new_stride = []
            for size, stride in zip(old_layout.size, old_layout.stride):
                if size != 1:
                    new_size.append(size)
                    new_stride.append(stride)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        # redirect to a generic view
        return View.create(x, [s for s in x.get_size() if s != 1])

    @staticmethod
    def squeezer(size):
        new_size = [s for s in size if s != 1]
        not_one = [i for i, s in enumerate(size) if s != 1]
        length = len(size)

        def reindex(index):
            assert len(index) == len(not_one), f"{index} {not_one}"
            new_index = [sympy.Integer(0)] * length
            for idx, s in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)

        return new_size, reindex

    def __init__(self, data):
        assert False, "use SqueezeView.create()"


@dataclasses.dataclass
class View(BaseView):
    size: List[Expr]
    reindex: Callable

    def make_indexer(self):
        base_indexer = self.data.make_indexer()

        def indexer(idx):
            return base_indexer(self.reindex(idx))

        return indexer

    @staticmethod
    def handle_negative_index(idx, size):
        idx = sympy.expand(idx)
        size = sympy.expand(size)
        sizevars = V.graph.sizevars
        if sizevars.size_hint(idx) < 0:
            sizevars.guard_lt(idx, 0)
            idx = idx + size
        return idx

    def reindex_str(self):
        index_old = [sympy.Symbol(f"i{n}") for n in range(len(self.size))]
        index_new = list(self.reindex(index_old))
        return f"lambda {', '.join(map(str, index_old))}: {index_new}"

    def __str__(self):
        return self.str_helper(
            [self.data, f"size={self.size}", f"reindex={self.reindex_str()}"]
        )

    __repr__ = __str__

    @classmethod
    def create(cls, x, new_size):
        assert isinstance(new_size, (tuple, list))
        old_size, new_size = cls.resolve_negative_size(x.get_size(), new_size)

        if is_contiguous_storage_and_layout(x):
            storage, old_layout = as_contiguous_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                FlexibleLayout.contiguous_strides(new_size),
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        reindex = cls.dynamic_reshape_indexer(old_size, new_size)
        return cls(x, tuple(new_size), reindex)

    @staticmethod
    def resolve_negative_size(old_size, new_size):
        new_size = [
            sympy.expand(x).subs(V.graph.sizevars.replacements) for x in new_size
        ]
        old_size = [
            sympy.expand(x).subs(V.graph.sizevars.replacements) for x in old_size
        ]

        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.Integer(1)
                new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
                break

        V.graph.sizevars.guard_equals(sympy_product(old_size), sympy_product(new_size))
        return old_size, new_size

    @classmethod
    def dynamic_reshape_indexer(cls, old_size, new_size):
        try:
            reindex = cls._dynamic_reshape_indexer(old_size, new_size)
        except (AssertionError, IndexError):
            # optimistic algorithm failed, lets do a fallback
            flat = [sympy_product(old_size)]
            reindex1 = cls._dynamic_reshape_indexer(old_size, flat)
            reindex2 = cls._dynamic_reshape_indexer(flat, new_size)
            reindex = fuse_reindexing(reindex1, reindex2)
        return reindex

    @staticmethod
    def _dynamic_reshape_indexer(old_size, new_size):
        """
        Perform a reshape entirely by modifying indexing math
        """
        size_hint = V.graph.sizevars.size_hint
        vars = [sympy.Symbol(f"view{i}") for i in range(len(new_size))]

        stack_new = list(zip(vars, new_size))
        stack_old = list(old_size)

        view_expr = []
        while stack_new and stack_old:
            size_old = stack_old.pop()
            var, size_new = stack_new.pop()
            if size_old == 1:
                view_expr.append(sympy.Integer(0))
                stack_new.append((var, size_new))  # re-add
            elif size_new == 1:
                stack_old.append(size_old)  # re-add
            elif size_hint(size_new) == size_hint(size_old):
                view_expr.append(var)
                V.graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) < size_hint(size_old):
                while size_hint(size_new) < size_hint(size_old):
                    var2, size_new2 = stack_new.pop()
                    var = var2 * size_new + var
                    size_new = size_new * size_new2
                view_expr.append(var)
                V.graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) > size_hint(size_old):
                divisor = sympy.Integer(1)
                modulus = size_old
                view_expr.append(ModularIndexing(var, divisor, modulus))
                divisor = divisor * modulus
                while size_hint(size_new) > size_hint(size_old):
                    modulus = stack_old.pop()
                    view_expr.append(ModularIndexing(var, divisor, modulus))
                    divisor = divisor * modulus
                    size_old = size_old * modulus
                V.graph.sizevars.guard_equals(size_new, size_old)
            else:
                assert False

        while stack_old:
            size_old = stack_old.pop()
            assert size_old == 1
            view_expr.append(sympy.Integer(0))

        while stack_new:
            var, size_new = stack_new.pop()
            assert size_new == 1

        view_expr = list(reversed(view_expr))
        assert len(view_expr) == len(old_size)

        def reindex(index):
            assert len(index) == len(vars), (len(index), len(vars))
            replacements = dict(zip(vars, index))
            return tuple(x.subs(replacements) for x in view_expr)

        return reindex

    def get_size(self):
        return self.size

    def make_loader(self):
        def load(index):
            return inner(self.reindex(index))

        inner = self.data.make_loader()
        return load


@dataclasses.dataclass
class ReinterpretView(BaseView):
    """Pretend our storage has a different layout"""

    layout: "Layout"

    def __str__(self):
        return self.str_helper(
            [
                self.data,
                self.layout,
            ]
        )

    __repr__ = __str__

    def get_name(self):
        return self.data.get_name()

    def get_device(self):
        return self.layout.device

    def get_dtype(self):
        return self.layout.dtype

    def get_size(self):
        return self.layout.size

    def get_stride(self):
        return self.layout.stride

    def make_loader(self):
        def loader(index):
            indexer = self.layout.make_indexer()
            upcast = (
                self.get_dtype() == torch.float16 or self.get_dtype == torch.bfloat16
            )
            return ops.load(self.get_name(), indexer(index), upcast)

        return loader

    def make_indexer(self):
        return self.layout.make_indexer()

    def get_layout(self):
        return self.layout

    def freeze_layout(self):
        pass

    def codegen_reference(self):
        size = V.graph.sizevars.codegen_shape_tuple(self.layout.size)
        stride = V.graph.sizevars.codegen_shape_tuple(self.layout.stride)
        offset = V.graph.sizevars.codegen_sizevar(self.layout.offset)
        if offset != "0":
            return f"as_strided({self.get_name()}, {size}, {stride}, {offset})"
        return f"as_strided({self.get_name()}, {size}, {stride})"


class SliceView(View):
    @classmethod
    def create(cls, x, dim, start, end, step=1):
        step = sympy.expand(step)
        assert step > 0
        try:
            if start == 0 and end >= 2**63 and step == 1:
                return x
        except TypeError:
            pass

        sizevars = V.graph.sizevars
        new_size = list(x.get_size())

        start = cls.handle_negative_index(start, new_size[dim])
        end = cls.handle_negative_index(end, new_size[dim])

        end = sizevars.guard_min(end, new_size[dim])
        start = sizevars.guard_min(sizevars.guard_min(start, new_size[dim]), end)
        if start == 0 and sizevars.size_hint(end - new_size[dim]) == 0 and step == 1:
            sizevars.guard_equals(end, new_size[dim])
            return x

        new_size[dim] = IndexingDiv(end - start + (step - 1), step)

        if is_storage_and_layout(x):
            # Fast path
            storage, old_layout = as_storage_and_layout(x)
            new_stride = list(old_layout.stride)
            new_stride[dim] = new_stride[dim] * step
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset + old_layout.stride[dim] * start,
            )
            return ReinterpretView(storage, new_layout)

        def reindex(index):
            assert len(index) == len(new_size), f"wrong ndim {index} {new_size}"
            index = list(index)
            index[dim] = index[dim] * step + start
            return index

        # redirect to a generic view
        return SliceView(x, size=new_size, reindex=reindex)


class BaseConstant(IRNode):
    def get_size(self):
        return ()

    def get_dtype(self):
        return self.dtype

    def get_device(self):
        return self.device

    def mark_reuse(self, users):
        pass

    def get_reads(self):
        return ()

    def is_extern(self):
        return False


@dataclasses.dataclass
class Constant(BaseConstant):
    value: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self):
        def loader(index):
            return ops.constant(self.value, self.dtype)

        return loader


@dataclasses.dataclass
class IndexingConstant(BaseConstant):
    index: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self):
        def loader(index):
            return ops.index_expr(self.index, self.dtype)

        return loader


@dataclasses.dataclass
class Layout(IRNode):
    device: torch.device
    dtype: torch.dtype
    size: List[Expr]
    stride: List[Expr]
    offset: Expr = Integer(0)

    def __str__(self):
        offset = ""
        if self.offset != 0:
            offset = f", offset={self.offset}"
        return (
            f"{type(self).__name__}('{self.device.type}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}{offset})"
        )

    __repr__ = __str__

    def is_contiguous(self):
        for left, right, size in zip(
            self.stride, FlexibleLayout.contiguous_strides(self.size), self.size
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_transposed(self):
        for left, right, size in zip(
            self.stride,
            reversed(FlexibleLayout.contiguous_strides(self.size)),
            self.size,
        ):
            if size != 1 and left != right:
                return False
        return True

    def as_fixed(self):
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.stride,
            self.offset,
        )

    def make_indexer(self):
        assert (
            FlexibleLayout.allow_indexing
        ), f"convert {type(self).__name__} to FixedLayout first"
        return self.as_fixed().make_indexer()


class FixedLayout(Layout):
    """A Tensor layout we cannot change"""

    def make_indexer(self):
        """A closure containing math to read a given element"""

        def indexer(index):
            assert len(index) == len(self.stride) == len(self.size)
            result = self.offset
            for idx, stride, sz in zip(index, self.stride, self.size):
                if sz != 1:
                    result = result + idx * stride
            return result

        return indexer


class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""

    allow_indexing = False

    @staticmethod
    def contiguous_strides(sizes):
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.Integer(1)]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    @staticmethod
    def fill_ordered(sizes, order):
        """
        Create a stride based on the order the dimensions should be filled in.

        In this format, channels last would be:
            [1, 3, 2, 0]
        """
        assert set(range(len(sizes))) == set(order)
        next_stride = sympy.Integer(1)
        strides = [None] * len(order)

        for i in order:
            strides[i] = next_stride
            next_stride = next_stride * sizes[i]
        return strides

    @staticmethod
    def stride_ordered(sizes, order):
        """
        Create a stride based on the sorted order of a permuted range.

        In this format, channels last would be:
            [3, 0, 2, 1]
        """
        assert set(range(len(sizes))) == set(order)
        lookup = {pos: idx for idx, pos in enumerate(order)}
        fill_order = [lookup[i] for i in range(len(order))]
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    def as_stride_order(self, order):
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.stride_ordered(self.size, order),
            self.offset,
        )

    def as_fill_order(self, order):
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.fill_ordered(self.size, order),
            self.offset,
        )

    def __init__(self, device, dtype, size):
        super(FlexibleLayout, self).__init__(
            device, dtype, size, FlexibleLayout.contiguous_strides(size)
        )


class AliasedLayout(Layout):
    """Shares the same storage as another tensor"""

    def __init__(self, view: "ReinterpretView"):
        layout = view.get_layout()
        super().__init__(
            layout.device,
            layout.dtype,
            layout.size,
            layout.stride,
        )
        self.view = view

    def make_indexer(self):
        return self.as_fixed().make_indexer()


class MutationLayout(Layout):
    def __init__(self, target: IRNode):
        super().__init__(
            target.get_device(),
            target.get_dtype(),
            target.get_size(),
            None,
        )
        self.target = target

    @classmethod
    def realize_into(cls, src, dst):
        dst.realize()
        V.graph.realize_users_of(dst.get_name())

        if isinstance(src, TensorBox):
            src = src.data

        if not isinstance(src, StorageBox) or src.is_user_of(dst.get_name()):
            need_copy = True
        else:
            src.realize()
            need_copy = not isinstance(src.data.layout, FlexibleLayout)

        if need_copy:
            src = Pointwise.create(
                device=src.get_device(),
                dtype=src.get_dtype(),
                inner_fn=src.make_loader(),
                ranges=[
                    V.graph.sizevars.guard_equals(a, b)
                    for a, b in zip(src.get_size(), dst.get_size())
                ],
            ).data
            src.realize()

        assert isinstance(src.data.layout, FlexibleLayout)
        src.data.layout = MutationLayout(dst)
        return src.data

    def as_fixed(self):
        return self

    def make_indexer(self):
        return self.target.make_indexer()


@dataclasses.dataclass
class Buffer(IRNode):
    name: str
    layout: Layout

    def make_indexer(self):
        return self.layout.make_indexer()

    def get_name(self):
        assert self.name
        return self.name

    def get_device(self):
        return self.layout.device

    def get_dtype(self):
        return getattr(self.layout, "dtype", None)

    def get_size(self):
        return self.layout.size

    def get_stride(self):
        return self.layout.stride

    def get_layout(self):
        return self.layout

    def get_storage_numel(self):
        return self.get_numel()

    def is_extern(self):
        return False

    def freeze_layout(self):
        if not isinstance(self.layout, MultiOutputLayout):
            self.layout = self.layout.as_fixed()

    def freeze_layout_with_stride_order(self, order):
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_stride_order(order)

    def freeze_layout_with_fill_order(self, order):
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_fill_order(order)

    def make_loader(self):
        def loader(index):
            indexer = self.layout.make_indexer()
            upcast = (
                self.get_dtype() == torch.float16 or self.get_dtype() == torch.bfloat16
            )
            return ops.load(self.name, indexer(index), upcast)

        return loader

    def is_no_op(self):
        return False

    def codegen_reference(self):
        return self.get_name()

    def decide_layout(self):
        pass

    def get_alias_names(self):
        if isinstance(self.layout, AliasedLayout):
            return [self.layout.view.get_name()]
        return ()

    def get_mutation_names(self):
        if isinstance(self.layout, MutationLayout):
            return [self.layout.target.get_name()]
        return ()

    def get_read_writes(self):
        with patch.object(FlexibleLayout, "allow_indexing", True):
            return extract_read_writes(
                self.make_loader(),
                self.get_size(),
            )

    def get_reads(self):
        return self.get_read_writes().reads

    def realize(self):
        pass


class InputBuffer(Buffer):
    pass


class ConstantBuffer(InputBuffer):
    override_device = None

    def make_loader(self):
        def loader(index):
            indexer = self.layout.make_indexer()
            return ops.load(
                V.graph.constant_name(self.name, self.override_device), indexer(index)
            )

        return loader

    def constant_to_device(self, device):
        return ConstantBuffer(V.graph.constant_name(self.name, device), self.layout)


@dataclasses.dataclass
class ComputedBuffer(Buffer):
    data: Loops

    def get_read_writes(self):
        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.data.get_reduction_type():
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_size(),
                    self.data.get_reduction_size(),
                )
            else:
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_size(),
                )

    def get_store_function(self):
        indexer = self.layout.as_fixed().make_indexer()
        if self.data.get_reduction_type():
            return partial(self.data.store_reduction, self.name, indexer)
        else:
            return partial(self.data.store_output, self.name, indexer)

    def decide_layout(self):
        """
        If our layout is still flexible, try to set it based on stride orders of reads.

        TODO(jansel): A better algorithm here would look at downstream consumers of this
                      value and try to do global graph-level layout optimization.
                      This is also something just begging to be autotuned.
        """
        if isinstance(self.layout, FlexibleLayout):
            _, (index_vars, reduction_vars), _ = dependencies.index_vars_squeeze(
                self.data.get_size(), self.data.get_reduction_size()
            )
            reads = self.get_read_writes().reads
            # only consider reads to buffer of same size
            reads = [
                r.index.subs({v: sympy.Integer(0) for v in reduction_vars})
                for r in reads
            ]

            if reads:
                stride_lengths = numpy.array(
                    [V.graph.sizevars.stride_hints(expr, index_vars) for expr in reads],
                    dtype=numpy.int64,
                )
                from .scheduler import pick_loop_order

                self.freeze_layout_with_fill_order(
                    pick_loop_order(stride_lengths, self.get_size())
                )

        if isinstance(self.layout, FlexibleLayout):
            self.freeze_layout()

    def simplify_reorder_and_tile(self):
        """
        This is the main place where we do loop transformations in a
        backend-agnostic way.

        Here we:
            1) Remove any 1 dimensions
            2) Fuse contiguous dimensions together
            3) Reorder dimensions based on stride orders
            4) Split dimensions into tiles
        """
        _, args, var_ranges = dependencies.index_vars_squeeze(
            self.data.get_size(), self.data.get_reduction_size(), prefix="q"
        )
        with patch.object(ConstantBuffer, "override_device", self.get_device()):
            body = LoopBody(
                self.get_store_function(),
                (args if self.get_reduction_type() else args[:1]),
                var_ranges,
            )
        index_formulas = [*body.indexing_exprs.values()]
        memory_addrs = [*body.reads, *body.writes]

        index_vars = []
        reduce_vars = []
        index_size = []
        reduce_size = []
        for v, s in var_ranges.items():
            if v in args[0]:
                assert not reduce_vars
                index_vars.append(v)
                index_size.append(s)
            else:
                assert v in args[1]
                reduce_vars.append(v)
                reduce_size.append(s)

        def simplify_and_reorder(x_vars, sizes):
            sizes, reindex1, prune = _simplify_loops(x_vars, sizes, index_formulas)
            x_vars = prune(x_vars)
            sizes, reindex2 = self._apply_loop_reordering(x_vars, sizes, memory_addrs)
            reindex = fuse_reindexing(reindex1, reindex2)
            return sizes, reindex

        iter_ranges, iter_reindex = simplify_and_reorder(index_vars, index_size)
        reduce_ranges, reduce_reindex = simplify_and_reorder(reduce_vars, reduce_size)

        # retrace the loop body with simplification and reordering applied
        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            iter_ranges, reduce_ranges, prefix="z"
        )
        body = LoopBody(
            body, [iter_reindex(iter_vars), reduce_reindex(reduce_vars)], var_ranges
        )

        # TODO(jansel): support tiling with modular indexing
        has_modular_indexing = any(
            ("ModularIndexing" in str(expr) or "IndexingDiv" in str(expr))
            for expr in body.indexing_exprs.values()
        )

        if (
            is_triton(self.get_device())
            and not self.get_reduction_type()
            and iter_ranges
            and not has_modular_indexing
            and config.triton.max_tiles > 1
        ):
            # TODO(jansel): should we include store strides here or just loads?
            strides = [
                V.graph.sizevars.stride_hints(expr, iter_vars)
                for expr in body.reads
                # TODO(jansel): how should we tile indirect loads?
                if "indirect" not in str(expr)
            ]
            tiled_ranges = self._tile_contiguous(iter_ranges, strides)
            if len(tiled_ranges) > 1:
                return (*tiled_ranges, reduce_ranges), body

            if config.triton.tile_broadcasting:
                # alternate tiling heuristic
                tiled_ranges, call = self._tile_broadcasting(iter_ranges, body, strides)
                if len(tiled_ranges) > 1:
                    return (*tiled_ranges, reduce_ranges), call

        return (iter_ranges, reduce_ranges), body

    @classmethod
    def _tile_contiguous(cls, iter_ranges: List[sympy.Expr], strides):
        """
        Break iter_ranges up into at most max_tiles tiles based on stride==1
        dimensions.

        Transformation on iter_range like:
            (s0, s1, s2, s3) => (s0, s1), (s2, s3)

        Where each group will be tiled in a different dimension in the
        output kernel.
        """
        tiles = []
        current_tile = []
        max_tiles = config.triton.max_tiles

        # TODO(jansel): this is a placeholder heuristic, we should be able to do much better
        for i in range(len(iter_ranges)):
            current_tile.append(iter_ranges[i])
            # break tiles on stride 1
            if any(stride[i] == 1 for stride in strides):
                tiles.append(current_tile)
                current_tile = []

        if current_tile:
            tiles.append(current_tile)

        if len(tiles) > max_tiles:
            split = len(tiles) - max_tiles + 1
            tiles = [[*itertools.chain(*tiles[:split])]] + tiles[split:]
            assert len(tiles) == max_tiles, (len(tiles), max_tiles, split)

        return tiles

    @classmethod
    def _tile_broadcasting(cls, iter_ranges: List[sympy.Expr], body, strides):
        """
        Break ranges up so that one dimension is all broadcasting.

        Transformation on iter_ranges like:
            (s0, s1, s2, s3) => (s0, s1), (s2, s3)

        Where each group will be tiled in a different dimension in the
        output kernel.
        """
        broadcasting_strides = [
            stride for stride in strides if any(s == 0 for s in stride)
        ]
        if not broadcasting_strides:
            return (iter_ranges,), body

        # TODO(jansel): consider another load?  for now just take first one
        broadcasting_stride = broadcasting_strides[0]

        broadcast_ranges = []
        broadcast_index = []
        other_ranges = []
        other_index = []

        # TODO(jansel): this is a placeholder heuristic, we should be able to do much better
        for i in range(len(iter_ranges)):
            if broadcasting_stride[i] == 0:
                broadcast_index.append(i)
                broadcast_ranges.append(iter_ranges[i])
            else:
                other_index.append(i)
                other_ranges.append(iter_ranges[i])

        def call(broadcast, other, reduction=None):
            assert not reduction
            assert len(broadcast) == len(broadcast_index)
            assert len(other) == len(other_index)
            args = [None] * len(iter_ranges)
            for i, v in itertools.chain(
                zip(broadcast_index, broadcast),
                zip(other_index, other),
            ):
                args[i] = v
            return body(args)

        if broadcast_ranges and other_ranges:
            return (broadcast_ranges, other_ranges), call
        else:
            return (iter_ranges,), body

    @staticmethod
    def _apply_loop_reordering(index_vars, sizes, memory_addrs):
        """
        Shuffle the order of loops around to hopefully improve performance.
        """
        from .scheduler import pick_loop_order

        try:
            strides = numpy.array(
                [
                    V.graph.sizevars.stride_hints(expr, index_vars)
                    for expr in memory_addrs
                ],
                dtype=numpy.int64,
            )
            assert strides.shape == (len(memory_addrs), len(index_vars))
            order = list(reversed(pick_loop_order(strides, sizes)))
        except Exception:
            if config.debug:
                log.warning(
                    f"Did not simplify complex index:\n{dict(zip(index_vars, sizes))}\n{memory_addrs}"
                )
            order = list(range(len(sizes)))
        sizes = [sizes[i] for i in order]
        return sizes, inverse_reorder(order)

    def get_reduction_size(self):
        return self.data.get_reduction_size()

    def get_reduction_type(self):
        return self.data.get_reduction_type()

    def is_no_op(self):
        return self.data.is_zero_elements()

    def should_allocate(self):
        return True

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        return self.data.constant_to_device(device)


@dataclasses.dataclass
class InputsKernel(Buffer):
    inputs: List[Buffer]

    def get_read_writes(self):
        return dependencies.ReadWrites(
            {dependencies.StarDep(x.get_name()) for x in self.inputs},
            {dependencies.StarDep(self.get_name())},
            set(),
        )

    @staticmethod
    def unwrap_storage(inputs):
        inputs_new = []
        for x in inputs:
            if isinstance(x, TensorBox):
                x = x.data
            if isinstance(x, StorageBox):
                x = x.data
            assert isinstance(x, (Buffer, ReinterpretView)), x
            inputs_new.append(x)
        return inputs_new

    def is_extern(self):
        return True


class NopKernel(InputsKernel):
    def is_no_op(self):
        return True


class ConcatKernel(NopKernel):
    """
    There isn't actually a real kernel for concat, we just change the
    storage for the upstream data.
    """

    @classmethod
    def create(cls, inputs, dim):
        device = inputs[0].get_device()
        dtype = inputs[0].get_dtype()
        new_size = list(inputs[0].get_size())
        offsets_start = [0]
        offsets_end = [new_size[dim]]
        assert 0 <= dim < len(new_size)
        for i in range(1, len(inputs)):
            input_size = inputs[i].get_size()
            offsets_start.append(new_size[dim])
            assert len(input_size) == len(new_size)
            assert inputs[i].get_dtype() == dtype
            assert inputs[i].get_device() == device
            for j in range(len(new_size)):
                if j == dim:
                    new_size[j] = new_size[j] + input_size[j]
                else:
                    new_size[j] = V.graph.sizevars.guard_equals(
                        new_size[j], input_size[j]
                    )
            offsets_end.append(new_size[dim])

        kernel = ConcatKernel(
            name=None,
            layout=FixedLayout(
                device=device,
                dtype=dtype,
                size=new_size,
                stride=FlexibleLayout.contiguous_strides(new_size),
            ),
            inputs=[],
        )
        kernel = StorageBox(kernel)
        for i in range(len(inputs)):
            kernel.data.inputs.append(
                cls.realize_into(
                    inputs[i],
                    SliceView.create(kernel, dim, offsets_start[i], offsets_end[i]),
                )
            )
        kernel.data.name = V.graph.register_buffer(kernel.data)
        kernel.data.inputs = cls.unwrap_storage(kernel.data.inputs)
        return kernel

    @classmethod
    def realize_into(cls, src, dst):
        assert isinstance(dst, ReinterpretView), dst
        if isinstance(src, TensorBox):
            # unwrap a TensorBox
            return cls.realize_into(src.data, dst)
        if isinstance(src, StorageBox):
            src.realize()
            if isinstance(src.data.layout, FlexibleLayout):
                src.data.layout = AliasedLayout(dst)
                return src.data
        # introduce a copy
        pw = Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=[
                V.graph.sizevars.guard_equals(a, b)
                for a, b in zip(src.get_size(), dst.get_size())
            ],
        )
        return cls.realize_into(pw, dst)

    def should_allocate(self):
        return True


@dataclasses.dataclass
class ExternKernel(InputsKernel):
    constant_args: List[Any] = ()
    output_view: Optional[ReinterpretView] = None

    def decide_layout(self):
        self.freeze_layout()

    @staticmethod
    def copy_input(x):
        pw = Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=x.make_loader(),
            ranges=x.get_size(),
        )
        pw.realize()
        return pw

    @classmethod
    def realize_input(cls, x):
        if x is None:
            return V.graph.add_tensor_constant(torch.tensor(()))
        if isinstance(x, Constant):
            return V.graph.add_tensor_constant(
                torch.tensor(x.value, dtype=x.get_dtype(), device=x.get_device())
            )
        if isinstance(x, TensorBox):
            return cls.realize_input(x.data)
        if isinstance(x, ReinterpretView):
            return x
        if isinstance(x, StorageBox):
            # TODO(jansel): impose layout preference on realized buffer
            x.realize()
            return x
        return cls.copy_input(x)

    @classmethod
    def require_stride1(cls, x):
        if len(x.get_stride()) == 0:
            return x
        for stride in x.get_stride():
            if stride == 1:
                return x
        return cls.copy_input(x)

    @classmethod
    def require_contiguous(cls, x):
        if is_contiguous_storage_and_layout(x):
            as_contiguous_storage_and_layout(x, freeze=True)
            return x
        x = cls.copy_input(x)
        assert is_contiguous_storage_and_layout(x)
        as_contiguous_storage_and_layout(x, freeze=True)
        return x

    def codegen_args(self):
        args = [x.codegen_reference() for x in self.inputs]
        args.extend(map(repr, self.constant_args))
        return args

    def codegen_size_asserts(self, wrapper):
        if config.size_asserts:
            size = V.graph.sizevars.codegen_shape_tuple(self.get_size())
            stride = V.graph.sizevars.codegen_shape_tuple(self.get_stride())
            wrapper.writeline(f"assert {self.get_name()}.size() == {size}")
            wrapper.writeline(f"assert {self.get_name()}.stride() == {stride}")

    def get_group_stride(self):
        """
        get output sizes and strides, for template_codegen
        """
        _size = self.get_size()
        _stride = self.get_stride()
        # iter_ranges = _size of output tensor, reduce_range = [] because no reduction
        return [_size, []], _stride


@dataclasses.dataclass
class ExternKernelOut(ExternKernel):
    output_view: Optional[ReinterpretView] = None

    def codegen(self, wrapper):
        args = self.codegen_args()
        if self.output_view:
            args.append(f"out={self.output_view.codegen_reference()}")
        else:
            args.append(f"out={self.codegen_reference()}")
        wrapper.writeline(f"{self.kernel}({', '.join(args)})")

    def __init__(self, layout, inputs, constant_args=(), output_view=None):
        super().__init__(None, layout, self.unwrap_storage(inputs), constant_args)
        self.output_view = output_view
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        return True


class ExternKernelAlloc(ExternKernel):
    def codegen(self, wrapper):
        wrapper.writeline(
            f"{self.get_name()} = {self.kernel}({', '.join(self.codegen_args())})"
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def __init__(self, layout, inputs, constant_args=()):
        super().__init__(None, layout, self.unwrap_storage(inputs), constant_args)
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        return False


class IndexPutFallback(ExternKernel):
    # TODO(jansel): delete this when this is fixed:
    # https://github.com/openai/triton/issues/558
    kernel = "aten.index_put_"

    def codegen(self, wrapper):
        x, *indices, values = [t.codegen_reference() for t in self.inputs]
        (accumulate,) = self.constant_args
        wrapper.writeline(
            f"{self.kernel}({x}, [{', '.join(indices)}], {values}, {accumulate!r})"
        )

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        assert isinstance(self.layout, MutationLayout)
        return (self.layout.target.get_name(),)

    def __init__(self, x, indices, values, accumulate=False):
        super().__init__(
            None,
            MutationLayout(x),
            self.unwrap_storage([x, *indices, values]),
            [accumulate],
        )
        self.name = V.graph.register_buffer(self)


class InplaceBernoulliFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    kernel = "aten.bernoulli_"

    def codegen(self, wrapper):
        (x,) = [t.codegen_reference() for t in self.inputs]
        wrapper.writeline(
            f"{self.kernel}({x}, {', '.join(map(repr, self.constant_args))})"
        )

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        assert isinstance(self.layout, MutationLayout)
        return (self.layout.target.get_name(),)

    def __init__(self, x, *constant_args):
        super().__init__(
            None,
            MutationLayout(x),
            self.unwrap_storage([x]),
            constant_args,
        )
        self.name = V.graph.register_buffer(self)


class MatrixMultiply(ExternKernelOut):
    kernel = "aten.mm.out"

    def __init__(self, layout, inputs, constant_args=(), output_view=None):
        super().__init__(layout, inputs, constant_args, output_view)
        if (
            config.triton.use_mm
            and len(inputs) > 0
            and inputs[0].get_device().type == "cuda"
        ):
            self.kernel = "triton_mm_out"

    @classmethod
    def create(cls, a, b):
        *m, k1 = a.get_size()
        k2, n = b.get_size()
        V.graph.sizevars.guard_equals(k1, k2)
        a = cls.realize_input(a)
        b = cls.realize_input(b)
        if len(m) != 1 and not a.get_layout().is_contiguous():
            a = cls.copy_input(a)
        else:
            a = cls.require_stride1(a)
        b = cls.require_stride1(b)
        return MatrixMultiply(
            layout=FlexibleLayout(
                device=a.get_device(),
                dtype=a.get_dtype(),
                size=list(m) + [n],
            ),
            inputs=[a, b],
        )


class BatchMatrixMultiply(ExternKernelOut):
    kernel = "aten.bmm.out"

    def __init__(self, layout, inputs, constant_args=(), output_view=None):
        super().__init__(layout, inputs, constant_args, output_view)
        if (
            config.triton.use_bmm
            and len(inputs) > 0
            and inputs[0].get_device().type == "cuda"
        ):
            self.kernel = "triton_bmm_out"

    @classmethod
    def create(cls, a, b):
        b1, m, k1 = a.get_size()
        b2, k2, n = b.get_size()
        b3 = V.graph.sizevars.guard_equals(b1, b2)
        V.graph.sizevars.guard_equals(k1, k2)
        a = cls.require_stride1(cls.realize_input(a))
        b = cls.require_stride1(cls.realize_input(b))

        output_layout = FlexibleLayout(
            device=a.get_device(),
            dtype=a.get_dtype(),
            size=[b3, m, n],
        ).as_fixed()

        if b3 == 1:
            # convert to normal mm
            data = MatrixMultiply(
                layout=output_layout.as_fixed(),
                inputs=[View.create(a, [m, k1]), View.create(b, [k2, n])],
            )
            data.output_view = ReinterpretView(
                data,
                FlexibleLayout(
                    device=a.get_device(),
                    dtype=a.get_dtype(),
                    size=[m, n],
                ).as_fixed(),
            )
        else:
            data = BatchMatrixMultiply(
                layout=output_layout,
                inputs=[a, b],
            )
        return data


class DeviceCopy(ExternKernelOut):
    @classmethod
    def create(cls, x, device):
        V.graph.device_types.add(device.type)
        V.graph.device_types.add(x.get_device().type)

        x = cls.realize_input(x)
        read_writes = x.get_read_writes()
        if not x.is_extern() and all(
            (r.name in V.graph.constants and hasattr(r, "index"))
            for r in read_writes.reads
        ):
            return x.constant_to_device(device)

        return DeviceCopy(
            FlexibleLayout(
                device=device,
                dtype=x.get_dtype(),
                size=x.get_size(),
            ),
            [x],
        )

    def codegen(self, wrapper):
        args = self.codegen_args()
        assert len(args) == 1
        if self.output_view:
            wrapper.writeline(
                f"{self.output_view.codegen_reference()}.copy_({args[0]})"
            )
        else:
            wrapper.writeline(f"{self.codegen_reference()}.copy_({args[0]})")


class DynamicScalar(IRNode):
    """
    The result of a call to aten._local_scalar_dense.

    This is not yet implemented.  The one model (so far) that calls this
    (fastNLP_Bert) does not actually use the result.  So we expect this
    node to get dead code eliminated.
    """

    def get_reads(self):
        return ()


class AdaptiveAvgPool2d(ExternKernelAlloc):
    kernel = "aten._adaptive_avg_pool2d"

    @classmethod
    def create(cls, x, target_size):
        x = cls.require_stride1(cls.realize_input(x))
        output_size = [
            *x.get_size()[: -len(target_size)],
            *map(sympy.Integer, target_size),
        ]
        return cls(
            FixedLayout(
                x.get_device(),
                x.get_dtype(),
                output_size,
                # TODO(jansel): fix channels last case
                FlexibleLayout.contiguous_strides(output_size),
            ),
            (x,),
            (tuple(target_size),),
        )


@dataclasses.dataclass
class FallbackKernel(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
    ):
        super(FallbackKernel, self).__init__(
            layout,
            tuple(tensor_args),
            tuple(nontensor_args),
        )
        if getattr(torch.ops.aten, kernel.__name__, None) is kernel:
            self.kernel = f"aten.{kernel.__name__}"
        else:
            self.kernel = (
                f"{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}"
            )
        self.unflatten_args = unflatten_args
        log.warning(f"Using FallbackKernel: {self.kernel}")

    def codegen_args(self):
        @dataclasses.dataclass
        class Shim:
            ref: Any

            def __repr__(self):
                return self.ref

        tensor_args = [Shim(x.codegen_reference()) for x in self.inputs]
        constant_args = [Shim(repr(x)) for x in self.constant_args]
        return list(map(repr, self.unflatten_args(tensor_args, constant_args)))

    @classmethod
    def create(cls, kernel, *args):
        args_flat, args_spec = pytree.tree_flatten(args)

        is_arg_tensor = []
        tensor_args = []
        non_tensor_args = []
        for arg in args_flat:
            is_arg_tensor.append(isinstance(arg, IRNode))
            if is_arg_tensor[-1]:
                tensor_args.append(arg)
            else:
                non_tensor_args.append(arg)

        def unflatten_args(new_tensor_args, new_non_tensor_args):
            new_args = []
            it_tensors = iter(new_tensor_args)
            it_non_tensors = iter(new_non_tensor_args)
            for is_tensor in is_arg_tensor:
                if is_tensor:
                    new_args.append(next(it_tensors))
                else:
                    new_args.append(next(it_non_tensors))
            return pytree.tree_unflatten(new_args, args_spec)

        tensor_args = [
            cls.require_contiguous(cls.realize_input(x)) for x in tensor_args
        ]

        # We don't have generic shape formulas, so just burn in the
        # shapes and run an example input.
        # TODO(jansel): replace this with dynamic shape formulas
        example_args = [
            torch.zeros(
                [V.graph.sizevars.guard_static_shape(s) for s in x.get_size()],
                dtype=x.get_dtype(),
                device=x.get_device(),
            )
            for x in tensor_args
        ]
        example_output = kernel(*unflatten_args(example_args, non_tensor_args))

        if isinstance(example_output, (list, tuple)):
            packed = FallbackKernel(
                MultiOutputLayout(),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            return [
                (
                    MultiOutput(
                        FixedLayout(
                            example_output[i].device,
                            example_output[i].dtype,
                            [sympy.Integer(s) for s in example_output[i].size()],
                            [sympy.Integer(s) for s in example_output[i].stride()],
                        ),
                        packed,
                        i,
                    )
                    if example_output[i] is not None
                    else None
                )
                for i in range(len(example_output))
            ]
        else:
            return FallbackKernel(
                FixedLayout(
                    example_output.device,
                    example_output.dtype,
                    [sympy.Integer(s) for s in example_output.size()],
                    [sympy.Integer(s) for s in example_output.stride()],
                ),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )


class MultiOutputLayout(IRNode):
    pass


class MultiOutput(ExternKernel):
    def codegen(self, wrapper):
        wrapper.writeline(
            f"{self.get_name()} = {self.inputs[0].get_name()}[{self.index}]"
        )
        self.codegen_size_asserts(wrapper)

    def __init__(self, layout, input, index):
        super().__init__(None, layout, [input], ())
        self.name = V.graph.register_buffer(self)
        self.index = index

    def should_allocate(self):
        return False


class Convolution(ExternKernelAlloc):
    config_conv = config.triton.convolution
    if config_conv == "aten":
        kernel = "aten.convolution"
    elif config_conv == "triton":
        kernel = "triton_ops_conv"
    else:
        assert config_conv == "autotune"
        kernel = "tuned_conv"

    def codegen(self, wrapper):
        if self.kernel == "triton_ops_conv":
            wrapper.header.writeline(
                f"import torchinductor.triton_ops.conv as {self.kernel}"
            )
        # choose from different conv kernels
        elif self.kernel == "tuned_conv":
            wrapper.header.writeline(
                f"from torchinductor.codegen.autotuner import {self.kernel}"
            )
        wrapper.writeline(
            f"{self.get_name()} = {self.kernel}({', '.join(self.codegen_args())})"
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        transposed: bool,
        output_padding: List[int],
        groups: int,
    ):
        x = cls.require_stride1(cls.realize_input(x))
        weight = cls.require_stride1(cls.realize_input(weight))
        stride = tuple(stride)
        padding = tuple(padding)
        dilation = tuple(dilation)
        assert isinstance(transposed, bool)
        output_padding = tuple(output_padding)
        assert isinstance(groups, int)

        weight_shape = [
            sympy.Integer(V.graph.sizevars.guard_static_shape(s))
            for s in weight.get_size()
        ]

        out_channels, in_channels1, *kernel_size = weight_shape
        in_channels1 = in_channels1 * groups
        if transposed:
            out_channels, in_channels1 = in_channels1, out_channels

        if bias is not None:
            bias = cls.require_stride1(cls.realize_input(bias))
            (bias_shape,) = [
                sympy.Integer(V.graph.sizevars.guard_static_shape(s))
                for s in bias.get_size()
            ]
            assert bias_shape == out_channels, f"{bias_shape} == {out_channels}"

        if len(x.get_size()) == 1 + len(kernel_size):
            in_channels2, *input_size = x.get_size()
            in_channels_stride, *_ = x.get_stride()
            output_size = []
        else:
            assert len(x.get_size()) == 2 + len(kernel_size)
            batch, in_channels2, *input_size = x.get_size()
            _, in_channels_stride, *_ = x.get_stride()
            output_size = [batch]

        V.graph.sizevars.guard_equals(in_channels1, in_channels2)

        output_size.append(out_channels)

        assert (
            len(stride)
            == len(padding)
            == len(dilation)
            == len(output_padding)
            == len(kernel_size)
            == len(input_size)
        )
        for i in range(len(stride)):
            if transposed:
                output_size.append(
                    (input_size[i] - 1) * stride[i]
                    - 2 * padding[i]
                    + dilation[i] * (kernel_size[i] - 1)
                    + output_padding[i]
                    + 1
                )
            else:
                output_size.append(
                    IndexingDiv(
                        input_size[i]
                        + 2 * padding[i]
                        - dilation[i] * (kernel_size[i] - 1)
                        - 1
                        + stride[i],
                        stride[i],
                    )
                    + 2 * output_padding[i]
                )
            output_size[-1] = sympy.Integer(
                V.graph.sizevars.guard_static_shape(output_size[-1])
            )

        if any(k != 1 for k in output_size[-len(stride) :]) and in_channels_stride == 1:
            # channels last format
            order = [0] + list(reversed(range(1, len(kernel_size) + 1)))
            if len(order) < len(output_size):
                # add batch dim if it exists
                order = [len(order)] + order
        else:
            order = list(reversed(range(len(output_size))))

        output_layout = FixedLayout(
            x.get_device(),
            x.get_dtype(),
            output_size,
            FlexibleLayout.stride_ordered(output_size, order),
        )

        if bias is not None:
            return Convolution(
                output_layout,
                (x, weight, bias),
                (stride, padding, dilation, transposed, output_padding, groups),
            )
        else:
            return Convolution(
                output_layout,
                (x, weight),
                (bias, stride, padding, dilation, transposed, output_padding, groups),
            )

    def map_args(self):
        # x, w, bias
        in_args = [x.codegen_reference() for x in self.inputs]
        # stride, padding, dilation, transposed, output_padding, groups
        const_args = self.constant_args
        if len(in_args) < 3:
            # otherwise, bias=None is the first constant_args
            const_args = const_args[1:]
        # stride of inputs and outputs
        stride_x = f"{in_args[0]}.stride()"
        stride_w = f"{in_args[1]}.stride()"
        stride_y = f"{self.get_name()}.stride()"

        args_dict = OrderedDict(
            [
                ("x", f"{in_args[0]}"),
                ("w", f"{in_args[1]}"),
                ("bias", f"{in_args[2]}" if len(in_args) >= 3 else "None"),
                ("y", f"{self.get_name()}"),
                ("stride_xn", stride_x + "[0]"),
                ("stride_xc", stride_x + "[1]"),
                ("stride_xh", stride_x + "[2]"),
                ("stride_xw", stride_x + "[3]"),
                ("stride_wn", stride_w + "[0]"),
                ("stride_wc", stride_w + "[1]"),
                ("stride_wh", stride_w + "[2]"),
                ("stride_ww", stride_w + "[3]"),
                ("stride_yn", stride_y + "[0]"),
                ("stride_yc", stride_y + "[1]"),
                ("stride_yh", stride_y + "[2]"),
                ("stride_yw", stride_y + "[3]"),
                (
                    "stride_biasn",
                    f"{in_args[2]}.stride()[0]" if len(in_args) >= 3 else "None",
                ),
                ("delta_x_ptr", "None"),
                ("BATCH", f"{in_args[0]}.shape[0]"),
                ("IN_C", f"{in_args[0]}.shape[1]"),
                ("IN_H", f"{in_args[0]}.shape[2]"),
                ("IN_W", f"{in_args[0]}.shape[3]"),
                ("KERNEL_N", f"{in_args[1]}.shape[0]"),
                ("KERNEL_H", f"{in_args[1]}.shape[2]"),
                ("KERNEL_W", f"{in_args[1]}.shape[3]"),
                ("OUT_H", f"{self.get_name()}.shape[2]"),
                ("OUT_W", f"{self.get_name()}.shape[3]"),
                ("stride_h", f"{const_args[0][0]}"),
                ("stride_w", f"{const_args[0][1]}"),
                ("padding_h", f"{const_args[1][0]}"),
                ("padding_w", f"{const_args[1][1]}"),
                ("dilation_h", f"{const_args[2][0]}"),
                ("dilation_w", f"{const_args[2][1]}"),
                # ("transposed", f"{const_args[3]}"),
                ("output_padding_h", f"{const_args[4][0]}"),
                ("output_padding_w", f"{const_args[4][1]}"),
                ("groups", f"{const_args[5]}"),
            ]
        )

        # accumulator type
        ACC_TYPE = (
            "tl.float32"
            if self.inputs[0].get_dtype()
            in [torch.float16, torch.bfloat16, torch.float32]
            else "tl.int32"
        )
        CONV1X1_NHWC = (
            "True"
            if self.inputs[0].get_stride()[1] == 1
            and self.inputs[1].shape[2] == 1
            and self.inputs[1].shape[3] == 1
            else "False"
        )
        # dict for tl.constexpr
        const_dict = OrderedDict(
            [
                ("ACC_TYPE", ACC_TYPE),
                ("CONV1X1_NHWC", CONV1X1_NHWC),
            ]
        )

        # dict for non-kernel args (e.g. delta_x_ptr)
        other_dict = OrderedDict(
            [
                ("device", f'"{self.inputs[0].get_device()}"'),
            ]
        )

        return args_dict, const_dict, other_dict


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

    def __str__(self):
        if isinstance(self.data, MutableBox):
            line0 = f"{type(self).__name__}({type(self.data).__name__}("
            endl = "))"
            inner = self.data.data
        else:
            line0 = f"{type(self).__name__}("
            inner = self.data
            endl = ")"

        lines = [
            line0,
            indent(str(inner)),
            endl,
        ]
        return "\n".join(lines)

    __repr__ = __str__


class TensorBox(MutableBox):
    @staticmethod
    def create(data):
        return TensorBox(StorageBox(data))


class StorageBox(MutableBox):
    def realize(self):
        if isinstance(
            self.data, (ComputedBuffer, InputsKernel, InputBuffer, ReinterpretView)
        ):
            return self.data.get_name()
        assert isinstance(self.data, (Pointwise, Reduction)), type(self.data)
        self.data = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=self.data.get_device(),
                dtype=self.data.get_dtype(),
                size=self.data.get_size(),
            ),
            data=self.data,
        )
        self.data.name = V.graph.register_buffer(self.data)
        return self.data.name

    def mark_reuse(self, users):
        if users <= 1:
            return
        if isinstance(self.data, (Pointwise, Reduction)):
            read_writes = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=self.data.get_device(),
                    dtype=self.data.get_dtype(),
                    size=self.data.get_size(),
                ),
                data=self.data,
            ).get_read_writes()

            # TODO(jansel): this heuristic is a wild guess
            if (
                len(read_writes.reads) > config.realize_reads_threshold
                or len(self.inner_fn_str()) > config.realize_bytes_threshold
            ):
                self.realize()


class LoopBody:
    """
    Captures the body of a Loops subclass into an FX graph.  Persists any
    indexing simplifications and makes it easier to analyze loop bodies.
    """

    def __init__(self, fn, args, var_ranges):
        super().__init__()
        self.var_ranges = var_ranges
        self.indexing_exprs = {}
        self.indexing_exprs_name = {}
        self.reads = []
        self.writes = []
        self.other = []
        self.submodules = {}
        self.subblocks = {}
        self.indirect_vars = []
        self.root_block = LoopBodyBlock(self, fn, args)
        self.indexing = None

    def add_index_expr(self, expr: sympy.Expr, category):
        getattr(self, category).append(expr)
        if expr not in self.indexing_exprs_name:
            name = f"index{len(self.indexing_exprs)}"
            self.indexing_exprs_name[expr] = name
            self.indexing_exprs[name] = expr
        return self.indexing_exprs_name[expr]

    def add_submodule(self, block, prefix):
        """Not actually for nn.Modules, but subblocks in generated code are mapped to FX call_module opcodes"""
        if prefix[-1].isnumeric() and prefix not in self.submodules:
            name = prefix
        else:
            name = f"{prefix}{len(self.submodules)}"
        self.submodules[name] = block
        return name

    def add_indirect(self):
        name = f"indirect{len(self.indirect_vars)}"
        var = sympy.Symbol(name, integer=True)
        self.indirect_vars.append([var])
        return var

    def __call__(self, *indices):
        index = list(itertools.chain(*indices))
        assert len(index) == len(self.var_ranges)
        assert all(v not in self.var_ranges for v in index)
        replacements = dict(zip(self.var_ranges.keys(), index))
        self.indexing = {
            name: expr.subs(replacements) for name, expr in self.indexing_exprs.items()
        }
        result = self.root_block()
        self.indexing = None
        return result


class LoopBodyBlock:
    """
    Captures the body of a Loops subclass into an FX graph.
    In normal cases there will be a 1:1 mapping between LoopBody and
    LoopBodyBlock, hower in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """

    def __init__(self, body: LoopBody, fn: Callable, args: List[Any]):
        self.gm = None
        self.body = body

        def add_index(expr, category):
            return tracer.create_proxy(
                "get_attr", self.body.add_index_expr(expr, category), (), {}
            )

        class CaptureIndexing(V.WrapperHandler):
            def load(self, name: str, index: sympy.Expr, upcast: bool = False):
                index = add_index(index, "reads")
                return self._inner.load(name, index, upcast)

            def store(self, name, index, value, mode=None):
                index = add_index(index, "writes")
                return self._inner.store(name, index, value, mode)

            def reduction(self, name, dtype, reduction_type, index, value):
                index = add_index(index, "writes")
                return self._inner.reduction(name, dtype, reduction_type, index, value)

            def index_expr(self, index, dtype):
                index = add_index(index, "other")
                return self._inner.index_expr(index, dtype)

            @staticmethod
            def masked(mask_proxy, masked_body: Callable, other_proxy):
                """
                Recursively capture the masked out body in another LoopBodyBlock
                """

                def shim(mask, other):
                    return V.ops.masked(mask, subblock, other)

                name = self.body.add_submodule(shim, "masked_subblock")
                subblock = LoopBodyBlock(self.body, masked_body, ())
                self.body.subblocks[name] = subblock
                return tracer.create_proxy(
                    "call_module", name, (mask_proxy, other_proxy), {}
                )

            @staticmethod
            def indirect_indexing(index_proxy):
                """
                Flow data from tensors into indexing formulas.
                Introduce a call_module to update the indexing.
                """

                def set_indirect(new_var):
                    self.replace_indirect(var, V.ops.indirect_indexing(new_var))

                var = self.body.add_indirect()
                tracer.create_proxy(
                    "call_module",
                    self.body.add_submodule(set_indirect, f"set_{var}"),
                    (index_proxy,),
                    {},
                )
                return var

        tracer = torch.fx.Tracer()
        tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
        proxy_ops = tracer.create_proxy("placeholder", "ops", (), {})
        from .sizevars import SimplifyIndexing

        with V.set_ops_handler(
            SimplifyIndexing(CaptureIndexing(proxy_ops), self.body.var_ranges)
        ):
            tracer.create_proxy("output", "output", (fn(*args),), {})
        self.graph = tracer.graph

    def replace_indirect(self, old, new):
        """Swap in a variable used in indirect indexing"""
        for name in self.body.indexing.keys():
            expr = getattr(self.gm, name)
            if old in expr.free_symbols:
                setattr(self.gm, name, expr.subs({old: new}))

    def __call__(self):
        self.gm = torch.fx.GraphModule(
            {**self.body.indexing, **self.body.submodules}, self.graph
        )
        result = self.gm.forward(V.get_ops_handler())
        self.gm = None
        return result
