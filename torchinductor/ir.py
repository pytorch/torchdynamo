import dataclasses
import functools
import textwrap
from functools import partial
from typing import Any
from typing import Callable
from typing import List

import sympy
import torch
from sympy import Expr
from sympy import Integer

from . import dependencies
from .codegen.common import product
from .dependencies import extract_read_writes
from .virtualized import MockHandler
from .virtualized import graph
from .virtualized import ops

indent = functools.partial(textwrap.indent, prefix="  ")


class ModularIndexing(sympy.Function):
    """
    ModularIndexing(a, b, c) => (a // b) % c
    """

    nargs = (3,)

    @classmethod
    def eval(cls, base, divisor, modulus):
        if base.is_integer and divisor.is_integer and modulus.is_integer:
            return (base // divisor) % modulus
        if base == 0:
            return sympy.Integer(0)


class CleanDiv(sympy.Function):
    """
    a // b where we know there is no rounding
    """

    nargs = (2,)

    @classmethod
    def eval(cls, base, divisor):
        if base == 0:
            return sympy.Integer(0)
        if base.is_integer and divisor.is_integer:
            return base // divisor
        if sympy.gcd(base, divisor) == divisor:
            return base / divisor


class IRNode(object):
    def str_helper(self, lines):
        lines = indent("\n".join(map(str, lines)))
        return f"{type(self).__name__}(\n{lines}\n)"


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

    @classmethod
    def create(cls, *args, **kwargs):
        return TensorBox.create(cls(*args, **kwargs))

    def index_length(self):
        return len(self.ranges)

    def inner_fn_str(self):
        with ops.set_handler(MockHandler()):
            index = [sympy.Symbol(f"i{n}") for n in range(self.index_length())]
            return f"lambda {', '.join(map(str, index))}: {self.inner_fn(index)}"


class Pointwise(Loops):
    def make_loader(self):
        return self.inner_fn

    def get_reduction_size(self):
        return []

    def get_reduction_type(self):
        return None

    def store_output(self, output_name, indexer, vars):
        return ops.store(output_name, indexer(vars), self.inner_fn(vars))


@dataclasses.dataclass
class Reduction(Loops):
    reduction_ranges: List[Expr]
    reduction_type: str

    __str__ = functools.partial(
        Loops.__str__, names=("ranges", "reduction_ranges", "reduction_type")
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


def as_storage_and_layout(x, freeze=True):
    """Try to simplify x into a StorageBox and a Layout"""
    if isinstance(x, TensorBox):
        return as_storage_and_layout(x.data)
    if isinstance(x, StorageBox) and isinstance(x.data, Buffer):
        if freeze:
            x.data.freeze_layout()
        return x, x.data.layout
    if isinstance(x, ReinterpretView):
        buffer, _ = as_storage_and_layout(x.data)
        return buffer, x.layout
    raise NotImplementedError


@dataclasses.dataclass
class BaseView(IRNode):
    data: IRNode

    def get_dtype(self):
        return self.data.get_dtype()

    def get_device(self):
        return self.data.get_device()


@dataclasses.dataclass
class ExpandView(BaseView):
    size: List[Expr]

    @classmethod
    def create(cls, x, new_size):
        new_size = list(map(sympy.expand, new_size))

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            skip = len(new_size) - len(old_layout.size)
            assert skip >= 0
            new_stride = [sympy.Integer(0)] * skip
            for stride, size in zip(old_layout.stride, old_layout.size):
                new_stride.append(stride if size != 1 else sympy.Integer(0))
            new_layout = Layout(
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
            new_layout = Layout(
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
            new_layout = Layout(
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
            storage, old_layout = as_storage_and_layout(x)
            new_layout = Layout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                FlexibleLayout.contiguous_strides(new_size),
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        return cls(x, tuple(new_size), cls.dynamic_reshape_indexer(old_size, new_size))

    @staticmethod
    def resolve_negative_size(old_size, new_size):
        new_size = [sympy.expand(x).subs(graph.sizevars.replacements) for x in new_size]
        old_size = [sympy.expand(x).subs(graph.sizevars.replacements) for x in old_size]

        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.Integer(1)
                new_size[i] = CleanDiv(product(old_size), product(new_size))
                break

        graph.sizevars.guard_equals(product(old_size), product(new_size))
        return old_size, new_size

    @staticmethod
    def dynamic_reshape_indexer(old_size, new_size):
        """
        Perform a reshape entirely by modifying indexing math
        """
        size_hint = graph.sizevars.size_hint
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
                graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) < size_hint(size_old):
                while size_hint(size_new) < size_hint(size_old):
                    var2, size_new2 = stack_new.pop()
                    var = var2 * size_new + var
                    size_new = size_new * size_new2
                view_expr.append(var)
                graph.sizevars.guard_equals(size_new, size_old)
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
                graph.sizevars.guard_equals(size_new, size_old)
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
            assert len(index) == len(vars)
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
        indexer = self.layout.make_indexer()

        def loader(index):
            return ops.load(self.get_name(), indexer(index))

        return loader

    def get_layout(self):
        return self.layout

    def freeze_layout(self):
        pass

    def codegen_reference(self):
        size = graph.sizevars.codegen_shape_tuple(self.layout.size)
        stride = graph.sizevars.codegen_shape_tuple(self.layout.stride)
        offset = graph.sizevars.codegen_sizevar(self.layout.offset)
        if offset != "0":
            return f"as_strided({self.get_name()}, {size}, {stride}, {offset})"
        return f"as_strided({self.get_name()}, {size}, {stride})"


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


class FixedLayout(Layout):
    """A Tensor layout we cannot change"""

    pass


class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""

    @staticmethod
    def contiguous_strides(sizes):
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.Integer(1)]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    def __init__(self, device, dtype, size):
        super(FlexibleLayout, self).__init__(
            device, dtype, size, FlexibleLayout.contiguous_strides(size)
        )


@dataclasses.dataclass
class Buffer(IRNode):
    name: str
    layout: Layout

    def get_name(self):
        assert self.name
        return self.name

    def get_device(self):
        return self.layout.device

    def get_dtype(self):
        return self.layout.dtype

    def get_size(self):
        return self.layout.size

    def get_stride(self):
        return self.layout.stride

    def get_layout(self):
        return self.layout

    def freeze_layout(self):
        self.layout = self.layout.as_fixed()

    def make_loader(self):
        indexer = self.layout.make_indexer()

        def loader(index):
            return ops.load(self.name, indexer(index))

        return loader

    def codegen_reference(self):
        return self.get_name()


@dataclasses.dataclass
class InputBuffer(Buffer):
    pass


@dataclasses.dataclass
class ComputedBuffer(Buffer):
    data: Loops

    def get_read_writes(self):
        indexer = self.layout.make_indexer()
        if self.data.get_reduction_type():
            return extract_read_writes(
                partial(self.data.store_reduction, self.name, indexer),
                self.data.get_size(),
                self.data.get_reduction_size(),
            )
        else:
            return extract_read_writes(
                partial(self.data.store_output, self.name, indexer),
                self.data.get_size(),
            )


@dataclasses.dataclass
class ExternKernel(Buffer):
    inputs: List[Buffer]

    def get_read_writes(self):
        return dependencies.ReadWrites(
            {dependencies.StarDep(x.get_name()) for x in self.inputs},
            {dependencies.StarDep(self.get_name())},
        )

    def codegen(self, wrapper):
        args = [x.codegen_reference() for x in self.inputs]
        args.append(f"out={self.codegen_reference()}")
        wrapper.body.writeline(f"{self.kernel}({', '.join(args)})")

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


class MatrixMultiply(ExternKernel):
    kernel = "torch.mm"

    @classmethod
    def create(cls, a, b):
        *m, k1 = a.get_size()
        k2, n = b.get_size()
        graph.sizevars.guard_equals(k1, k2)
        a = cls.realize_input(a)
        b = cls.realize_input(b)
        if len(m) != 1 and not a.get_layout().is_contiguous():
            a = cls.copy_input(a)
        else:
            a = cls.require_stride1(a)
        b = cls.require_stride1(b)
        data = MatrixMultiply(
            name=None,
            layout=FlexibleLayout(
                device=a.get_device(),
                dtype=a.get_dtype(),
                size=list(m) + [n],
            ),
            inputs=[a, b],
        )
        data.name = graph.register_buffer(data)
        return data


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
        if isinstance(self.data, TensorBox):
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

    def mark_reuse(self, users):
        pass  # TODO(jansel): realize the buffers?


class StorageBox(MutableBox):
    def realize(self):
        if isinstance(self.data, (ComputedBuffer, ExternKernel, InputBuffer)):
            return self.data.name
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
        self.data.name = graph.register_buffer(self.data)
        return self.data.name
