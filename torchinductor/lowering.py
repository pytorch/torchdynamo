import functools
import itertools
from typing import List

import sympy
import torch
import torch.fx

from . import config
from . import ir
from .codegen.common import product
from .ir import ExpandView
from .ir import PermuteView
from .ir import Pointwise
from .ir import Reduction
from .ir import SqueezeView
from .ir import TensorBox
from .ir import View
from .virtualized import V
from .virtualized import ops

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
            for i in range(len(args)):
                if isinstance(args[i], ir.Constant):
                    args[i] = ir.Constant(
                        args[i].value, dtype, args[indices[0]].get_device()
                    )

        if broadcast and indices:
            for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
                args[i] = x
            for i in range(len(args)):
                if isinstance(args[i], ir.Constant):
                    args[i] = ExpandView.create(
                        args[i], list(args[indices[0]].get_size())
                    )

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
            V.graph.sizevars.guard_equals(a, b)
            if len(sympy.expand(b).free_symbols) < len(sympy.expand(a).free_symbols):
                output.append(b)  # prefer shorter formula
            else:
                output.append(a)
    return tuple(reversed(output))


def promote_constants(inputs):
    if not any(isinstance(x, (int, float)) for x in inputs):
        return inputs
    ex = next(x for x in inputs if isinstance(x, TensorBox))
    return [
        (
            ExpandView.create(
                ir.Constant(x, ex.get_dtype(), ex.get_device()), list(ex.get_size())
            )
            if isinstance(x, (int, float))
            else x
        )
        for x in inputs
    ]


def make_pointwise(fn, override_dtype=None, override_device=None):
    def inner(*inputs: List[TensorBox]):
        inputs = promote_constants(inputs)
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        for other in inputs[1:]:
            assert isinstance(other, ir.BaseConstant) or len(ranges) == len(
                other.get_size()
            ), f"ndim mismatch {fn} {ranges} {other.get_size()}"

        def inner_fn(index):
            assert len(index) == len(ranges), f"wrong ndim {index} {ranges}"
            return fn(*[load(index) for load in loaders])

        return Pointwise.create(
            device=override_device or inputs[0].get_device(),
            dtype=override_dtype or inputs[0].get_dtype(),
            inner_fn=inner_fn,
            ranges=ranges,
        )

    return inner


def to_dtype(x: TensorBox, dtype: torch.dtype):
    if x.get_dtype() == dtype:
        return x

    def _to_dtype(x):
        return ops.to_dtype(x, dtype)

    return make_pointwise(_to_dtype, override_dtype=dtype)(x)


def register_pointwise(
    aten_fn,
    name=None,
    broadcast=True,
    type_promote=True,
    override_dtype=None,
    override_device=None,
):
    """A pointwise function that maps ops.{name} to inputs"""
    name = name or aten_fn.__name__

    def fn(*args, **kwargs):
        return getattr(ops, name)(*args, **kwargs)

    fn = make_pointwise(
        fn, override_dtype=override_dtype, override_device=override_device
    )
    fn = register_lowering(aten_fn, broadcast=broadcast, type_promote=type_promote)(fn)
    return fn


@register_lowering(aten.where, broadcast=True, type_promote=False)
def where(cond, a, b):
    def fn(*args):
        return ops.where(*args)

    dtype = torch.promote_types(a.get_dtype(), b.get_dtype())
    return make_pointwise(fn, override_dtype=dtype)(
        cond, to_dtype(a, dtype), to_dtype(b, dtype)
    )


@register_lowering(aten.broadcast_tensors, broadcast=False, type_promote=False)
def broadcast_tensors(*inputs):
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        return broadcast_tensors(*inputs[0])
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


@register_lowering(aten.alias)
@register_lowering(aten.detach)
def detach(x):
    assert isinstance(x, TensorBox)
    return x  # AOT autograd handles this for us


@register_lowering(aten.squeeze)
def squeeze(x, dim=None):
    assert isinstance(x, TensorBox)
    if dim is None:
        return TensorBox(SqueezeView.create(x.data))

    dim = _validate_dim(x, dim, 0)
    new_shape = list(x.get_size())
    removed = new_shape.pop(dim)
    assert removed == 1, removed
    return view(x, new_shape)


@register_lowering(aten.expand)
def expand(x, sizes):
    if isinstance(x, ir.BaseConstant):
        return ExpandView.create(x, tuple(sizes))
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    if tuple(x.get_size()) == tuple(sizes):
        return x
    x.mark_reuse(V.graph.sizevars.size_hint(product(sizes) / product(x.get_size())))
    return TensorBox(ExpandView.create(x.data, tuple(sizes)))


@register_lowering(aten.repeat)
def repeat(x, repeats):
    old_size = list(x.get_size())
    if len(repeats) > len(old_size):
        old_size = [sympy.Integer(1)] * (len(repeats) - len(old_size)) + old_size
        x = view(x, list(old_size))
    assert len(repeats) == len(x.get_size())

    new_size = list(x.get_size())

    for i in range(len(repeats)):
        assert repeats[i] >= 1
        if repeats[i] > 1:
            new_size[i] = new_size[i] * repeats[i]

    if all((a == 1 or b == 1) for a, b in zip(repeats, old_size)):
        return expand(x, new_size)

    def inner_fn(index):
        assert len(index) == len(repeats)
        index = list(index)
        for i in range(len(repeats)):
            if repeats[i] > 1:
                if old_size[i] == 1:
                    index[i] = sympy.Integer(0)
                else:
                    index[i] = ir.ModularIndexing(index[i], 1, old_size[i])
        return x_loader(index)

    x.mark_reuse(V.graph.sizevars.size_hint(product(new_size) / product(old_size)))
    x_loader = x.make_loader()
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(new_size),
    )


@register_lowering(aten._unsafe_view)
@register_lowering(aten.view)
def view(x, sizes):
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    return TensorBox(View.create(x.data, sizes))


@register_lowering(aten.permute)
def permute(x, dims):
    assert isinstance(x, TensorBox)
    assert isinstance(dims, (list, tuple))
    return TensorBox(PermuteView.create(x.data, tuple(dims)))


@register_lowering(aten.slice)
def slice_(x, dim, start, end, step=1):
    assert isinstance(x, TensorBox)
    dim = _validate_dim(x, dim, 0)
    return TensorBox(ir.SliceView.create(x.data, dim, start, end, step))


@register_lowering(aten.cat)
def cat(inputs, dim=0):
    if len(inputs) == 1:
        return inputs[0]
    dim = _validate_dim(inputs[0], dim, 0)
    return TensorBox(ir.ConcatKernel.create(inputs, dim))


@register_lowering(aten.select)
def select(x, dim, idx):
    idx = View.handle_negative_index(idx, x.get_size()[dim])
    return squeeze(slice_(x, dim, idx, idx + 1), dim)


@register_lowering(aten.split)
def split(x, sizes, dim):
    dim = _validate_dim(x, dim, 0)
    x_size = V.graph.sizevars.guard_static_shape(x.get_size()[dim])
    if isinstance(sizes, int):
        sizes = [sizes] * ((x_size + sizes - 1) // sizes)
    result = []
    start = 0
    for size in sizes:
        end = start + size
        result.append(slice_(x, dim, start, end))
        start = end
    return result


@register_lowering(aten.unsqueeze)
def unsqueeze(x, dim):
    dim = _validate_dim(x, dim, 1)
    new_shape = list(x.get_size())
    new_shape.insert(dim, sympy.Integer(1))
    return view(x, new_shape)


def _validate_dim(x, dim, offset):
    assert isinstance(dim, int)
    ndim = len(x.get_size())
    if dim < 0:
        dim += ndim + offset
    assert 0 <= dim < ndim + offset
    return dim


@register_lowering(aten.glu)
def glu(x, dim=-1):
    dim = _validate_dim(x, dim, 0)
    new_len = V.graph.sizevars.guard_static_shape(x.get_size()[dim]) // 2
    a = slice_(x, dim, 0, new_len)
    b = slice_(x, dim, new_len, new_len * 2)
    return mul(a, sigmoid(b))


@register_lowering(aten.mm)
def mm(a: TensorBox, b: TensorBox):
    return TensorBox.create(ir.MatrixMultiply.create(a, b))


@register_lowering(aten.bmm)
def bmm(a: TensorBox, b: TensorBox):
    return TensorBox.create(ir.BatchMatrixMultiply.create(a, b))


@register_lowering(aten._embedding_bag, type_promote=False)
def _embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
):
    kernel = (
        aten._embedding_bag_forward_only if config.forward_only else aten._embedding_bag
    )
    return list(
        map(
            TensorBox.create,
            ir.FallbackKernel.create(
                kernel,
                weight,
                indices,
                offsets,
                scale_grad_by_freq,
                mode,
                False,  # TODO(jansel): support sparse
                per_sample_weights,
                include_last_offset,
            ),
        )
    )


@register_lowering(aten.convolution)
def convolution(
    x: TensorBox,
    weight: TensorBox,
    bias: TensorBox,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
):
    return TensorBox.create(
        ir.Convolution.create(
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
    )


@register_lowering(aten.clone)
def clone(x, *, memory_format=0):
    # TODO(jansel): memory format
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
        ranges=list(x.get_size()),
    )


@register_lowering(torch.arange)
def arange(start, end=None, step=1, *, dtype=None, device=None):
    if end is None:
        end = start
        start = 0

    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(step, int)

    dtype = dtype or torch.get_default_dtype()
    length = (end - start) // step
    start = sympy.Integer(start)
    step = sympy.Integer(step)

    return Pointwise.create(
        device=device or torch.tensor(0.0).device,
        dtype=dtype,
        inner_fn=lambda index: ops.index_expr(step * index[0] + start, dtype),
        ranges=[sympy.Integer(length)],
    )


DTYPE_ID_LOOKUP = {
    6: torch.float32,
    7: torch.float64,
}


def constant_like(n):
    def _constant_like(x, *, dtype, device, layout, pin_memory=False):
        assert not pin_memory
        assert layout == 0
        assert (
            dtype in DTYPE_ID_LOOKUP
        ), f"id {dtype}={x.get_dtype()} missing from DTYPE_ID_LOOKUP"
        dtype = DTYPE_ID_LOOKUP[dtype]
        assert dtype == x.get_dtype(), "is this correct?"
        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=lambda index: ops.constant(n, dtype),
            ranges=list(x.get_size()),
        )

    return _constant_like


register_lowering(aten.zeros_like)(constant_like(0))
register_lowering(aten.ones_like)(constant_like(1))


@register_lowering(aten.gather, type_promote=False)
def gather(x, dim, index):
    assert isinstance(x, TensorBox)
    assert isinstance(dim, int)
    assert "int" in str(index.get_dtype())
    assert 0 <= dim < len(x.get_size())

    x_loader = x.make_loader()
    index_loader = index.make_loader()

    def fn(idx):
        idx = list(idx)
        var_index = index_loader(idx)
        idx[dim] = sympy.Symbol(str(var_index))
        return x_loader(idx)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=index.get_size(),
    )


@register_lowering(aten.embedding, type_promote=False)
def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    assert not sparse
    assert isinstance(weight, TensorBox)
    assert isinstance(indices, TensorBox)
    assert "int" in str(indices.get_dtype())

    weight_loader = weight.make_loader()
    indices_loader = indices.make_loader()
    indices_ndim = len(indices.get_size())
    new_size = [*indices.get_size(), *weight.get_size()[1:]]

    def fn(idx):
        assert len(idx) == len(new_size), f"{idx} != {new_size}"
        var_index = indices_loader(idx[:indices_ndim])
        weight_idx = [sympy.Symbol(str(var_index))] + [*idx[indices_ndim:]]
        return weight_loader(weight_idx)

    return Pointwise.create(
        device=weight.get_device(),
        dtype=weight.get_dtype(),
        inner_fn=fn,
        ranges=new_size,
    )


@register_lowering(aten.max_pool2d_with_indices)
def max_pool2d_with_indices(
    x, kernel_size, stride=(1, 1), padding=0, dilation=1, ceil_mode=False
):
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert padding == 0 or len(padding) == 2
    assert dilation == 1 or tuple(dilation) == (1, 1), "TODO(jansel): support dilation"
    assert len(x.get_size()) in (3, 4)
    if padding == 0:
        padding = [0, 0]

    x.realize()  # we will read this many times, so make sure it is computed

    x_loader = x.make_loader()

    *batch, c, h, w = x.get_size()

    h_out = ir.IndexingDiv(
        h + 2 * padding[0] - (kernel_size[0] - 1) + (stride[0] - 1), stride[0]
    )
    w_out = ir.IndexingDiv(
        w + 2 * padding[1] - (kernel_size[1] - 1) + (stride[1] - 1), stride[1]
    )

    if ceil_mode:
        h_out_ = ir.IndexingDiv(
            h + 2 * padding[0] - (kernel_size[0] - 1) + 2 * (stride[0] - 1), stride[0]
        )
        w_out_ = ir.IndexingDiv(
            w + 2 * padding[1] - (kernel_size[1] - 1) + 2 * (stride[1] - 1), stride[1]
        )

        if (
            V.graph.sizevars.size_hint(h_out - h_out_) == 0
            and V.graph.sizevars.size_hint(w_out - w_out_) == 0
        ):
            # ceil mode is actually a no-op, lets guard on that
            V.graph.sizevars.guard_equals(h_out, h_out_)
            V.graph.sizevars.guard_equals(w_out, w_out_)
            ceil_mode = False
        else:
            h_out = h_out_
            w_out = w_out_

    new_size = list(batch) + [c, h_out, w_out]

    def fn(idx, return_index):
        *prefix, bh, bw = idx
        maxval = None
        maxindex = None
        for ih, iw in itertools.product(range(kernel_size[0]), range(kernel_size[1])):
            ih = bh * stride[0] + ih - padding[0]
            iw = bw * stride[1] + iw - padding[1]
            if padding[0] or padding[1] or ceil_mode:
                # TODO(jansel): rewrite this to use a boundary condition helper
                mask = ops.and_(
                    ops.and_(
                        ops.ge(
                            ops.index_expr(ih, torch.int64),
                            ops.index_expr(sympy.Integer(0), torch.int64),
                        ),
                        ops.ge(
                            ops.index_expr(iw, torch.int64),
                            ops.index_expr(sympy.Integer(0), torch.int64),
                        ),
                    ),
                    ops.and_(
                        ops.lt(
                            ops.index_expr(ih, torch.int64),
                            ops.index_expr(h, torch.int64),
                        ),
                        ops.lt(
                            ops.index_expr(iw, torch.int64),
                            ops.index_expr(w, torch.int64),
                        ),
                    ),
                )
                val = ops.masked(
                    mask, lambda: x_loader([*prefix, ih, iw]), float("-inf")
                )
            else:
                val = x_loader([*prefix, ih, iw])
            index = ops.index_expr(ih * w + iw, torch.int64)
            if maxval is None:
                maxindex = index
                maxval = val
            else:
                maxindex = ops.where(ops.gt(val, maxval), index, maxindex)
                maxval = ops.maximum(val, maxval)
        if return_index:
            return maxindex
        else:
            return maxval

    r1 = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=functools.partial(fn, return_index=False),
        ranges=new_size,
    )
    r2 = Pointwise.create(
        device=x.get_device(),
        dtype=torch.int64,
        inner_fn=functools.partial(fn, return_index=True),
        ranges=new_size,
    )
    # TODO(jansel): should we force these to be realized?
    return r1, r2


@register_lowering(aten._adaptive_avg_pool2d)
def _adaptive_avg_pool2d(x, output_size):
    assert isinstance(x, TensorBox)
    assert len(output_size) == 2
    return TensorBox.create(ir.AdaptiveAvgPool2d.create(x, output_size))


def _validate_reduction_axis(x, axis):
    size = x.get_size()
    if isinstance(axis, int):
        axis = [axis]
    elif axis is None:
        axis = range(len(size))
    axis = list(axis)
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(size)
        assert 0 <= axis[i] < len(size)
    assert len(set(axis)) == len(axis), "reduction axis not unique"
    return axis


def make_reduction(reduction_type: str):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        if dtype is not None:
            x = to_dtype(x, dtype)
        assert reduction_type == "sum" or axis is None, "TODO: max with index"
        size = x.get_size()
        axis = set(_validate_reduction_axis(x, axis))

        kept_sizes = []
        kept_idx = []
        reduced_sizes = []
        reduced_idx = []
        for i in range(len(size)):
            if i in axis:
                reduced_idx.append(i)
                reduced_sizes.append(size[i])
            else:
                kept_idx.append(i)
                kept_sizes.append(size[i])

        def loader(index, reduction_index):
            assert len(reduction_index) == len(reduced_idx)
            if keepdims:
                assert len(index) == len(size)
                assert all(index[i] == 0 for i in reduced_idx)
                index = [index[i] for i in kept_idx]
            assert len(index) == len(kept_idx)
            new_index = [None] * (len(index) + len(reduction_index))
            for idx, var in itertools.chain(
                zip(kept_idx, index), zip(reduced_idx, reduction_index)
            ):
                new_index[idx] = var
            return inner_loader(new_index)

        if keepdims:
            new_size = list(size)
            for i in reduced_idx:
                new_size[i] = sympy.Integer(1)
        else:
            new_size = kept_sizes

        inner_loader = x.make_loader()
        result = Reduction.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=loader,
            ranges=new_size,
            reduction_ranges=reduced_sizes,
            reduction_type=reduction_type,
        )
        result.realize()
        return result

    return inner


@register_lowering(aten.mean)
def mean(x, axis=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    sum_result = sum_(x, axis, keepdim)
    denom = product(size[i] for i in axis)
    denom = ir.IndexingConstant(denom, x.get_dtype(), x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    return div(sum_result, denom)


@register_lowering(aten.var)
def var_(x, axis, correction=1, keepdim=False):
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    diffs = square(sub(x, mean(x, axis, keepdim=True)))
    sum_result = sum_(diffs, axis, keepdim)

    denom = product(size[i] for i in axis)
    if correction:
        denom = denom - correction
    denom = ir.IndexingConstant(denom, x.get_dtype(), x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    return div(sum_result, denom)


@register_lowering(aten.std)
def std(x, axis, correction=1, keepdim=False):
    return sqrt(var_(x, axis, correction, keepdim=keepdim))


sum_ = register_lowering(aten.sum)(make_reduction("sum"))
register_lowering(aten.max)(make_reduction("max"))
register_lowering(aten.min)(make_reduction("min"))

div = register_pointwise(aten.div)
exp = register_pointwise(aten.exp)
mul = register_pointwise(aten.mul)
relu = register_pointwise(aten.relu)
sigmoid = register_pointwise(aten.sigmoid)
sqrt = register_pointwise(aten.sqrt)
square = register_pointwise(aten.square)
sub = register_pointwise(aten.sub)

register_pointwise(aten.abs)
register_pointwise(aten.add)
register_pointwise(aten.log)
register_pointwise(aten.logical_not)
register_pointwise(aten.maximum)
register_pointwise(aten.minimum)
register_pointwise(aten.neg)
register_pointwise(aten.reciprocal)
register_pointwise(aten.sign)
register_pointwise(aten.silu)

register_pointwise(aten.le, type_promote=False, override_dtype=torch.bool)
register_pointwise(aten.lt, type_promote=False, override_dtype=torch.bool)
register_pointwise(aten.ge, type_promote=False, override_dtype=torch.bool)
register_pointwise(aten.gt, type_promote=False, override_dtype=torch.bool)
register_pointwise(aten.eq, type_promote=False, override_dtype=torch.bool)
register_pointwise(aten.ne, type_promote=False, override_dtype=torch.bool)
