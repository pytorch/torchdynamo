import math
from typing import List
from typing import Optional

import functorch._src.decompositions
import torch
from torch import Tensor

from torchinductor import config

aten = torch.ops.aten

# python key tracing is broken in the latest pytorch, but when we update we can do:
# from torch._decomp import get_decompositions
# decompositions = get_decompositions([...])

# note AOT Autograd decomps are included by default in torchdynamo
decompositions = {}


def register_decomposition(ops):
    return functorch._src.decompositions.register_decomposition(ops, decompositions)


@register_decomposition([aten.clamp])
def clamp(x, min=None, max=None):
    if min is not None:
        x = torch.maximum(x, torch.tensor(min, dtype=x.dtype, device=x.device))
    if max is not None:
        x = torch.minimum(x, torch.tensor(max, dtype=x.dtype, device=x.device))
    return x


@register_decomposition([aten._softmax])
def _softmax(x, dim, half_to_float):
    # TODO(jansel): check numerical stability (see SoftMaxKernel.cpp)
    if half_to_float and x.dtype in (torch.bfloat16, torch.float16):
        x = x.to(torch.float32)
    x = torch.exp(x)
    x_sum = torch.sum(x, dim, keepdim=True)
    scale = torch.reciprocal(x_sum)
    return x * scale


@register_decomposition([aten._log_softmax])
def _log_softmax(x, dim, half_to_float):
    # TODO(jansel): check numerical stability (see SoftMaxKernel.cpp)
    if half_to_float and x.dtype in (torch.bfloat16, torch.float16):
        x = x.to(torch.float32)
    x_sum = torch.log(torch.sum(torch.exp(x), dim, keepdim=True))
    return x - x_sum


@register_decomposition([aten.t])
def t(x):
    ndim = x.ndimension()
    if x.ndim in (0, 1):
        return x
    assert ndim == 2
    return torch.transpose(x, 0, 1)


@register_decomposition([aten.transpose.int])
def transpose(x, dim0: int, dim1: int):
    dims = list(range(x.ndim))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    return torch.permute(x, dims)


@register_decomposition([aten.addmm])
def addmm(input, mat1, mat2):
    return torch.mm(mat1, mat2) + input


@register_decomposition([aten.elu])
def elu(self, alpha=1, scale=1, input_scale=1):
    negcoef = alpha * scale
    return torch.where(
        self <= 0, (torch.exp(self * input_scale) - 1) * negcoef, self * scale
    )


@register_decomposition([aten.tanh])
def tanh(x):
    return 2.0 / (1.0 + torch.exp(-2.0 * x)) - 1.0


@register_decomposition([aten.hardtanh])
def hardtanh(x, min_val=-1.0, max_val=1.0):
    return torch.clamp(x, min_val, max_val)


@register_decomposition([aten.hardsigmoid])
def hardsigmoid(x):
    return torch.clamp(x / 6.0 + 0.5, 0.0, 1.0)


@register_decomposition([aten.hardswish])
def hardswish(x):
    return torch.where(
        torch.gt(x, -3),
        torch.where(torch.lt(x, 3), x * (x + 3.0) / 6.0, x),
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
    )


@register_decomposition([aten.leaky_relu])
def leaky_relu(x, negative_slope=0.01):
    return torch.relu(x) + (-negative_slope) * torch.relu(-x)


@register_decomposition([aten.rsqrt])
def rsqrt(x):
    return torch.reciprocal(torch.sqrt(x))


@register_decomposition([aten.log2])
def log2(x):
    return torch.log(x) * (1.0 / math.log(2.0))


@register_decomposition([aten.gelu])
def gelu(x, approximate="none"):
    if config.approximations or approximate != "none":
        # tanh approximation is much faster
        return (
            0.5
            * x
            * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)))
        )
    else:
        return x * 0.5 * (1.0 + torch.special.erf(x * math.sqrt(0.5)))


@register_decomposition([aten.special_erf])
def special_erf(x):
    # TODO(jansel): this might be crazy slow.  Triton doesn't have the
    #               cuda ::erf() builtin.  I've made a feature request for this,
    #               so it may be coming soon.

    # from https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = torch.sign(x)
    x = torch.abs(x)

    # A & S 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * torch.exp(-x * x)

    return sign * y


@register_decomposition([aten.rsub.Tensor, aten.rsub.Scalar])
def rsub(a, b):
    if isinstance(b, (int, float)):
        b = torch.tensor(b, dtype=a.dtype, device=a.device)
    return b - a


@register_decomposition([aten.masked_fill.Scalar])
def masked_fill(value, mask, other):
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=value.dtype, device=value.device)
    value, mask, other = torch.broadcast_tensors(value, mask, other)
    return torch.where(mask, other, value)


def _batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training: bool,
    momentum: float,
    eps: float,
):
    assert not training, "TODO: support training"
    assert input.ndimension() == 4
    view_size = [1, -1, 1, 1]

    # TODO(jansel): try broadcasting earlier to get things into a single kernel

    invstd = torch.reciprocal(torch.sqrt(running_var + eps))
    if weight is None:
        weight = 1
    if bias is None:
        bias = 0
    alpha = invstd * weight
    beta = bias - running_mean * alpha
    result = input * alpha.view(view_size) + beta.view(view_size)
    return result


@register_decomposition([aten.native_batch_norm])
def native_batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training: bool,
    momentum: float,
    eps: float,
):
    result = _batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    null = torch.tensor([], device=input.device)
    return (result, null, null)


@register_decomposition([aten.cudnn_batch_norm])
def cudnn_batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training: bool,
    momentum: float,
    eps: float,
):
    result = _batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    null = torch.tensor([], device=input.device)
    null_uint8 = torch.tensor([], device=input.device, dtype=torch.uint8)
    return (result, null, null, null_uint8)


@register_decomposition(aten.frac.default)
def frac(input: Tensor) -> Tensor:
    return input - torch.trunc(input)


@register_decomposition(aten.celu.default)
def celu(input: Tensor, alpha: float = 1.0) -> Tensor:
    inv_alpha = 1.0 / alpha
    return aten.elu(input, alpha, 1.0, inv_alpha)


@register_decomposition(aten.mish.default)
# @pw_cast_for_opmath
def mish(x: Tensor) -> Tensor:
    return x * x.exp().log1p().tanh()


@register_decomposition(aten.softplus.default)
def softplus(a: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    a_beta = a * beta
    return torch.where((a_beta) > threshold, a, (a_beta).exp().log1p() / beta)


@register_decomposition(aten.softshrink.default)
def softshrink(a: Tensor, lambd: float = 0.5) -> Tensor:
    return torch.where(a > lambd, a - lambd, torch.where(a < -lambd, a + lambd, 0))


@register_decomposition(aten.deg2rad.default)
def deg2rad(a: Tensor) -> Tensor:
    M_PI_180 = 0.017453292519943295769236907684886127134428718885417
    return a * M_PI_180


@register_decomposition(aten.rad2deg.default)
def rad2deg(a: Tensor) -> Tensor:
    M_180_PI = 57.295779513082320876798154814105170332405472466564
    return a * M_180_PI


@register_decomposition(aten.relu.default)
def relu(a: Tensor) -> Tensor:
    return torch.clamp(a, min=0)


@register_decomposition(aten.sinc.default)
# @pw_cast_for_int_to_real
def sinc(a: Tensor) -> Tensor:
    PI = 3.14159265358979323846
    pi_a = PI * a
    return torch.where(a == 0.0, 1.0, torch.sin(pi_a) / pi_a)


@register_decomposition(aten.heaviside.default)
def heaviside(input: Tensor, value: Tensor) -> Tensor:
    sign = torch.where(input > 0, 1, 0)
    return torch.where(input == 0, value, sign)


@register_decomposition(aten.logit)
# @pw_cast_for_int_to_real
def logit(self: Tensor, eps: Optional[float] = None) -> Tensor:
    if eps is None:
        eps = -1.0
    lo = eps
    hi = 1 - eps
    self = torch.clamp(self, lo, hi)
    return (self / (1 - self)).log()


def _squeeze_multiple(self: Tensor, dims: List[int]) -> Tensor:
    ndim = self.dim()
    for idx in range(ndim - 1, -1, -1):
        if idx in dims or idx - ndim in dims:
            self = self.squeeze(idx)
    return self


# based on https://github.com/pytorch/pytorch/pull/77219
@register_decomposition([aten.logsumexp.default])
def logsumexp(self, dim, keepdim=False) -> Tensor:
    if self.numel() == 0:
        return torch.sum(torch.exp(self), dim, keepdim).log()
    maxes = torch.amax(self, dim, keepdim=True)
    maxes_squeezed = maxes if keepdim else _squeeze_multiple(maxes, dim)
    maxes_squeezed = torch.masked_fill(
        maxes_squeezed, maxes_squeezed.abs() == float("inf"), 0
    )
    result = torch.sum(torch.exp(self - maxes), dim, keepdim)
    return result.log().add(maxes_squeezed)


def check_stack_inputs(tensors: List[Tensor]):
    entry_shape = tensors[0].shape
    for i in range(1, len(tensors)):
        assert tensors[i].shape == entry_shape, (
            f"stack expects each tensor to be equal size, but got {entry_shape} at entry 0"
            f"and {tensors[i].shape} at entry {i}"
        )


def get_stack_inputs(tensors: List[Tensor], dim: int):
    check_stack_inputs(tensors)
    return [t.unsqueeze(dim) for t in tensors]


# https://github.com/pytorch/pytorch/blob/f6eb81178633e/torch/_decomp/decompositions.py#L1235
@register_decomposition([aten.stack.default])
def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    assert len(tensors) > 0, "stack expects a non-empty TensorList"
    wrapped_dim = canonicalize_dim(tensors[0].dim() + 1, dim)
    if wrapped_dim < tensors[0].dim() and not tensors[0].is_sparse:
        check_stack_inputs(tensors)
        result_sizes = list(tensors[0].shape)
        result_sizes.insert(wrapped_dim, len(tensors))
        out = torch.cat(tensors, wrapped_dim)
        return out.view(result_sizes)
    else:
        return torch.cat(get_stack_inputs(tensors, wrapped_dim), dim)


# https://github.com/pytorch/pytorch/blob/c25bdeea26f95/torch/_prims/utils.py#L250
def canonicalize_dim(rank: int, idx: int) -> int:
    # TODO: add a comment for why this is
    _rank = rank if rank != 0 else 1

    if idx >= 0 and idx < _rank:
        return idx

    if idx < 0:
        _idx = idx + _rank
    else:
        _idx = idx

    if _idx < 0 or _idx > _rank:
        msg = "Received out of bounds index {0} for tensor of rank {1}!".format(
            idx, rank
        )
        raise ValueError(msg)

    return _idx


# https://github.com/pytorch/pytorch/blob/c25bdeea26f95d/torch/_prims/utils.py#L273
def canonicalize_dims(rank: int, indices):
    if isinstance(indices, int):
        return canonicalize_dim(rank, indices)

    return tuple(canonicalize_dim(rank, x) for x in indices)
