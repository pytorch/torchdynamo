import math
from typing import List

import functorch._src.decompositions
import torch
from torch import Tensor

from torch._decomp import get_decompositions
from torchinductor import config

aten = torch.ops.aten

decompositions = get_decompositions([
    aten.l1_loss,
    aten.mse_loss,
    aten.stack,
    aten.native_layer_norm,
    aten.native_batch_norm,
    aten.cudnn_batch_norm,
    aten.leaky_relu,
    aten.hardtanh,
    aten.hardsigmoid,
    aten.hardswish,
    aten.transpose.int,
])


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


