import math

import torch
from functorch._src.decompositions import register_decomposition

aten = torch.ops.aten
decompositions = {}


@register_decomposition([aten.clamp], decompositions)
def clamp(x, min=None, max=None):
    if min is not None:
        x = torch.maximum(x, torch.tensor(min, dtype=x.dtype, device=x.device))
    if max is not None:
        x = torch.minimum(x, torch.tensor(max, dtype=x.dtype, device=x.device))
    return x


@register_decomposition([aten._softmax], decompositions)
def _softmax(x, dim, half_to_float):
    # TODO(jansel): check numerical stability (see SoftMaxKernel.cpp)
    if half_to_float and x.dtype in (torch.bfloat16, torch.float16):
        x = x.to(torch.float32)
    x = torch.exp(x)
    x_sum = torch.sum(x, dim, keepdim=True)
    scale = torch.reciprocal(x_sum)
    return x * scale


@register_decomposition([aten._log_softmax], decompositions)
def _log_softmax(x, dim, half_to_float):
    # TODO(jansel): check numerical stability (see SoftMaxKernel.cpp)
    if half_to_float and x.dtype in (torch.bfloat16, torch.float16):
        x = x.to(torch.float32)
    assert half_to_float is False, "TODO"
    x_sum = torch.log(torch.sum(torch.exp(x), dim, keepdim=True))
    return x - x_sum


@register_decomposition([aten.t], decompositions)
def t(x):
    ndim = x.ndimension()
    if x.ndim in (0, 1):
        return x
    assert ndim == 2
    return torch.transpose(x, 0, 1)


@register_decomposition([aten.transpose.int], decompositions)
def transpose(x, dim0: int, dim1: int):
    dims = list(range(x.ndim))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    return torch.permute(x, dims)


@register_decomposition([aten.addmm], decompositions)
def addmm(input, mat1, mat2):
    return torch.mm(mat1, mat2) + input


@register_decomposition([aten.elu], decompositions)
def elu(self, alpha=1, scale=1, input_scale=1):
    negcoef = alpha * scale
    return torch.where(
        self <= 0, (torch.exp(self * input_scale) - 1) * negcoef, self * scale
    )


@register_decomposition([aten.tanh], decompositions)
def tanh(x):
    return 2.0 / (1.0 + torch.exp(-2.0 * x)) - 1.0


@register_decomposition([aten.leaky_relu], decompositions)
def leaky_relu(x, negative_slope=0.01):
    return torch.relu(x) + (-negative_slope) * torch.relu(-x)


@register_decomposition([aten.gelu], decompositions)
def gelu(x):
    # tanh approximation:
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x * x * x)))
    return x * 0.5 * (1.0 + torch.special.erf(x * math.sqrt(0.5)))


@register_decomposition([aten.special_erf], decompositions)
def special_erf(x):
    # TODO(jansel): this might be crazy slow.  Triton doesn't have the cuda ::erf() builtin.

    # note: triton may add a builtin for this
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


@register_decomposition([aten.rsub.Tensor, aten.rsub.Scalar], decompositions)
def rsub(a, b):
    if isinstance(b, (int, float)):
        b = torch.tensor(b, dtype=a.dtype, device=a.device)
    return b - a


@register_decomposition([aten.pow.Tensor_Scalar], decompositions)
def pow(a, b):
    # triton doesn't support pow, so need to rewrite it
    if isinstance(b, float) and b == int(b):
        return pow(a, int(b))
    if isinstance(b, int) and b == 0:
        return torch.ones_like(a)
    elif isinstance(b, int) and b == 1:
        return a
    elif isinstance(b, int):
        if b < 0:
            return pow(torch.reciprocal(a), -b)

        result = pow(a, b // 2)
        result = result * result
        if (b % 2) == 1:
            result = result * a
        return result
    else:
        assert False, "TODO: check correctness here"
        return torch.sign(a) * torch.exp(torch.log(torch.abs(a)) * b)


@register_decomposition([aten.masked_fill.Scalar], decompositions)
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


@register_decomposition([aten.native_batch_norm], decompositions)
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


@register_decomposition([aten.cudnn_batch_norm], decompositions)
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
