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
    return x / x_sum


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
