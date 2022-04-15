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
    assert half_to_float is False
    x = torch.exp(x)
    x_sum = torch.sum(x, dim, keepdim=True)
    return x / x_sum
