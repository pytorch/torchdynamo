import torch
from functorch._src.decompositions import register_decomposition

aten = torch.ops.aten
decompositions = {}


@register_decomposition([aten.addmm], decompositions)
def rsub(a, b, alpha=1):
    return -aten.sub(a, b)
