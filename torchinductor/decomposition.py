import logging
import math
import numbers
from enum import Enum
from typing import Optional
from typing import Tuple

import functorch._src.decompositions
import torch
from functorch._src.aot_autograd import aot_autograd_decompositions
from torch import Tensor
from torch._decomp import get_decompositions

from torchinductor import config

log = logging.getLogger(__name__)
aten = torch.ops.aten
log = logging.getLogger(__name__)

decompositions = get_decompositions(
    [
        aten._adaptive_avg_pool2d_backward,
        aten.avg_pool2d_backward,
        aten.clamp_max,
        aten.clamp_min,
        aten.cudnn_batch_norm,
        aten.cudnn_batch_norm_backward,
        aten.elu_backward,
        aten._embedding_bag,
        aten.embedding_dense_backward,
        aten.expand_as,
        aten._fused_moving_avg_obs_fq_helper,
        aten.gelu_backward,
        aten.glu_backward,
        aten.grid_sampler_2d,
        aten.hardsigmoid,
        aten.hardsigmoid_backward,
        aten.hardswish,
        aten.hardswish_backward,
        aten.hardtanh,
        aten.hardtanh_backward,
        aten.l1_loss,
        aten.leaky_relu,
        aten.leaky_relu_backward,
        aten._log_softmax_backward_data,
        aten.logsumexp.default,
        aten.max_pool2d_with_indices_backward,
        aten.mse_loss,
        aten.narrow,
        aten.native_batch_norm,
        aten.native_batch_norm_backward,
        aten.native_dropout_backward,
        aten.native_group_norm,
        aten.native_layer_norm,
        aten.native_layer_norm_backward,
        aten.norm,
        aten.reflection_pad2d_backward,
        aten.select_backward,
        aten.select_scatter,
        aten.sigmoid_backward,
        aten.silu_backward,
        aten.slice_backward,
        aten._softmax_backward_data,
        aten.stack,
        aten.tanh_backward,
        aten.threshold_backward,
        aten.transpose.int,
        aten.upsample_nearest2d_backward,
    ]
)
decompositions.update(aot_autograd_decompositions)

if not config.fallback_random:
    # these decomps have different results than eager mode
    decompositions.update(
        get_decompositions(
            [
                aten.native_dropout,
            ]
        )
    )


def register_decomposition(ops):
    for op in [ops] if callable(ops) else ops:
        if op in decompositions:
            log.warning(f"duplicate decomp: {ops}")
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


@register_decomposition([aten.round.decimals])
def round_dec(x, decimals=0):
    ten_pow_decimals = 10.0**decimals
    return aten.round(x * ten_pow_decimals) * (1.0 / ten_pow_decimals)


@register_decomposition([aten.div.Tensor_mode])
def div_mode(a, b, rounding_mode=None):
    result = aten.div(a, b)
    if rounding_mode == "floor":
        return torch.floor(result)
    if rounding_mode == "trunc":
        return torch.trunc(result)
    return result


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


@register_decomposition([aten.special_erf, aten.erf])
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
    if isinstance(b, numbers.Number):
        b = torch.tensor(b, dtype=a.dtype, device=a.device)
    return b - a


@register_decomposition([aten.masked_fill.Scalar])
def masked_fill(value, mask, other):
    if isinstance(other, numbers.Number):
        other = torch.tensor(other, dtype=value.dtype, device=value.device)
    value, mask, other = torch.broadcast_tensors(value, mask, other)
    return torch.where(mask, other, value)


@register_decomposition([aten.nan_to_num])
def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    if nan is None:
        nan = 0.0
    if posinf is None:
        posinf = torch.finfo(x.dtype).max
    if neginf is None:
        neginf = torch.finfo(x.dtype).min
    nan, posinf, neginf = (
        torch.tensor(v, dtype=x.dtype, device=x.device) for v in (nan, posinf, neginf)
    )
    x = torch.where(x != x, nan, x)
    x = torch.where(x == float("inf"), posinf, x)
    x = torch.where(x == float("-inf"), neginf, x)
    return x


@register_decomposition([aten.all.default])
def all(input):
    return torch.logical_not(torch.any(torch.logical_not(input)))


@register_decomposition([aten.all.dim])
def all_dim(input, dim, keeepdim=False):
    return torch.logical_not(torch.any(torch.logical_not(input), dim, keeepdim))


@register_decomposition(aten.hardswish_)
def hardswish_(x):
    return x.copy_(aten.hardswish(x))


@register_decomposition(aten.hardtanh_)
def hardtanh_(x, min_val=-1, max_val=1):
    return x.copy_(aten.hardtanh(x, min_val, max_val))


@register_decomposition(aten.leaky_relu_)
def leaky_relu_(x, negative_slope=0.01):
    return x.copy_(aten.leaky_relu(x, negative_slope))


@register_decomposition(aten.silu_)
def silu_(x):
    return x.copy_(aten.silu(x))


@register_decomposition(aten.masked_fill_)
def masked_fill_(x, mask, value):
    return x.copy_(aten.masked_fill(x, mask, value))


@register_decomposition([aten.log1p])
def log1p(x):
    return torch.log(x + 1)


@register_decomposition([aten.baddbmm])
def baddbmm(self, batch1, batch2, beta=1, alpha=1):
    result = torch.bmm(batch1, batch2)
    if not isinstance(alpha, numbers.Number) or alpha != 1:
        result = result * alpha
    if not isinstance(beta, numbers.Number) or beta != 1:
        self = self * beta
    return self + result


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


@register_decomposition(aten.nll_loss_forward)
def nll_loss_forward(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
) -> Tuple[Tensor, Tensor]:
    """
    This is copied from:
    https://github.com/pytorch/pytorch/pull/78491

    We should remove it when that PR lands.
    """

    # self can be [N, C] or [C]
    # target can be [N] or []

    n_dims = self.dim()
    channel_dim = 1
    if n_dims < 2:
        channel_dim = 0

    if weight is not None:
        # Here is a specific case with reduction mean and non-batched tensors
        # https://github.com/pytorch/pytorch/issues/61309
        # In this case weight is cancelled: w * x[t] / w -> x[t]
        if not (reduction == Reduction.MEAN.value and n_dims < 2):
            w = weight.unsqueeze(0) if n_dims > 1 else weight
            self = self * w

    target_ = target.unsqueeze(channel_dim)
    # target can be [N, 1] or [1]

    result = -torch.gather(self, channel_dim, target_).squeeze(channel_dim)

    ignore_index_mask = None
    if ignore_index >= 0:
        ignore_index_mask = target != ignore_index
        result = result * ignore_index_mask

    if reduction == Reduction.NONE.value and n_dims > 1:
        total_weight = self.new_full((), 0.0)
        return result, total_weight

    if weight is not None:
        w = weight.unsqueeze(0).expand(self.shape) if n_dims > 1 else weight
        wsum = torch.gather(w, channel_dim, target_).squeeze(channel_dim)
        if ignore_index_mask is not None:
            wsum = wsum * ignore_index_mask
        total_weight = wsum.sum()
    elif ignore_index_mask is not None:
        total_weight = ignore_index_mask.sum().to(self)
    else:
        total_weight = self.new_full((), 1.0 * result.numel())

    if result.dim() > 0:
        if reduction == Reduction.SUM.value:
            result = result.sum()
        elif reduction == Reduction.MEAN.value:
            if weight is None:
                result = (
                    result.sum() / total_weight if ignore_index >= 0 else result.mean()
                )
            else:
                result = result.sum() / total_weight

    return result, total_weight


@register_decomposition([aten.index_put])
def index_put(self, indices, values, accumulate=False):
    return torch.index_put_(self.clone(), indices, values, accumulate)


@register_decomposition([aten.narrow])
def narrow(self, dim, start, length):
    return aten.slice(self, dim, start, start + length)


@register_decomposition([aten.conj_physical])
def conj_physical(self):
    assert not self.is_complex(), "TODO: implement this"
    return self


@register_decomposition([aten.lift])
def lift(self):
    return self


@register_decomposition([aten.type_as])
def type_as(self, other):
    return self.type(other.type())


if not config.fallback_random:

    @register_decomposition([aten.bernoulli_])
    def bernoulli_(self, p=0.5):
        return self.copy_(torch.rand_like(self) < p)
