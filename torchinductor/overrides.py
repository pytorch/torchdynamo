import contextlib
import logging
import random
import weakref

import torch
from torch import _prims

log = logging.getLogger(__name__)


@contextlib.contextmanager
def patch_functions():
    prior = torch.nn.functional.dropout
    torch.nn.functional.dropout = lowmem_dropout
    yield
    torch.nn.functional.dropout = prior


def replace_fx(gm: torch.fx.GraphModule):
    # Sometimes patch_functions() misses things already in the graph
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_function" and node.target in replacements:
            with gm.graph.inserting_before(node):
                node.replace_all_uses_with(
                    gm.graph.call_function(
                        replacements[node.target], node.args, node.kwargs
                    )
                )
            gm.graph.erase_node(node)
    gm.recompile()
    return gm


def _philox_rand_like_meta(input, seed, offset):
    return _prims.TensorMeta(input)


def _philox_rand_like(input, seed, offset):
    # placeholder only used in tracing
    return torch.rand_like(input)


philox_rand_like = _prims._make_prim(
    schema="philox_rand_like(Tensor input, Tensor seed, int offset) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_philox_rand_like_meta,
    impl_aten=_philox_rand_like,
    doc="",
)


def _philox_seed_like_meta(x):
    return _prims.TensorMeta(_philox_seed_like(x))


def _philox_seed_like(x):
    # we need a tensor input here so AOT autograd properly captures this
    # with just a device input, this becomes a constant
    return torch.tensor(random.randrange(2**31), device=x.device, dtype=torch.int32)


philox_seed_like = _prims._make_prim(
    schema="philox_seed_like(Tensor other) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_philox_seed_like_meta,
    impl_aten=_philox_seed_like,
    doc="",
)


def null_ref():
    return None


class LowmemDropout(torch.autograd.Function):
    next_offset = 0
    seed = {}
    last_tracer_ref = null_ref

    @staticmethod
    def reset(tracer=None):
        LowmemDropout.next_offset = 0
        LowmemDropout.seed = {}
        LowmemDropout.last_tracer_ref = (
            weakref.ref(tracer) if tracer is not None else null_ref
        )

    @staticmethod
    def get_seed_offset(x):
        if isinstance(x, torch.fx.experimental.proxy_tensor.ProxyTensor):
            if LowmemDropout.last_tracer_ref() is not x.proxy.tracer:
                # tracer changed, need to reset state
                LowmemDropout.reset(x.proxy.tracer)
        else:
            # no tracer, need to reset state
            LowmemDropout.reset()

        device = x.device
        if device not in LowmemDropout.seed:
            # Compute the seed just once per trace so we pass fewer
            # things from forward to backward
            LowmemDropout.seed[device] = philox_seed_like(x)

        seed = LowmemDropout.seed[device]
        offset = LowmemDropout.next_offset
        LowmemDropout.next_offset += x.numel()
        return seed, offset

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p
        scale = float(1.0 / (1.0 - p))
        ctx.fallback = x.device.type == "cpu"

        # remove when https://github.com/pytorch/torchdynamo/pull/636 lands
        if ctx.fallback:
            result, mask = torch.ops.aten.native_dropout(x, p, True)
            ctx.save_for_backward(mask)
            return result

        seed, offset = LowmemDropout.get_seed_offset(x)
        ctx.save_for_backward(seed)
        ctx.offset = offset

        bool_mask = philox_rand_like(x, seed, offset) > p
        return bool_mask.to(x.dtype) * x * scale

    @staticmethod
    def backward(ctx, grad_output):
        p = ctx.p
        scale = float(1.0 / (1.0 - p))

        # remove when https://github.com/pytorch/torchdynamo/pull/636 lands
        if ctx.fallback:
            (mask,) = ctx.saved_tensors
            return (
                torch.ops.aten.native_dropout_backward(grad_output, mask, scale),
                None,
            )

        (seed,) = ctx.saved_tensors
        bool_mask = philox_rand_like(grad_output, seed, ctx.offset) > p
        return bool_mask.to(grad_output.dtype) * grad_output * scale, None


@torch.fx.wrap
def lowmem_dropout(input, p, training=True, inplace=False):
    if isinstance(input, torch.fx.Proxy):
        # double check we don't FX trace this
        return input.tracer.create_proxy(
            "call_function",
            lowmem_dropout,
            (input, p, training),
            {},
        )
    if not training:
        return input
    return LowmemDropout.apply(input, p)


replacements = {torch.nn.functional.dropout: lowmem_dropout}
