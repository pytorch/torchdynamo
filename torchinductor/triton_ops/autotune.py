import logging

import triton
from triton import Config
from triton import cdiv
from triton import heuristics
from triton import next_power_of_2

from torchinductor import config
from torchinductor.utils import conditional_product

from .conv_perf_model import early_config_prune
from .conv_perf_model import estimate_conv_time

log = logging.getLogger(__name__)


class Autotuner(triton.code_gen.Autotuner):
    """
    Customized triton autotuner
    """

    def _bench(self, *args, config, **kwargs):
        try:
            return super()._bench(*args, config=config, **kwargs)
        except triton.code_gen.OutOfResources as e:
            log.warning("OutOfResources: %s %s", e, config)
            return (float("inf"), float("inf"), float("inf"))


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None):
    """
    A copy of triton.autotune that calls our subclass above
    """

    def decorator(fn):
        def wrapper(kernel):
            return Autotuner(
                kernel, fn.arg_names, configs, key, reset_to_zero, prune_configs_by
            )

        fn.kernel_decorators.append(wrapper)
        return fn

    return decorator


def triton_config(size_hints, x, y=None, z=None, num_stages=1):
    """
    Construct a pointwise triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """
    # Ideally we want to read this from some device config
    maxGridSize = [2147483647, 65535, 65535]

    target = conditional_product(x, y, z)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    if y:
        y = min(y, size_hints[1])
    if z:
        z = min(z, size_hints[2])

    # if we are below original block size, scale up where we can;
    # or if the calculated grid size is larger than the limit, we bump up the corresponding dimension
    while x < size_hints[0] and (
        x * maxGridSize[0] < size_hints[0] or conditional_product(x, y, z) < target
    ):
        x *= 2
    while (
        y
        and y < size_hints[1]
        and (
            y * maxGridSize[1] < size_hints[1] or conditional_product(x, y, z) < target
        )
    ):
        y *= 2
    while (
        z
        and z < size_hints[2]
        and (
            z * maxGridSize[2] < size_hints[2] or conditional_product(x, y, z) < target
        )
    ):
        z *= 2

    cfg = {"XBLOCK": x}
    if y:
        cfg["YBLOCK"] = y
    if z:
        cfg["ZBLOCK"] = z
    num_warps = next_power_of_2(min(max(conditional_product(x, y, z) // 256, 1), 8))
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def triton_config_reduction(size_hints, x, r, num_stages=2):
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """

    target = conditional_product(x, r)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    r = min(r, size_hints[1])

    # if we are below original block size, scale up where we can
    while x < size_hints[0] and conditional_product(x, r) < target:
        x *= 2
    while r < size_hints[1] and conditional_product(x, r) < target:
        r *= 2

    cfg = {"XBLOCK": x, "RBLOCK": r}
    num_warps = next_power_of_2(min(max(conditional_product(x, r) // 256, 1), 8))
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def triton_config_tiled_reduction(size_hints, x, y, r, num_stages=2):
    """
    Construct a tile reduction triton config with some adjustment
    heuristics based on size_hints. Size_hints is a tuple of numels in
    each tile dimension and will be rounded up to the nearest power of 2.
    """

    target = conditional_product(x, y, r)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    y = min(y, size_hints[1])
    r = min(r, size_hints[2])

    # if we are below original block size, scale up where we can
    while x < size_hints[0] and conditional_product(x, y, r) < target:
        x *= 2
    while r < size_hints[2] and conditional_product(x, y, r) < target:
        r *= 2
    while y < size_hints[1] and conditional_product(x, y, r) < target:
        y *= 2

    cfg = {"XBLOCK": x, "YBLOCK": y, "RBLOCK": r}
    num_warps = next_power_of_2(min(max(conditional_product(x, y, r) // 256, 1), 8))
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def apply_triton_config(config):
    """
    Decorator that applies a fixed triton config using triton.heuristics.
    """

    def getter(name):
        def get(args):
            return config.kwargs[name]

        return get

    return heuristics({name: getter(name) for name in config.kwargs.keys()})


def pointwise_heuristics(size_hints):
    """
    Construct @triton.heuristics() based on size_hints.
    """

    if len(size_hints) == 1:
        return apply_triton_config(triton_config(size_hints, 1024))
    if len(size_hints) == 2:
        if not config.triton.autotune:
            return apply_triton_config(triton_config(size_hints, 64, 64))
        return autotune(
            [
                triton_config(size_hints, 32, 32),
                triton_config(size_hints, 8, 256),
                triton_config(size_hints, 256, 8),
                triton_config(size_hints, 1, 1024),
                triton_config(size_hints, 1024, 1),
            ],
            key=["xnumel", "ynumel"],
        )
    if len(size_hints) == 3:
        if not config.triton.autotune:
            return apply_triton_config(triton_config(size_hints, 16, 16, 16))
        return autotune(
            [
                triton_config(size_hints, 16, 16, 16),
                triton_config(size_hints, 64, 8, 8),
                triton_config(size_hints, 8, 64, 8),
                triton_config(size_hints, 8, 8, 64),
                triton_config(size_hints, 1024, 1, 1),
                triton_config(size_hints, 1, 1024, 1),
                triton_config(size_hints, 1, 1, 1024),
            ],
            key=["xnumel", "ynumel", "znumel"],
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def reduction_heuristics(size_hints):
    """args to @triton.heuristics()"""

    if len(size_hints) == 2:
        if not config.triton.autotune:
            return apply_triton_config(triton_config_reduction(size_hints, 32, 128))
        return autotune(
            [
                triton_config_reduction(size_hints, 64, 64),
                triton_config_reduction(size_hints, 8, 512),
                triton_config_reduction(size_hints, 1, 2048, num_stages=1),
            ],
            key=["xnumel", "rnumel"],
        )
    """
    # This is not tested yet:
    if len(size_hints) == 3:
        if not config.triton.autotune:
            return apply_triton_config(
                triton_config_tiled_reduction(size_hints, 16, 16, 16)
            )
        return autotune(
            [
                triton_config_tiled_reduction(size_hints, 16, 16, 16),
                triton_config_tiled_reduction(size_hints, 1, 32, 128),
                triton_config_tiled_reduction(size_hints, 32, 1, 128),
                triton_config_tiled_reduction(size_hints, 1, 1, 2048, num_stages=1),
            ],
            key=["xnumel", "ynumel", "rnumel"],
        )
    """
    raise NotImplementedError(f"size_hints: {size_hints}")


def conv_heuristics():
    configs = [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 16, "BLOCK_K": 32}, num_stages=4, num_warps=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=4, num_warps=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 64}, num_stages=4, num_warps=2
        ),
    ]
    key = [
        "BATCH",
        "IN_C",
        "IN_H",
        "IN_W",
        "KERNEL_N",
        "KERNEL_H",
        "KERNEL_W",
        "OUT_H",
        "OUT_W",
        # parameters of conv
        "stride_h",
        "stride_w",
        "padding_h",
        "padding_w",
        "dilation_h",
        "dilation_w",
        "output_padding_h",
        "output_padding_w",
        "groups",
    ]
    prune_configs_by = {
        "early_config_prune": early_config_prune,
        "perf_model": estimate_conv_time,
        "top_k": 10,
    }
    return autotune(configs, key, prune_configs_by=prune_configs_by)


def grid(xnumel, ynumel=None, znumel=None):
    """Helper function to compute triton grids"""

    def grid_fn(meta):
        result = [cdiv(xnumel, meta["XBLOCK"])]
        if ynumel:
            result.append(cdiv(ynumel, meta["YBLOCK"]))
            if znumel:
                result.append(cdiv(znumel, meta["ZBLOCK"]))
        return result

    return grid_fn
