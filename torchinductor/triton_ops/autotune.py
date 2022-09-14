import builtins
import hashlib
import json
import logging
import os.path
import time
from typing import List

import triton
from triton import Config
from triton import cdiv
from triton import heuristics
from triton import next_power_of_2
from triton.ops.matmul import get_configs_io_bound
from triton.ops.matmul_perf_model import early_config_prune as mm_early_config_prune

from torchinductor import config
from torchinductor.ir import ReductionHint
from torchinductor.triton_ops.mm_perf_model import estimate_matmul_time
from torchinductor.utils import conditional_product

from .conv_perf_model import early_config_prune as conv_early_config_prune
from .conv_perf_model import estimate_conv_time

log = logging.getLogger(__name__)


class CachingAutotuner(triton.code_gen.Autotuner):
    """
    Simplified version of Triton autotuner that has no invalidation key
    and caches the best config to disk.
    """

    def __init__(self, kernel, arg_names, configs, save_cache_hook):
        super().__init__(kernel, arg_names, configs, [], None, None)
        assert not self.early_config_prune
        assert not self.perf_model
        self.save_cache_hook = save_cache_hook

    def _bench(self, *args, config, **kwargs):
        try:
            return super()._bench(*args, config=config, **kwargs)
        except triton.code_gen.OutOfResources as e:
            log.warning("OutOfResources: %s %s", e, config)
            return (float("inf"), float("inf"), float("inf"))

    def __call__(self, *args, **kwargs):
        if len(self.configs) > 1:
            bench_start = time.time()
            timings = {
                config: self._bench(*args, config=config, **kwargs)
                for config in self.configs
            }
            self.bench_time = time.time() - bench_start
            self.configs = [builtins.min(timings, key=timings.get)]
            if self.save_cache_hook:
                self.save_cache_hook(self.configs[0])

        config = self.configs[0]
        if config.pre_hook is not None:
            config.pre_hook(dict(zip(self.arg_names, args)))
        return self.kernel(
            *args,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            **kwargs,
            **config.kwargs,
        )


def hash_configs(configs: List[Config]):
    """
    Hash used to check for changes in configurations
    """
    hasher = hashlib.sha256()
    for cfg in configs:
        hasher.update(
            f"{sorted(cfg.kwargs.items())} {cfg.num_warps} {cfg.num_stages}\n".encode(
                "utf-8"
            )
        )
    return hasher.hexdigest()


def load_cached_autotuning(
    cache_filename: str, configs_hash: str, configs: List[Config]
):
    """
    Read a cached autotuning result from disk
    """
    if not os.path.exists(cache_filename):
        return None

    best_config = json.loads(open(cache_filename).read())
    if best_config.get("configs_hash") != configs_hash:
        return None

    matching_configs = [
        cfg
        for cfg in configs
        if all(val == best_config.get(key) for key, val in cfg.kwargs.items())
    ]
    if len(matching_configs) != 1:
        return None

    return matching_configs[0]


def cached_autotune(
    configs: List[Config],
    filename=None,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)

    # on disk caching logic
    if filename is not None and len(configs) > 1:
        cache_filename = os.path.splitext(filename)[0] + ".best_config"
        configs_hash = hash_configs(configs)
        best_config = load_cached_autotuning(cache_filename, configs_hash, configs)
        if best_config:
            configs = [best_config]

        def save_cache_hook(cfg):
            with open(cache_filename, "w") as fd:
                fd.write(json.dumps({**cfg.kwargs, "configs_hash": configs_hash}))

    else:
        save_cache_hook = None

    def decorator(fn):
        def wrapper(kernel):
            return CachingAutotuner(
                kernel, fn.arg_names, configs, save_cache_hook=save_cache_hook
            )

        fn.kernel_decorators.append(wrapper)
        return fn

    return decorator


def unique_configs(configs: List[Config]):
    """Remove duplicate configurations"""
    seen = set()
    pruned_configs = []
    for cfg in configs:
        key = tuple(cfg.kwargs.items())
        if key not in seen:
            seen.add(key)
            pruned_configs.append(cfg)
    return pruned_configs


def triton_config(size_hints, x, y=None, z=None, num_stages=1) -> Config:
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


def triton_config_reduction(size_hints, x, r, num_stages=2) -> Config:
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
    num_warps = next_power_of_2(min(max(conditional_product(x, r) // 128, 1), 8))
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
    # Autotuner with 1-config is no-op
    return cached_autotune([config])


def pointwise_heuristics(size_hints, filename=None):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    if len(size_hints) == 1:
        return apply_triton_config(triton_config(size_hints, 1024))
    if len(size_hints) == 2:
        if not config.triton.autotune:
            return apply_triton_config(triton_config(size_hints, 64, 64))
        return cached_autotune(
            [
                triton_config(size_hints, 32, 32),
                triton_config(size_hints, 8, 256),
                triton_config(size_hints, 256, 8),
                triton_config(size_hints, 1, 1024),
                triton_config(size_hints, 1024, 1),
            ],
            filename=filename,
        )
    if len(size_hints) == 3:
        if not config.triton.autotune:
            return apply_triton_config(triton_config(size_hints, 16, 16, 16))
        return cached_autotune(
            [
                triton_config(size_hints, 16, 16, 16),
                triton_config(size_hints, 64, 8, 8),
                triton_config(size_hints, 8, 64, 8),
                triton_config(size_hints, 8, 8, 64),
                triton_config(size_hints, 1024, 1, 1),
                triton_config(size_hints, 1, 1024, 1),
                triton_config(size_hints, 1, 1, 1024),
            ],
            filename=filename,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def reduction_heuristics(size_hints, reduction_hint=False, filename=None):
    """args to @triton.heuristics()"""
    rnumel = size_hints[-1]
    if len(size_hints) == 2:
        contiguous_config = triton_config_reduction(
            size_hints, 1, (rnumel if 256 <= rnumel < 2048 else 2048), num_stages=1
        )
        outer_config = triton_config_reduction(size_hints, 128, 8)
        tiny_config = triton_config_reduction(
            size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, rnumel
        )
        if reduction_hint == ReductionHint.INNER:
            return apply_triton_config(contiguous_config)
        elif reduction_hint == ReductionHint.OUTER:
            return apply_triton_config(outer_config)
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return apply_triton_config(tiny_config)
        if not config.triton.autotune:
            return apply_triton_config(triton_config_reduction(size_hints, 32, 128))
        return cached_autotune(
            [
                triton_config_reduction(size_hints, 64, 64),
                triton_config_reduction(
                    size_hints, 128, 8
                ),  # this one is the best for outer reduction
                triton_config_reduction(
                    size_hints, 8, 512
                ),  # this and the next one seem very similar but both are needed for perf
                contiguous_config,
            ],
            filename=filename,
        )
    """
    # This is not tested yet:
    if len(size_hints) == 3:
        if not config.triton.autotune:
            return apply_triton_config(
                triton_config_tiled_reduction(size_hints, 16, 16, 16)
            )
        return cached_autotune(
            [
                triton_config_tiled_reduction(size_hints, 16, 16, 16),
                triton_config_tiled_reduction(size_hints, 1, 32, 128),
                triton_config_tiled_reduction(size_hints, 32, 1, 128),
                triton_config_tiled_reduction(size_hints, 1, 1, 2048, num_stages=1),
            ],
            filename=filename,
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
        # triton.Config(
        #     {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 64}, num_stages=4, num_warps=2
        # ),
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
        "early_config_prune": conv_early_config_prune,
        "perf_model": estimate_conv_time,
        "top_k": 10,
    }
    return triton.autotune(configs, key, prune_configs_by=prune_configs_by)


def mm_heuristics():
    mm_heuristic = heuristics(
        {
            "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
        }
    )
    return mm_heuristic


def mm_autotune(get_io_bound_configs=False):
    configs = [
        # basic configs for compute-bound matmuls
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
        # good for int8
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
    ]
    if get_io_bound_configs:
        configs += get_configs_io_bound()
    key = ["M", "N", "K"]
    prune_configs_by = {
        "early_config_prune": mm_early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    }
    return triton.autotune(configs, key, prune_configs_by=prune_configs_by)


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
