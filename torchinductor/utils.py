import functools
import logging.config
import operator

import sympy
import torch

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "simple": {"format": "%(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "torchinductor": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}


@functools.lru_cache(None)
def init_logging():
    logging.config.dictConfig(LOGGING_CONFIG)


@functools.lru_cache(None)
def has_triton():
    try:
        import triton

        return triton is not None
    except (ImportError, ModuleNotFoundError):
        return False


@functools.lru_cache(None)
def has_torchvision_roi_align():
    try:
        from torchvision.ops import roi_align  # noqa

        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except (ImportError, ModuleNotFoundError):
        return False


def conditional_product(*args):
    return functools.reduce(operator.mul, [x for x in args if x])


def sympy_product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


def unique(it):
    return {id(x): x for x in it}.values()
