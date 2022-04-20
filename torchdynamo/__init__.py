import torch

from . import convert_frame
from . import resume_execution
from .eval_frame import disable
from .eval_frame import optimize
from .eval_frame import optimize_assert
from .eval_frame import reset_code
from .eval_frame import run
from .eval_frame import skip

__all__ = [
    "optimize",
    "optimize_assert",
    "run",
    "disable",
    "reset",
    "list_backends",
    "skip",
]


def reset():
    """Clear all compile caches and restore initial state"""
    for code in convert_frame.input_codes.seen + convert_frame.output_codes.seen:
        reset_code(code)
    convert_frame.input_codes.clear()
    convert_frame.output_codes.clear()
    resume_execution.ContinueExecutionCache.cache.clear()


def list_backends():
    """
    Return valid strings that can be passed to:
        @torchdynamo.optimize(<backend>)
        def foo(...):
           ....
    """
    from .optimizations import BACKENDS

    return list(sorted(BACKENDS.keys()))


# Monkey patching autograd.Variable name to fix FX codegen. FX generates a call by roughly doing
# f"{fn.__module__}.{fn.__name__}(...). This yields torch.autograd.variable.Variable(...) in the
# output of an FX graph.  Unfortunately the module name torch.autograd.variable is shadowed by this
# deprecated function, causing the issue facebookresearch/torchdynamo#82.
# A PyTorch PR is already in flight - https://github.com/pytorch/pytorch/pull/76079. We will remove
# this when that PR is merged.
torch.autograd.Variable.__module__ = "torch.autograd"
