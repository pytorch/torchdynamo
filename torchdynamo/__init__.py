from . import convert_frame
from . import resume_execution
from .eval_frame import disable
from .eval_frame import optimize
from .eval_frame import optimize_assert
from .eval_frame import reset_code
from .eval_frame import run
from .eval_frame import skip
from .guards import guard_failures
from .guards import orig_code_map

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
    orig_code_map.clear()
    guard_failures.clear()
    resume_execution.ContinueExecutionCache.cache.clear()


def list_backends():
    """
    Return valid strings that can be passed to:
        @torchdynamo.optimize(<backend>)
        def foo(...):
           ....
    """
    from .optimizations import BACKENDS

    return [*sorted([*BACKENDS.keys(), "inductor"])]
