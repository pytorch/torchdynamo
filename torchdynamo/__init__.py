from . import convert_frame
from . import resume_execution
from .eval_frame import disable
from .eval_frame import optimize
from .eval_frame import optimize_assert
from .eval_frame import reset_code
from .eval_frame import run

__all__ = [
    "optimize",
    "optimize_assert",
    "run",
    "disable",
    "reset",
]


def reset():
    """Clear all compile caches and restore initial state"""
    for code in convert_frame.input_codes.seen + convert_frame.output_codes.seen:
        reset_code(code)
    convert_frame.input_codes.clear()
    convert_frame.output_codes.clear()
    resume_execution.ContinueExecutionCache.cache.clear()
