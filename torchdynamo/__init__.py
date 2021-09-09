from . import eval_frame
from . import symbolic_convert


def context(fx_compile_fn):
    return eval_frame.context(symbolic_convert.convert_frame(fx_compile_fn))
