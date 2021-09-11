from . import eval_frame
from . import symbolic_convert


def optimize(fx_compile_fn):
    return eval_frame.optimize(symbolic_convert.convert_frame(fx_compile_fn))


run = eval_frame.run
