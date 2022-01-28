import torchdynamo.convert_frame
import torchdynamo.resume_execution

from . import eval_frame


def optimize(fx_compile_fn):
    return eval_frame.optimize(torchdynamo.convert_frame.convert_frame(fx_compile_fn))


def optimize_assert(fx_compile_fn):
    return eval_frame.optimize(
        torchdynamo.convert_frame.convert_frame_assert(fx_compile_fn)
    )


run = eval_frame.run


def reset():
    from torchdynamo._eval_frame import reset_code

    for code in (
        torchdynamo.convert_frame.input_codes.seen
        + torchdynamo.convert_frame.output_codes.seen
    ):
        reset_code(code)
    torchdynamo.convert_frame.input_codes.clear()
    torchdynamo.convert_frame.output_codes.clear()
    torchdynamo.resume_execution.ContinueExecutionCache.cache.clear()
