import functools
import logging

from . import skipfiles
from ._eval_frame import set_eval_frame
from ._eval_frame import set_eval_frame_run_only


class _Context:
    def __init__(self):
        self.prior = None

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_eval_frame(self.prior)
        self.prior = None

    def __call__(self, fn):
        assert callable(fn)

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)

        return _fn


class _OptimizeContext(_Context):
    def __init__(self, callback):
        super(_OptimizeContext, self).__init__()
        assert callable(callback)
        self.callback = callback

    def __enter__(self):
        assert self.prior is None
        self.prior = set_eval_frame(self.callback)


class _RunContext(_Context):
    def __enter__(self):
        assert self.prior is None
        self.prior = set_eval_frame_run_only()


def catch_errors_wrapper(callback):
    @functools.wraps(callback)
    def catch_errors(frame, cache_size):
        try:
            if frame.f_lasti >= 0 or skipfiles.check(frame.f_code.co_filename):
                return None
            return callback(frame, cache_size)
        except Exception:
            logging.basicConfig()
            logging.exception("Error while processing frame")
            raise

    return catch_errors


def optimize(compile_fn):
    return _OptimizeContext(catch_errors_wrapper(compile_fn))


def run():
    """Don't do any dynamic compiles, just use prior optimizations"""
    return _RunContext()
