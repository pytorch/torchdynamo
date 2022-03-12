import functools
import logging
import threading

from . import config
from . import convert_frame
from . import skipfiles
from ._eval_frame import set_eval_frame
from .mutation_guard import install_generation_tagging_new


def nothing():
    pass


unset = object()

compile_lock = threading.Lock()


class _TorchDynamoContext:
    def __init__(self, callback, on_enter=nothing):
        super().__init__()
        assert callable(callback) or callback is False or callback is None
        self.callback = callback
        self.prior = unset
        self.on_enter = on_enter

    def __enter__(self):
        self.on_enter()
        self.prior = set_eval_frame(self.callback)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_eval_frame(self.prior)
        self.prior = unset

    def __call__(self, fn):
        assert callable(fn)
        callback = self.callback
        on_enter = self.on_enter

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            on_enter()
            prior = set_eval_frame(callback)
            try:
                return fn(*args, **kwargs)
            finally:
                set_eval_frame(prior)

        return _fn


class OptimizeContext(_TorchDynamoContext):
    def __init__(self, callback):
        super().__init__(callback=callback, on_enter=install_generation_tagging_new)


class RunOnlyContext(_TorchDynamoContext):
    def __init__(self):
        super().__init__(callback=False)


class DisableContext(_TorchDynamoContext):
    def __init__(self):
        super().__init__(callback=None)


def catch_errors_wrapper(callback):
    @functools.wraps(callback)
    def catch_errors(frame, cache_size):
        try:
            if frame.f_lasti >= 0 or skipfiles.check(frame.f_code.co_filename):
                if config.debug:
                    print(f"skipping {frame.f_code.co_name} {frame.f_code.co_filename}")
                return None
            if (
                frame.f_code.co_filename == "<string>"
                and frame.f_code.co_name == "__new__"
            ):
                # nametuple constructor
                return None
            with compile_lock:
                return callback(frame, cache_size)
        except Exception:
            logging.basicConfig()
            logging.exception("Error while processing frame")
            raise

    return catch_errors


def _optimize_catch_errors(compile_fn):
    return OptimizeContext(catch_errors_wrapper(compile_fn))


def optimize(fx_compile_fn, nopython=False):
    if nopython:
        return optimize_assert(fx_compile_fn)
    return _optimize_catch_errors(convert_frame.convert_frame(fx_compile_fn))


def optimize_assert(fx_compile_fn):
    return _optimize_catch_errors(convert_frame.convert_frame_assert(fx_compile_fn))


def run(fn=None):
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        assert callable(fn)
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None):
    """context manager to disable TorchDynamo"""
    if fn is not None:
        assert callable(fn)
        return DisableContext()(fn)
    return DisableContext()
