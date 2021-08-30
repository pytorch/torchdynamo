import dis
import functools
import logging

from . import skipfiles
from .bytecode_transformation import debug_checks, transform_code_object, insert_nops
from .eval_frame import set_eval_frame


def debug_insert_nops(frame):
    debug_checks(frame.f_code)
    return transform_code_object(frame.f_code, insert_nops)


default_callback = debug_insert_nops


class _Context:
    def __init__(self, callback):
        assert callable(callback)
        self.callback = callback
        self.prior = None

    def __enter__(self):
        assert self.prior is None
        self.prior = set_eval_frame(self.callback)

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


def context(callback=default_callback):
    def catch_errors(frame):
        try:
            if frame.f_lasti >= 0 or skipfiles.check(frame.f_code.co_filename):
                return frame.f_code
            return callback(frame)
        except Exception:
            logging.basicConfig()
            bc = dis.Bytecode(frame.f_code)
            for i in bc:
                print(i)
            logging.exception(f"Error while processing frame:\n{bc.info()}\n{bc.dis()}")
            raise

    return _Context(catch_errors)
