import functools
import logging
import threading

from . import config
from . import convert_frame
from . import skipfiles
from .mutation_guard import install_generation_tagging_new

try:
    from . import _eval_frame
except (ModuleNotFoundError, ImportError) as e:
    raise RuntimeError("run `python setup.py develop` to compile C extensions") from e

set_eval_frame = _eval_frame.set_eval_frame
reset_code = _eval_frame.reset_code
unsupported = _eval_frame.unsupported
skip_code = _eval_frame.skip_code
set_guard_fail_hook = _eval_frame.set_guard_fail_hook
set_guard_error_hook = _eval_frame.set_guard_error_hook


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

        # hooks to properly handle inlining
        if isinstance(self, DisableContext):
            _fn._torchdynamo_disable = True
        else:
            _fn._torchdynamo_inline = fn

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


def optimize(backend, nopython=False):
    """
    The main entrypoint of TorchDynamo.  Do graph capture and call
    backend() to optimize extracted graphs.

    Args:
        backend: One of two things:
            - Either, a function taking a torch.fx.GraphModule and
            example_inputs and returning a python callable that runs the
            graph faster.
            - Or, a string backend name in `torchdynamo.list_backends()`
        nopython: If True, graph breaks will be errors and there will
            be a single whole-program graph.

    Example Usage:

        @torchdynamo.optimize("ofi")
        def toy_example(a, b):
            ...

        or

        with torchdynamo.optimize(my_compiler):
           ...
    """
    if nopython:
        return optimize_assert(backend)
    return _optimize_catch_errors(convert_frame.convert_frame(backend))


def optimize_assert(backend):
    """
    The same as `torchdynamo.optimize(backend, nopython=True)`
    """
    return _optimize_catch_errors(convert_frame.convert_frame_assert(backend))


def run(fn=None):
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        assert callable(fn)
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None):
    """Decorator and context manager to disable TorchDynamo"""
    if fn is not None:
        assert callable(fn)
        return DisableContext()(fn)
    return DisableContext()
