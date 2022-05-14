import contextlib
import functools
import logging
import threading
import itertools

from . import config
from . import convert_frame
from . import skipfiles
from .mutation_guard import install_generation_tagging_new
from torchdynamo.utils import checkpoint_params, clone_inputs
#from torchdynamo.testing import same
import torch
from torchdynamo.optimizations.normalize import normalize_ir

log = logging.getLogger(__name__)

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


def same(left, right):
    return len(left) == len(right) and all(
        torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(left, right)
    )


def check_requires_grad(gm, example_inputs):
    if torch.is_grad_enabled():
        if any(
            getattr(x, "requires_grad", False)
            for x in itertools.chain(example_inputs, gm.parameters(True))
        ):
            return True
    return False


def jit_trace(gm, example_inputs):
    """Wrapper around jit.trace to handle hooks"""
    restore_backward_hooks = []

    def visit(mod):
        if mod._backward_hooks:
            restore_backward_hooks.append((mod, mod._backward_hooks))
            mod._backward_hooks = []

    if not check_requires_grad(gm, example_inputs):
        # in inference mode it is safe to ignore backwards hooks to allow tracing
        gm.apply(visit)

    try:
        return torch.jit.trace(gm.forward, example_inputs)
    finally:
        for mod, hooks in restore_backward_hooks:
            mod._backward_hooks = hooks


null_context = contextlib.nullcontext

unset = object()

compile_lock = threading.Lock()


class _TorchDynamoContext:
    def __init__(self, callback, on_enter=nothing, backend_ctx_ctor=null_context):
        super().__init__()
        assert callable(callback) or callback is False or callback is None
        self.callback = callback
        self.prior = unset
        self.on_enter = on_enter
        self.extra_ctx_ctor = backend_ctx_ctor

    def __enter__(self):
        self.on_enter()
        self.prior = set_eval_frame(self.callback)
        self.backend_ctx = self.extra_ctx_ctor()
        self.backend_ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_eval_frame(self.prior)
        self.prior = unset
        self.backend_ctx.__exit__(exc_type, exc_val, exc_tb)

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
    def __init__(self, callback, backend_ctx_ctor):
        super().__init__(
            callback=callback,
            on_enter=install_generation_tagging_new,
            backend_ctx_ctor=backend_ctx_ctor,
        )


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


def _optimize_catch_errors(compile_fn, backend_ctx_ctor=null_context):
    return OptimizeContext(
        catch_errors_wrapper(compile_fn), backend_ctx_ctor=backend_ctx_ctor
    )


class WrapperBackend:

    def __init__(self, backend=None):
        self.backend = backend

    @property
    def example_inputs(self):
        return clone_inputs(self.original_example_inputs)

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):

        self.restore = checkpoint_params(gm)
        self.original_example_inputs = clone_inputs(example_inputs)
        self.gm = normalize_ir(gm, self.original_example_inputs)
        self.scripted = jit_trace(self.gm, self.example_inputs)
        

        if not callable(self.backend):
            print("self.backend is not callable")

        if self.backend is None or self.backend is self.gm.forward:
            return self.gm.forward

        if not config.verify_correctness:
            return self.backend(gm, self.original_example_inputs)

        # if verify_correctness=True
        try:
            self.restore()

            correct = gm.forward(*self.example_inputs)
            result = self.backend(gm, self.original_example_inputs)(*self.example_inputs)

            # TODO: replace `same` function with the one in testing
            if same(correct, result):
                return self.backend(gm, self.original_example_inputs)

            print(f"incorrect results of backend {self}")
            return self.gm.forward

        except Exception:
            log.exception("error in verify_correctness")
            return self.gm.forward
        finally:
            self.restore()


def optimize(backend, nopython=False):
    """
    The main entrypoint of TorchDynamo.  Do graph capture and call
    backend() to optimize extracted graphs.

    Args:
        backend: One of the two things:
            - Either, a function/callable taking a torch.fx.GraphModule and
            example_inputs and returning a python callable that runs the
            graph faster.
            One can also provide additional context for the backend, like
            torch.jit.fuser("fuser2"), by setting the backend_ctx_ctor attribute.
            See AOTAutogradMemoryEfficientFusionWithContext for the usage.
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

    backend_ctx_ctor = null_context
    if hasattr(backend, "backend_ctx_ctor"):
        backend_ctx_ctor = getattr(backend, "backend_ctx_ctor")

    wrapper_backend = WrapperBackend(backend)
    if nopython:
        return optimize_assert(wrapper_backend, backend_ctx_ctor)
    return _optimize_catch_errors(
        convert_frame.convert_frame(wrapper_backend), backend_ctx_ctor
    )


def optimize_assert(backend, backend_ctx_ctor=null_context):
    """
    The same as `torchdynamo.optimize(backend, nopython=True)`
    """
    return _optimize_catch_errors(
        convert_frame.convert_frame_assert(backend), backend_ctx_ctor
    )


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


def skip(fn=None):
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """
    if fn is None:
        return skip
    assert callable(fn)
    skip_code(fn.__code__)
    fn._torchdynamo_disable = True
    return fn
