import dis
import functools
import logging

from .eval_frame import set_eval_frame
from .bytecode_transformation import debug_checks, transform_code_object, insert_nops

"""
class BoxedValue:
    def __init__(self, value):
        self.value = value


def stack_op(fn):
    nargs = len(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def impl(self, val):
        self.stack.append(fn(*self.popn(nargs)))

    return impl


def load_op(fn):
    @functools.wraps(fn)
    def impl(self, val):
        self.stack.append(fn(self, val))

    return impl


class FrameInterpreter:
    def LOAD_FAST(self, val):
        self.stack.append(self.f_locals[val].value)

    def LOAD_CLOSURE(self, val):
        self.stack.append(self.f_locals[val].value)

    def LOAD_FAST(self, val):
        self.stack.append(self.f_locals[val].value)

    def STORE_FAST(self, val):
        self.f_locals[val] = BoxedValue(self.stack.pop())

    def LOAD_GLOBAL(self, val):
        self.stack.append(self.f_globals[val])

    def STORE_DEREF(self, val):
        self.f_locals[val] = BoxedValue(CellType(self.stack.pop()))

    BINARY_POWER = stack_op(lambda tos1, tos: tos1 ** tos)
    BINARY_MULTIPLY = stack_op(lambda tos1, tos: tos1 * tos)
    BINARY_MATRIX_MULTIPLY = stack_op(lambda tos1, tos: tos1 @ tos)
    BINARY_FLOOR_DIVIDE = stack_op(lambda tos1, tos: tos1 // tos)
    BINARY_TRUE_DIVIDE = stack_op(lambda tos1, tos: tos1 / tos)
    BINARY_MODULO = stack_op(lambda tos1, tos: tos1 % tos)
    BINARY_ADD = stack_op(lambda tos1, tos: tos1 + tos)
    BINARY_SUBTRACT = stack_op(lambda tos1, tos: tos1 - tos)
    BINARY_SUBSCR = stack_op(lambda tos1, tos: tos1[tos])
    BINARY_LSHIFT = stack_op(lambda tos1, tos: tos1 << tos)
    BINARY_RSHIFT = stack_op(lambda tos1, tos: tos1 >> tos)
    BINARY_AND = stack_op(lambda tos1, tos: tos1 & tos)
    BINARY_XOR = stack_op(lambda tos1, tos: tos1 ^ tos)
    BINARY_OR = stack_op(lambda tos1, tos: tos1 | tos)
    LOAD_CONST = load_op(lambda self, val: val)
"""


def debug_insert_nops(frame):
    code = frame.f_code
    if frame.f_lasti >= 0:
        return code  # already running?
    if code.co_filename == __file__:
        return code  # dont touch ourself
    debug_checks(code)
    return transform_code_object(code, insert_nops)


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
            return callback(frame)
        except Exception:
            logging.basicConfig()
            bc = dis.Bytecode(frame.f_code)
            logging.exception(f"Error while processing frame:\n{bc.info()}\n{bc.dis()}")
            raise

    return _Context(catch_errors)
