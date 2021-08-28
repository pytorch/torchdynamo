import contextlib
import dis
import logging

from . import eval_frame
from .bytecode_transformation import debug_checks, transform_code_object, insert_nops

log = logging.getLogger(__name__)

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


def transform_frame(frame):
    code = frame.f_code
    if frame.f_lasti >= 0:
        return code  # already running?
    if code.co_filename == __file__:
        return code  # dont touch ourself
    debug_checks(code)
    return transform_code_object(code, insert_nops)


@contextlib.contextmanager
def context(callback=transform_frame):
    def catch_errors(frame):
        try:
            return callback(frame)
        except Exception:
            eval_frame.set_eval_frame(None)
            bc = dis.Bytecode(frame.f_code)
            log.exception(f"Error while processing frame:\n{bc.info()}\n{bc.dis()}")
            raise

    prior = eval_frame.set_eval_frame(catch_errors)
    yield
    eval_frame.set_eval_frame(prior)
