import functools

from ptdynamo import eval_frame

import contextlib
import dis
import logging
import sys
import inspect
import operator
from typing import Any, Dict, List

EXECUTE_SENTINEL = eval_frame.get_execute_sentinel()
eval_frame.get_skip_files().update({
    __file__,
    contextlib.__file__,
    dis.__file__,
})
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""

class FrameInterpreter:
 #   def __init__(self, frame: "PyFrameObject"):
 #      super().__init__()
 #      self.f_locals: typing.Dict[str, BoxedValue] = {k: BoxedValue(v) for k, v in frame.f_locals}
 #      self.f_globals: typing.Dict[str, Any] = frame.f_globals
 #      self.f_builtins: typing.Dict[str, Any] = frame.f_builtins

    def run(self):
#      instructions = list(dis.get_instructions())
#      for inst in instructions:
#          print(inst)
#
        return EXECUTE_SENTINEL


"""


class BoxedValue:
    def __init__(self, value):
        self.value = value


def simple_op(fn):
    nargs = len(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def impl(self, val):
        args = [self.stack.pop() for _ in range(nargs)]
        self.stack.append(fn(*reversed(args)))

    return impl


class FrameInterpreter:
    def __init__(self, frame: "PyFrameObject"):
        super().__init__()
        self.stack: List[BoxedValue] = list()
        self.frame = frame
        self.f_locals: Dict[str, BoxedValue] = {k: BoxedValue(v) for k, v in frame.f_locals.items()}
        self.f_globals: Dict[str, Any] = frame.f_globals
        self.f_builtins: Dict[str, Any] = frame.f_builtins
        self.instructions = list(dis.get_instructions(self.frame.f_code))
        self.jump_targets = {
            inst.offset: idx for idx, inst in enumerate(self.instructions)
            if inst.is_jump_target
        }
        self.instruction_pointer = 0

    def run(self):
        while True:
            inst = self.instructions[self.instruction_pointer]
            if inst.opname == "RETURN_VALUE":
                return self.stack.pop()

            self.instruction_pointer += 1
            getattr(self, inst.opname)(inst.argval)

    def LOAD_FAST(self, val):
        self.stack.append(self.f_locals[val].value)

    def LOAD_GLOBAL(self, val):
        self.stack.append(self.f_globals[val])

    BINARY_POWER = simple_op(lambda tos1, tos: tos1 ** tos)
    BINARY_MULTIPLY = simple_op(lambda tos1, tos: tos1 * tos)
    BINARY_MATRIX_MULTIPLY = simple_op(lambda tos1, tos: tos1 @ tos)
    BINARY_FLOOR_DIVIDE = simple_op(lambda tos1, tos: tos1 // tos)
    BINARY_TRUE_DIVIDE = simple_op(lambda tos1, tos: tos1 / tos)
    BINARY_MODULO = simple_op(lambda tos1, tos: tos1 % tos)
    BINARY_ADD = simple_op(lambda tos1, tos: tos1 + tos)
    BINARY_SUBTRACT = simple_op(lambda tos1, tos: tos1 - tos)
    BINARY_SUBSCR = simple_op(lambda tos1, tos: tos1[tos])
    BINARY_LSHIFT = simple_op(lambda tos1, tos: tos1 << tos)
    BINARY_RSHIFT = simple_op(lambda tos1, tos: tos1 >> tos)
    BINARY_AND = simple_op(lambda tos1, tos: tos1 & tos)
    BINARY_XOR = simple_op(lambda tos1, tos: tos1 ^ tos)
    BINARY_OR = simple_op(lambda tos1, tos: tos1 | tos)


def interpret_frame(frame):
    print(f"interpret_frame {frame.f_code.co_filename}")
    return FrameInterpreter(frame).run()


@contextlib.contextmanager
def activate(callback=interpret_frame):
    def catch_errors(frame):
        try:
            return callback(frame)
        except Exception:
            eval_frame.set_eval_frame(None)
            bc = dis.Bytecode(frame.f_code)
            log.exception(f"Error while processing frame:\n{bc.info()}\n{bc.dis()}")
            sys.exit(-1)
            raise

    prior = eval_frame.set_eval_frame(catch_errors)
    yield
    eval_frame.set_eval_frame(prior)
