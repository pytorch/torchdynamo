import functools

from ptdynamo import eval_frame

import contextlib
import logging
import sys
import inspect
import operator
from typing import Any, Dict, List
import dis
# import xdis.std as dis
# import xdis
import types

eval_frame.get_skip_files().update({
    __file__,
    contextlib.__file__,
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
    def __init__(self, frame: "PyFrameObject"):
        super().__init__()
        self.stack: List[BoxedValue] = list()
        self.frame = frame
        self.f_locals: Dict[str, BoxedValue] = {k: BoxedValue(v) for k, v in frame.f_locals.items()}

        # print(eval_frame.get_locals_array(frame))
        self.f_globals: Dict[str, Any] = frame.f_globals
        self.f_builtins: Dict[str, Any] = frame.f_builtins
        self.instructions = list(dis.get_instructions(self.frame.f_code))
        self.jump_targets = {
            inst.offset: idx for idx, inst in enumerate(self.instructions)
            if inst.is_jump_target
        }
        self.instruction_pointer = 0

    def popn(self, n):
        return reversed([self.stack.pop() for _ in range(n)])

    def run(self):
        # print(self.frame.f_code.co_code)
        # print(self.frame.f_code.co_lnotab)
        print(dis.Bytecode(self.frame.f_code).info())
        print(dis.Bytecode(self.frame.f_code).dis())
        for i in dis.Bytecode(self.frame.f_code):
            print(i)
        return EXECUTE_SENTINEL

        while True:
            inst = self.instructions[self.instruction_pointer]
            if inst.opname == "RETURN_VALUE":
                return self.stack.pop()

            self.instruction_pointer += 1
            getattr(self, inst.opname)(inst.argval)

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


def interpret_frame(frame):
    print(f"interpret_frame {frame.f_code.co_filename}")
    return FrameInterpreter(frame).run()


def lnotab_writer(lineno, byteno=0):
    lnotab = []

    def update(lineno_new, byteno_new):
        nonlocal byteno, lineno
        byte_offset = byteno_new - byteno
        line_offset = lineno_new - lineno
        byteno += byte_offset
        lineno += line_offset
        assert 0 <= byte_offset < 256 and 0 <= line_offset < 256
        lnotab.extend((byte_offset, line_offset))

    return lnotab, update


def assemble(instructions: List[dis.Instruction], firstlineno):
    code = []
    lnotab, update_lineno = lnotab_writer(firstlineno)
    for inst in instructions:
        if inst.starts_line is not None:
            update_lineno(inst.starts_line, len(code))
        arg = inst.arg or 0
        assert 0 <= arg < 256, "TODO: handle extended args"
        code.extend((inst.opcode, arg))

    return bytes(code), bytes(lnotab)


def translate_frame(frame):
    code = frame.f_code
    if frame.f_lasti:
        return code  # already running?

    instructions = list(dis.Bytecode(code))
    bytecode, lnotab = assemble(instructions, code.co_firstlineno)
    assert code.co_code == bytecode
    assert code.co_lnotab == lnotab
    return types.CodeType(
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_posonlyargcount,  # python 3.8+
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        bytecode,
        code.co_consts,
        code.co_names,
        code.co_varnames,
        code.co_filename,
        code.co_name,
        code.co_firstlineno,
        lnotab,
        code.co_freevars,
        code.co_cellvars,
    )


@contextlib.contextmanager
def activate(callback=translate_frame):
    def catch_errors(frame):
        try:
            return callback(frame)
        except Exception:
            eval_frame.set_eval_frame(None)
            bc = dis.Bytecode(frame.f_code)
            log.exception(f"Error while processing frame:\n{bc.info()}\n{bc.dis()}")
            eval_frame.abort()
            raise

    prior = eval_frame.set_eval_frame(catch_errors)
    yield
    eval_frame.set_eval_frame(prior)
