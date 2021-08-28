import dataclasses
import dis
import types
from typing import Any, Optional, List


@dataclasses.dataclass
class Instruction:
    """ A mutable version of dis.Instruction """
    opcode: int
    opname: str
    arg: int
    argval: Any
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False
    target: Optional["Instruction"] = None


def convert_instruction(i: dis.Instruction):
    return Instruction(
        i.opcode,
        i.opname,
        i.arg,
        i.argval,
        i.offset,
        i.starts_line,
        i.is_jump_target,
    )


class _NotProvided:
    pass


def create_instruction(name, arg=None, argval=_NotProvided):
    if argval is _NotProvided:
        argval = arg
    return Instruction(
        opcode=dis.opmap[name],
        opname=name,
        arg=arg,
        argval=argval,
    )


def lnotab_writer(lineno, byteno=0):
    """
    Used to create typing.CodeType.co_lnotab
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    Note this format is changing in Python 3.10 and we will need to update
    """
    lnotab = []

    def update(lineno_new, byteno_new):
        nonlocal byteno, lineno
        byte_offset = byteno_new - byteno
        line_offset = lineno_new - lineno
        byteno += byte_offset
        lineno += line_offset
        assert 0 <= byte_offset < 256 and 0 <= line_offset < 256, "can this be negative or overflow?"
        lnotab.extend((byte_offset, line_offset))

    return lnotab, update


def assemble(instructions: List[dis.Instruction], firstlineno):
    """ Do the opposite of dis.get_instructions() """
    code = []
    lnotab, update_lineno = lnotab_writer(firstlineno)
    for inst in instructions:
        if inst.starts_line is not None:
            update_lineno(inst.starts_line, len(code))
        arg = inst.arg or 0
        assert 0 <= arg < 256, "TODO: handle extended args"
        code.extend((inst.opcode, arg))

    return bytes(code), bytes(lnotab)


def virtualize_jumps(instructions):
    """ Replace jump targets with pointers to make editing easier """
    jump_targets = dict()

    for inst in instructions:
        if inst.is_jump_target:
            jump_targets[inst.offset] = inst

    for inst in instructions:
        if inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
            inst.target = jump_targets[inst.argval]


def devirtualize_jumps(instructions):
    """ Fill in args for virtualized jump target after instructions may have moved """
    for inst in instructions:
        if inst.opcode in dis.hasjabs:
            inst.arg = inst.target.offset
        elif inst.opcode in dis.hasjrel:
            inst.arg = inst.target.offset - inst.offset - instruction_size(inst)
        else:
            continue
        inst.argval = inst.target.offset
        inst.argrepr = f"to {inst.target.offset}"
        inst.target = None


def instruction_size(inst):
    arg = inst.arg or 0
    assert 0 <= arg < 256, "TODO: support EXTENDED_ARG"
    return 2


def check_offsets(instructions):
    offset = 0
    for inst in instructions:
        assert inst.offset == offset
        offset += instruction_size(inst)


def update_offsets(instructions):
    offset = 0
    for inst in instructions:
        inst.offset = offset
        offset += instruction_size(inst)


def debug_bytes(*args):
    index = range(max(map(len, args)))
    result = []
    for arg in [index] + list(args) + [[int(a != b) for a, b in zip(args[-1], args[-2])]]:
        result.append(" ".join(f"{x:03}" for x in arg))

    return "bytes mismatch\n" + "\n".join(result)


def debug_checks(code):
    """ Make sure our assembler produces same bytes as we start with """
    dode = transform_code_object(code, lambda x: None)
    assert code.co_code == dode.co_code, debug_bytes(code.co_code, dode.co_code)
    assert code.co_lnotab == code.co_lnotab, debug_bytes(code.co_lnotab, dode.co_lnotab)


def transform_code_object(code, transformations):
    instructions = list(map(convert_instruction, dis.get_instructions(code)))
    check_offsets(instructions)
    virtualize_jumps(instructions)

    transformations(instructions)

    update_offsets(instructions)
    devirtualize_jumps(instructions)
    bytecode, lnotab = assemble(instructions, code.co_firstlineno)
    return types.CodeType(
        code.co_argcount,
        code.co_posonlyargcount,  # python 3.8+
        code.co_kwonlyargcount,
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


def insert_nops(instructions):
    """ used to debug jump updates """
    instructions.insert(0, create_instruction("NOP"))
    instructions.insert(0, create_instruction("NOP"))