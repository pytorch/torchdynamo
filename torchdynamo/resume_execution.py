from typing import Any
from typing import Dict
from typing import List

from .bytecode_transformation import create_instruction
from .bytecode_transformation import Instruction
from .bytecode_transformation import transform_code_object
from .utils import ExactWeakKeyDictionary

# taken from code.h in cpython
CO_OPTIMIZED = 0x0001
CO_NEWLOCALS = 0x0002
CO_VARARGS = 0x0004
CO_VARKEYWORDS = 0x0008
CO_NESTED = 0x0010
CO_GENERATOR = 0x0020
CO_NOFREE = 0x0040
CO_COROUTINE = 0x0080
CO_ITERABLE_COROUTINE = 0x0100
CO_ASYNC_GENERATOR = 0x0200


class ContinueExecutionCache:
    cache = ExactWeakKeyDictionary()

    @classmethod
    def lookup(cls, code, *key):
        if code not in cls.cache:
            cls.cache[code] = dict()
        key = tuple(key)
        if key not in cls.cache[code]:
            cls.cache[code][key] = cls.generate(code, *key)
        return cls.cache[code][key]

    @classmethod
    def generate(
        cls,
        code,
        offset: int,
        nstack: int,
        argnames: List[str],
    ):
        assert not (
            code.co_flags
            & (CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR)
        )
        assert code.co_flags & CO_OPTIMIZED

        def update(instructions: List[Instruction], code_options: Dict[str, Any]):
            args = [f"___stack{i}" for i in range(nstack)]
            args.extend(v for v in argnames if v not in args)
            freevars = tuple(code_options["co_cellvars"] or []) + tuple(
                code_options["co_freevars"] or []
            )
            code_options["co_cellvars"] = tuple()
            code_options["co_freevars"] = freevars
            code_options["co_argcount"] = len(args)
            code_options["co_posonlyargcount"] = 0
            code_options["co_kwonlyargcount"] = 0
            code_options["co_varnames"] = tuple(
                args + [v for v in code_options["co_varnames"] if v not in args]
            )
            code_options["co_flags"] = code_options["co_flags"] & ~(
                CO_VARARGS | CO_VARKEYWORDS
            )
            (target,) = [i for i in instructions if i.offset == offset]
            prefix = [
                create_instruction("LOAD_FAST", f"___stack{i}") for i in range(nstack)
            ]
            prefix.append(create_instruction("JUMP_ABSOLUTE", target=target))
            # TODO(jansel): add dead code elimination here
            instructions[:] = prefix + instructions

        return transform_code_object(code, update)


"""
# partially finished support for with statements

def convert_locals_to_cells(
        instructions: List[Instruction],
        code_options: Dict[str, Any]):

    code_options["co_cellvars"] = tuple(
        var
        for var in code_options["co_varnames"]
        if var not in code_options["co_freevars"]
        and not var.startswith("___stack")
    )
    cell_and_free = code_options["co_cellvars"] + code_options["co_freevars"]
    for inst in instructions:
        if str(inst.argval).startswith("___stack"):
            continue
        elif inst.opname == "LOAD_FAST":
            inst.opname = "LOAD_DEREF"
        elif inst.opname == "STORE_FAST":
            inst.opname = "STORE_DEREF"
        elif inst.opname == "DELETE_FAST":
            inst.opname = "DELETE_DEREF"
        else:
            continue
        inst.opcode = dis.opmap[inst.opname]
        assert inst.argval in cell_and_free, inst.argval
        inst.arg = cell_and_free.index(inst.argval)

def patch_setup_with(
    instructions: List[Instruction],
    code_options: Dict[str, Any]
):
    nonlocal need_skip
    need_skip = True
    target_index = [
        idx for idx, i in enumerate(instructions) if i.offset == offset
    ][0]
    assert instructions[target_index].opname == "SETUP_WITH"
    convert_locals_to_cells(instructions, code_options)

    stack_depth_before = nstack + stack_effect(instructions[target_index].opcode,
                                               instructions[target_index].arg)

    inside_with = []
    inside_with_resume_at = None
    stack_depth = stack_depth_before
    idx = target_index + 1
    for idx in range(idx, len(instructions)):
        inst = instructions[idx]
        if inst.opname == "BEGIN_FINALLY":
            inside_with_resume_at = inst
            break
        elif inst.target is not None:
            unimplemented("jump from with not supported")
        elif inst.opname in ("BEGIN_FINALLY", "WITH_CLEANUP_START", "WITH_CLEANUP_FINISH", "END_FINALLY",
                             "POP_FINALLY", "POP_EXCEPT",
                             "POP_BLOCK", "END_ASYNC_FOR"):
            unimplemented("block ops not supported")
        inside_with.append(inst)
        stack_depth += stack_effect(inst.opcode, inst.arg)
    assert inside_with_resume_at

    instructions = [
        create_instruction("LOAD_FAST", f"___stack{i}") for i in range(nstack)
    ] + [
        create_instruction("SETUP_WITH", target=instructions[target_index].target)
        ... call the function ...
        unpack_tuple
    ] + [
        create_instruction("JUMP_ABSOLUTE", target=inside_with_resume_at)
    ]
"""
