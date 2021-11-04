from typing import Any
from typing import Dict
from typing import List

from torchdynamo.bytecode_transformation import Instruction
from torchdynamo.bytecode_transformation import create_instruction
from torchdynamo.bytecode_transformation import transform_code_object
from torchdynamo.utils import ExactWeakKeyDictionary

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
            args = [f"___stack{i}" for i in range(nstack)] + list(argnames)
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
