import dis
import sys
import traceback
import types
import typing

from torch import fx

from torchdynamo import config
from torchdynamo.bytecode_transformation import debug_checks
from torchdynamo.bytecode_transformation import is_generator
from torchdynamo.bytecode_transformation import transform_code_object
from torchdynamo.guards import GuardedCode
from torchdynamo.symbolic_convert import InstructionTranslator
from torchdynamo.symbolic_convert import counters
from torchdynamo.symbolic_convert import unimplemented

TERMINAL_OPCODES = {
    dis.opmap["RETURN_VALUE"],
    dis.opmap["JUMP_ABSOLUTE"],
    dis.opmap["JUMP_FORWARD"],
    # TODO(jansel): need to handle exceptions, etc
}
JUMP_OPCODES = set(dis.hasjrel + dis.hasjabs)


def remove_dead_code(instructions):
    """Dead code elimination"""
    indexof = {id(inst): i for i, inst in enumerate(instructions)}
    live_code = set()

    def find_live_code(start):
        for i in range(start, len(instructions)):
            if i in live_code:
                return
            live_code.add(i)
            inst = instructions[i]
            if inst.opcode in JUMP_OPCODES:
                find_live_code(indexof[id(inst.target)])
            if inst.opcode in TERMINAL_OPCODES:
                return

    find_live_code(0)
    return [inst for i, inst in enumerate(instructions) if i in live_code]


def remove_pointless_jumps(instructions):
    """Eliminate jumps to the next instruction"""
    pointless_jumps = {
        id(a)
        for a, b in zip(instructions, instructions[1:])
        if a.opname == "JUMP_ABSOLUTE" and a.target is b
    }
    return [inst for inst in instructions if id(inst) not in pointless_jumps]


class Tracker:
    def __init__(self):
        self.seen = []
        self.seen_ids = set()

    def add(self, obj):
        if obj not in self:
            self.seen.append(obj)
            self.seen_ids.add(id(obj))

    def __contains__(self, item):
        return id(item) in self.seen_ids

    def clear(self):
        self.seen.clear()
        self.seen_ids.clear()


input_codes = Tracker()
output_codes = Tracker()


def dummy_fx_compile(gm: fx.GraphModule):
    return gm.forward


def convert_frame_assert(compiler_fn: typing.Callable):
    """Fully convert a frame into an FX graph"""

    def _convert_frame_assert(frame: types.FrameType, cache_size: int):
        code = frame.f_code
        input_codes.add(code)
        if code.co_filename.startswith("<eval_with_key>") or code in output_codes:
            return None  # skip FX output
        if is_generator(code):
            unimplemented("generator")
        if cache_size >= config.cache_size_limit:
            unimplemented("cache_size_limit reached")
        debug_checks(code)
        tracer = None

        def transform(instructions, code_options):
            nonlocal tracer
            tracer = InstructionTranslator(
                instructions,
                frame.f_code,
                frame.f_locals,
                frame.f_globals,
                frame.f_builtins,
                code_options,
                compiler_fn,
            )
            tracer.run()
            if tracer.output_instructions and tracer.fully_converted:
                instructions[:] = tracer.output_instructions
            elif tracer.output_instructions:
                instructions[:] = tracer.output_instructions + instructions
            else:
                unimplemented("did not convert frame")

            if config.dead_code_elimination:
                instructions[:] = remove_pointless_jumps(remove_dead_code(instructions))

        code = transform_code_object(frame.f_code, transform)
        output_codes.add(code)
        if config.debug:
            print(
                "\nORIGINAL BYTECODE",
                code.co_name,
                code.co_filename,
                code.co_firstlineno,
            )
            # print(dis.Bytecode(frame.f_code).info())
            print(dis.Bytecode(frame.f_code).dis())
            print("MODIFIED BYTECODE")
            # print(dis.Bytecode(code).info())
            print(dis.Bytecode(code).dis())
            print("\nGUARDS:")
            for guard in sorted(tracer.guards):
                print(" -", str(guard))
            print()
        assert tracer.guards is not None
        return GuardedCode(code, tracer.guards, frame.f_locals, frame.f_globals)

    return _convert_frame_assert


def convert_frame(compiler_fn: typing.Callable):
    """Try to convert a frame into an FX graph, if error leave frame unmodified"""
    inner_convert = convert_frame_assert(compiler_fn)

    def _convert_frame(frame: types.FrameType, cache_size: int):
        counters["frames"]["total"] += 1
        try:
            result = inner_convert(frame, cache_size)
            counters["frames"]["ok"] += 1
            return result
        except NotImplementedError:
            pass
        except Exception:
            sys.stderr.write("=" * 10 + " Stack Trace " + "=" * 10 + "\n")
            traceback.print_exc()
            if config.debug:
                sys.stderr.write(
                    "=" * 10 + " Exception (above) while processing " + "=" * 10 + "\n"
                )
                sys.stderr.write(
                    dis.Bytecode(frame.f_code).info()
                    + "\n"
                    + dis.Bytecode(frame.f_code).dis()
                )
                sys.stderr.write("=" * 10 + " End debug info " + "=" * 10 + "\n")
        return None

    return _convert_frame
