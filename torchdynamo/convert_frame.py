import dis
import logging
import pprint
import types
import typing

from torch import fx

import torchdynamo
from torchdynamo.bytecode_transformation import debug_checks
from torchdynamo.bytecode_transformation import transform_code_object
from torchdynamo.guards import GuardedCode
from torchdynamo.symbolic_convert import InstructionTranslator
from torchdynamo.symbolic_convert import counters


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

    def _convert_frame_assert(frame: types.FrameType):
        code = frame.f_code
        input_codes.add(code)
        if code.co_filename.startswith("<eval_with_key>") or code in output_codes:
            return None  # skip FX output
        debug_checks(code)
        tracer = None

        def transform(instructions, code_options):
            nonlocal tracer
            tracer = InstructionTranslator(
                instructions,
                frame.f_locals,
                frame.f_globals,
                frame.f_builtins,
                code_options,
                compiler_fn,
            )
            tracer.run()

        code = transform_code_object(frame.f_code, transform)
        output_codes.add(code)
        if torchdynamo.DEBUG:
            print("\nORIGINAL")
            # print(dis.Bytecode(frame.f_code).info())
            print(dis.Bytecode(frame.f_code).dis())
            print("\nNEW CODE")
            # print(dis.Bytecode(code).info())
            print(dis.Bytecode(code).dis())

            tracer.graph.print_tabular()
            print()
            pprint.pprint(tracer.guards)
        assert tracer.guards is not None
        return GuardedCode(code, tracer.guards, frame.f_locals, frame.f_globals)

    return _convert_frame_assert


def convert_frame(compiler_fn: typing.Callable):
    """Try to convert a frame into an FX graph, if error leave frame unmodified"""
    inner_convert = convert_frame_assert(compiler_fn)

    def _convert_frame(frame: types.FrameType):
        counters["frames"]["total"] += 1
        try:
            result = inner_convert(frame)
            counters["frames"]["ok"] += 1
            return result
        except NotImplementedError:
            pass
        except Exception:
            logging.exception(f"ERROR\n{dis.Bytecode(frame.f_code).dis()}")
        return None

    return _convert_frame
