import dis
import functools
import inspect
import os
import sys
import traceback
import types
import typing
from typing import Callable
from typing import List

import torch
from torch import fx

from . import config
from .bytecode_analysis import remove_dead_code
from .bytecode_analysis import remove_pointless_jumps
from .bytecode_transformation import is_generator
from .bytecode_transformation import transform_code_object
from .guards import GuardedCode
from .symbolic_convert import InstructionTranslator
from .utils import CleanupManager
from .utils import Unsupported
from .utils import counters
from .utils import unimplemented


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


def wrap_compiler_fn(compiler_fn):
    """Shim to convert 1 arg to 2 arg compiler_fn"""

    @functools.wraps(compiler_fn)
    def inner(gm: fx.GraphModule, example_inputs: List):
        return compiler_fn(gm)

    return inner


def wrap_restore_state(fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        prior_grad_mode = torch.is_grad_enabled()
        rng_state = torch.clone(torch.random.get_rng_state())
        try:
            return fn(*args, **kwargs)
        finally:
            torch._C._set_grad_enabled(prior_grad_mode)
            torch.random.set_rng_state(rng_state)

    return _fn


def convert_frame_assert(compiler_fn: Callable, one_graph=True):
    """Fully convert a frame into an FX graph"""
    if len(inspect.signature(compiler_fn).parameters) == 1:
        # older 1-arg version
        compiler_fn = wrap_compiler_fn(compiler_fn)

    def _convert_frame_assert(frame: types.FrameType, cache_size: int):
        code = frame.f_code
        input_codes.add(code)
        if code.co_filename.startswith("<eval_with_key>") or code in output_codes:
            return None  # skip FX output
        if (
            os.environ.get("DEBUG_FUNCTION")
            and os.environ.get("DEBUG_FUNCTION") != code.co_name
        ):
            return None
        if is_generator(code):
            unimplemented("generator")
        if cache_size >= config.cache_size_limit:
            unimplemented("cache_size_limit reached")
        output = None

        # from .utils import print_once;  print_once(code.co_filename)

        def transform(instructions, code_options):
            nonlocal output
            tracer = InstructionTranslator(
                instructions,
                frame.f_code,
                frame.f_locals,
                frame.f_globals,
                frame.f_builtins,
                code_options,
                compiler_fn,
                one_graph,
            )
            tracer.run()
            output = tracer.output
            assert output.output_instructions
            instructions[:] = output.output_instructions
            code_options.update(output.code_options)

            if config.dead_code_elimination:
                instructions[:] = remove_pointless_jumps(remove_dead_code(instructions))

        try:
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
                for guard in sorted(output.guards):
                    print(" -", str(guard))
                print()
            assert output.guards is not None
            CleanupManager.instance[code] = output.cleanups
            return GuardedCode(code, output.guards, frame.f_locals, frame.f_globals)
        except Exception as e:
            if config.debug:
                print(
                    "\nWONT CONVERT",
                    e,
                    code.co_name,
                    code.co_filename,
                    code.co_firstlineno,
                )
                # print(dis.Bytecode(frame.f_code).info())
                print(dis.Bytecode(frame.f_code).dis())
                traceback.print_exc()
            raise

    return wrap_restore_state(_convert_frame_assert)


def convert_frame(compiler_fn: typing.Callable):
    """Try to convert a frame into an FX graph, if error leave frame unmodified"""
    inner_convert = convert_frame_assert(compiler_fn, one_graph=False)

    def _convert_frame(frame: types.FrameType, cache_size: int):
        counters["frames"]["total"] += 1
        try:
            result = inner_convert(frame, cache_size)
            counters["frames"]["ok"] += 1
            return result
        except Unsupported:
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
