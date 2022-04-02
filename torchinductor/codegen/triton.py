import functools
import itertools
import operator
from itertools import chain

import torch

from .. import codecache
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import OpOverrides
from .common import PointwiseKernel

product = functools.partial(functools.reduce, operator.mul)


class TritonPrinter(ExprPrinter):
    pass


texpr = TritonPrinter().doprint


class TritonOverrides(OpOverrides):
    @staticmethod
    def to_dtype(x, dtype: torch.dtype):
        triton_type_name = str(dtype).split(".")[-1]
        return f"{x}.to(tl.{triton_type_name})"

    @staticmethod
    def abs(x):
        return f"tl.abs({x})"


class TritonPointwiseKernel(PointwiseKernel):
    overrides = TritonOverrides
    load_format = "tl.load({var} + {index}, mask=mask)"
    store_format = "tl.store({var} + {index}, {value}, mask=mask)"
    sexpr = texpr

    def rename_indexing(self, index, load_store=False):
        index = super().rename_indexing(index, load_store)
        if load_store and index == 0:
            return "tl.zeros((BLOCK_SIZE,), dtype=tl.int32)"
        return index

    def generate(self, graph):
        code = IndentedBuffer()
        code.splice(
            f"""
                import triton
                import triton.language as tl
                from {codecache.__name__} import pointwise_heuristics

                @triton.heuristics(pointwise_heuristics())
                @triton.jit
            """
        )

        argdefs = [
            *self.args.input_buffers.values(),
            *self.args.output_buffers.values(),
        ]
        for var in self.args.sizevars.values():
            # argdefs.append(f"{var}: tl.constexpr")
            argdefs.append(f"{var}")
        argdefs += [
            "numel",
            "BLOCK_SIZE: tl.constexpr",
            "NEED_MASK: tl.constexpr",
        ]

        code.writeline(f"def kernel({', '.join(argdefs)}):")
        with code.indent():
            code.splice(
                """
                    offset = tl.program_id(0) * BLOCK_SIZE
                    indices = offset + tl.arange(0, BLOCK_SIZE)
                    if NEED_MASK:
                        mask = indices < numel
                    else:
                        mask = None
                """
            )
            for axis, var, size in reversed(
                tuple(zip(itertools.count(), self.itervars, self.ranges))
            ):
                if axis > 0:
                    code.writelines(
                        [
                            f"{var} = indices % {texpr(size)}",
                            f"indices = indices // {texpr(size)}",
                        ]
                    )
                else:
                    code.writeline(f"{var} = indices")

            code.splice(self.loads.getvalue())
            code.splice(self.compute.getvalue())
            code.splice(self.stores.getvalue())

        wrapper = IndentedBuffer()
        wrapper.writeline("TritonCodeCache.load('''")
        wrapper.splice(code.getvalue())
        wrapper.writeline("''').kernel")
        return wrapper.getvalue()

    def call_kernel(self, schedule, code: IndentedBuffer, name: str):
        call_args = list(
            chain(
                self.args.input_buffers.keys(),
                self.args.output_buffers.keys(),
                self.args.sizevars.keys(),
            )
        )
        code.writeline(f"{name}_numel = {texpr(product(self.call_ranges))}")
        call_args.append(f"{name}_numel")
        code.writeline(f"{name}[grid({name}_numel)](")
        with code.indent():
            code.writeline(", ".join(call_args))
        code.writeline(")")
