import collections
import itertools
from itertools import chain

import sympy
import torch

from .. import codecache
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import OpOverrides
from .common import PointwiseKernel
from .common import product


class TritonPrinter(ExprPrinter):
    pass


texpr = TritonPrinter().doprint


class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    def to_dtype(x, dtype: torch.dtype):
        triton_type_name = str(dtype).split(".")[-1]
        return f"{x}.to(tl.{triton_type_name})"

    @staticmethod
    def abs(x):
        return f"tl.abs({x})"

    @staticmethod
    def minimum(a, b):
        return f"tl.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"tl.maximum({a}, {b})"


class TritonPointwiseKernel(PointwiseKernel):
    overrides = TritonOverrides
    sexpr = texpr

    def __init__(self, numel):
        super(TritonPointwiseKernel, self).__init__()
        self.numel = numel
        self.iter_range_tree = dict()
        self.iter_vars_count = itertools.count()

    def add_ranges(self, lengths):
        # make sure needed vars in in call_args
        self.rename_indexing(lengths[:-1])
        itervars = []
        tree = self.iter_range_tree
        for sv in lengths:
            if sv not in tree:
                tree[sv] = (sympy.Symbol(f"i{next(self.iter_vars_count)}"), dict())
            iv, tree = tree[sv]
            itervars.append(iv)
        return itervars

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        if len(index.free_symbols) == 0:
            broadcast = " + " + self.cse.generate(
                self.loads, "tl.zeros((BLOCK_SIZE, ), tl.int32)"
            )
        else:
            broadcast = ""
        line = f"tl.load({var} + {texpr(index)}{broadcast}, mask=mask)"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value):
        var = self.args.output(name)
        index = self.rename_indexing(index)
        line = f"tl.store({var} + {texpr(index)}, {value}, mask=mask)"
        self.stores.writeline(line)

    @classmethod
    def codegen(cls, graph, outputs, schedule):
        kernels = []

        # loop nests by number of elements
        loop_nests = collections.defaultdict(list)
        for output_name, node in outputs.items():
            loop_nests[product(node.get_size())].append((output_name, node))

        for numel, named_nodes in loop_nests.items():
            with TritonPointwiseKernel(numel) as kernel:
                for output_name, node in named_nodes:
                    node.store_output(output_name, kernel.add_ranges(node.get_size()))
                kernels.append(kernel)

        for kernel in kernels:
            kernel_name = schedule.next_kernel_name()
            schedule.define_kernel(kernel_name, kernel.codegen_kernel())
            kernel.call_kernel(schedule, kernel_name)

    def codegen_kernel(self):
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
                    indices0 = offset + tl.arange(0, BLOCK_SIZE)
                    if NEED_MASK:
                        mask = indices0 < numel
                    else:
                        mask = None
                """,
                strip=True,
            )

            def walk_indices(indices, tree):
                """Splat out all our indexing math"""
                nonlocal indices_count
                indices_count += 1
                subindices = f"indices{indices_count}"
                for size, (var, subtree) in tree.items():
                    if subtree:
                        size = TritonPrinter.paren(texpr(self.rename_indexing(size)))
                        code.splice(
                            f"""
                                {var} = {indices} % {size}
                                {subindices} = {indices} // {size}
                            """,
                            strip=True,
                        )
                        walk_indices(subindices, subtree)
                    else:
                        code.writeline(f"{var} = {indices}")

            indices_count = 0
            walk_indices("indices0", self.iter_range_tree)

            code.splice(self.loads.getvalue())
            code.splice(self.compute.getvalue())
            code.splice(self.stores.getvalue())

        wrapper = IndentedBuffer()
        wrapper.writeline("TritonCodeCache.load('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''').kernel")
        return wrapper.getvalue()

    def call_kernel(self, schedule, name: str):
        code = schedule.body
        call_args = list(
            chain(
                self.args.input_buffers.keys(),
                self.args.output_buffers.keys(),
                self.args.sizevars.keys(),
            )
        )
        code.writeline(f"{name}_numel = {texpr(self.numel)}")
        call_args.append(f"{name}_numel")
        code.writeline(f"{name}[grid({name}_numel)](")
        with code.indent():
            code.writeline(", ".join(call_args))
        code.writeline(")")
