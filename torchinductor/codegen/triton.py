import itertools
from itertools import chain

import sympy
import torch

from .. import codecache
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import Kernel
from .common import OpOverrides
from .common import Scheduler
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


class TritonKernel(Kernel):
    overrides = TritonOverrides
    sexpr = texpr
    schedule_group_fn = product

    def __init__(self, numel, reduction_numel):
        super(TritonKernel, self).__init__()
        if reduction_numel is None:
            reduction_numel = sympy.Integer(1)
        self.numel = numel
        self.reduction_numel = reduction_numel
        self.iter_range_tree = dict()
        self.reduction_range_tree = dict()
        self.iter_vars = []
        self.reduction_vars = []
        self.iter_vars_count = itertools.count()
        self.inside_reduction = reduction_numel != 1

    def _add_ranges(self, lengths, var_list, tree, prefix):
        # make sure needed vars in in call_args
        self.rename_indexing(lengths[:-1])
        itervars = []
        for sv in lengths:
            if sv not in tree:
                sym = sympy.Symbol(f"{prefix}{next(self.iter_vars_count)}")
                tree[sv] = (sym, dict())
                var_list.append(sym)
            iv, tree = tree[sv]
            itervars.append(iv)
        return itervars

    def add_ranges(self, lengths):
        return self._add_ranges(lengths, self.iter_vars, self.iter_range_tree, "i")

    def add_reduction_ranges(self, lengths):
        return self._add_ranges(
            lengths, self.reduction_vars, self.reduction_range_tree, "r"
        )

    def set_ranges(self, lengths, reduction_lengths):
        return self.add_ranges(lengths), self.add_reduction_ranges(reduction_lengths)

    def indexing(self, index: sympy.Expr, reductions=True):
        index = self.rename_indexing(index)
        offset = index.subs({v: 0 for v in chain(self.iter_vars, self.reduction_vars)})
        base_part = index.subs({v: 0 for v in self.reduction_vars}) - offset
        reduction_part = index.subs({v: 0 for v in self.iter_vars}) - offset
        addr = []
        if offset != 0:
            addr.append(texpr(offset))

        if base_part != 0:
            addr.append(texpr(base_part))
        else:
            addr.append("tl.zeros((BLOCK_SIZE, ), tl.int32)")

        if self.inside_reduction and reductions:
            addr[-1] = f"tl.reshape({addr[-1]}, (BLOCK_SIZE, 1))"

            if reduction_part != 0:
                addr.append(texpr(reduction_part))
            else:
                addr.append("tl.zeros((REDUCTION_SIZE, ), tl.int32)")

            addr[-1] = f"tl.reshape({addr[-1]}, (1, REDUCTION_SIZE))"
        else:
            assert reduction_part == 0

        return " + ".join(addr)

    def mask(self, reductions=True):
        return (
            "mask=mask_reduction"
            if (self.inside_reduction and reductions)
            else "mask=mask"
        )

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        line = f"tl.load({var} + {self.indexing(index)}, {self.mask()})"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, reductions=True):
        var = self.args.output(name)
        line = f"tl.store({var} + {self.indexing(index, reductions)}, {value}, {self.mask(reductions)})"
        self.stores.writeline(line)

    def reduction(self, name, dtype, reduction_type, index, value):
        default = {"sum": 0, "max": "float('-inf')", "min": "float('inf')"}
        res = self.cse.generate(
            self.compute,
            f"tl.where(mask_reduction, {value}, "
            f"{default[reduction_type]}) if NEED_MASK else {value}",
        )
        res = self.cse.generate(self.compute, f"tl.{reduction_type}({res}, 1)")
        self.store(name, index, res, reductions=False)

    @classmethod
    def codegen(cls, outputs, schedule):
        kernels = cls.schedule_kernels(outputs, TritonKernel)
        cls.codegen_define_and_call(kernels, schedule)

    @classmethod
    def codegen_define_and_call(cls, kernels, schedule):
        for kernel in kernels:
            kernel_name = schedule.next_kernel_name()
            schedule.define_kernel(kernel_name, kernel.codegen_kernel())
            kernel.call_kernel(schedule, kernel_name)

    def codegen_kernel(self):
        code = IndentedBuffer()
        heuristics = (
            "reduction_heuristics" if self.inside_reduction else "pointwise_heuristics"
        )
        code.splice(
            f"""
                import triton
                import triton.language as tl
                from {codecache.__name__} import reduction_heuristics, pointwise_heuristics

                @triton.heuristics({heuristics}())
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

        if self.inside_reduction:
            argdefs += [
                "numel",
                "reduction_numel",
                "BLOCK_SIZE: tl.constexpr",
                "REDUCTION_SIZE: tl.constexpr",
                "NEED_MASK: tl.constexpr",
            ]

        else:
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
                """,
                strip=True,
            )

            if self.inside_reduction:
                code.splice(
                    """
                        reduction0 = tl.arange(0, REDUCTION_SIZE)
                        if NEED_MASK:
                            mask = indices0 < numel
                            mask_reduction = (tl.reshape(mask, (BLOCK_SIZE, 1)) &
                                              tl.reshape(reduction0 < reduction_numel, (1, REDUCTION_SIZE)))
                        else:
                            mask = None
                            mask_reduction = None
                    """,
                    strip=True,
                )
            else:
                code.splice(
                    """
                    if NEED_MASK:
                        mask = indices0 < numel
                    else:
                        mask = None
                """,
                    strip=True,
                )

            def walk_indices(indices, tree, prefix, cnt):
                """Splat out all our indexing math"""
                subindices = f"{prefix}{next(cnt)}"
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
                        walk_indices(subindices, subtree, prefix, cnt)
                    else:
                        code.writeline(f"{var} = {indices}")

            walk_indices(
                "indices0", self.iter_range_tree, "indices", itertools.count(1)
            )
            walk_indices(
                "reduction0", self.reduction_range_tree, "reduction", itertools.count(1)
            )

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
        if self.inside_reduction:
            code.writeline(f"{name}_reduction_numel = {texpr(self.reduction_numel)}")
            call_args.append(f"{name}_reduction_numel")
        code.writeline(f"{name}[grid({name}_numel)](")
        with code.indent():
            code.writeline(", ".join(call_args))
        code.writeline(")")

    @classmethod
    def schedule_kernels(cls, outputs, make_kernel):
        scheduler = Scheduler(cls.schedule_group_fn)
        scheduler.enqueue(outputs)

        kernels = []

        # first schedule reductions, as we can fuse non-reductions into them
        for (
            group,
            reduction_group,
        ), named_nodes in scheduler.reduction_loop_nests.items():
            with make_kernel(group, reduction_group) as kernel:
                kernels.append(kernel)
                for output_name, node in named_nodes:
                    vars, reduction_vars = kernel.set_ranges(
                        node.get_size(), node.get_reduction_size()
                    )
                    node.store_reduction(output_name, vars, reduction_vars)

                # fuse compatible non-reductions
                for output_name, node in scheduler.loop_nests.pop(
                    group * reduction_group, []
                ):
                    sizes = node.get_size()
                    split = split_sizes(sizes, group, reduction_group)
                    if split is None:
                        scheduler.enqueue({output_name: node})
                        continue
                    vars, reduction_vars = kernel.set_ranges(
                        sizes[:split], sizes[split:]
                    )
                    node.store_output(output_name, vars + reduction_vars)

                # fuse more compatible non-reductions
                kernel.inside_reduction = False
                for output_name, node in scheduler.loop_nests.pop(group, []):
                    vars, _ = kernel.set_ranges(node.get_size(), [])
                    node.store_output(output_name, vars)
                kernel.inside_reduction = True

        for group, named_nodes in scheduler.loop_nests.items():
            with make_kernel(group, None) as kernel:
                kernels.append(kernel)
                for output_name, node in named_nodes:
                    vars, _ = kernel.set_ranges(node.get_size(), [])
                    node.store_output(output_name, vars)

        return kernels


def split_sizes(sizes, prod1, prod2):
    for i in range(len(sizes)):
        if product(sizes[:i]) == prod1 and product(sizes[i:]) == prod2:
            return i
    return None
