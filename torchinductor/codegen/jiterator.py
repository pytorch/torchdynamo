import collections
import contextlib
import dataclasses
import functools
from itertools import chain
from typing import List

import sympy
import torch

from .. import codecache
from .. import config
from .. import ir
from ..scheduler import Scheduler
from ..virtualized import V
from ..virtualized import ops
from .common import BracesBuffer
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import Kernel
# from .common import KernelArgs
from .common import OpOverrides

DTYPE_TO_CPP = {
    torch.float32: "float",
    torch.float64: "double",
    torch.int64: "long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
}
INDEX_TYPE = "long"


class CppPrinter(ExprPrinter):
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} / {div})"
        return f"{x} % {mod}"

    def _print_IndexingDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} / {div})"


cexpr = CppPrinter().doprint
class JiteratorOverrides(OpOverrides):
    """Map element-wise ops to CUDA code"""

    @staticmethod
    def to_dtype(x, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def relu(x):
        return f"{x} * ({x}>0)"

    @staticmethod
    def minimum(a, b):
        return f"::min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"::max({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"{a} ? {b} : {c}"

    @staticmethod
    def constant(val, dtype):
        if val == float("inf"):
            return f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif val == float("-inf"):
            return f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        return ops.to_dtype(repr(val), dtype)

    @staticmethod
    def index_expr(expr, dtype):
        return ops.to_dtype(cexpr(V.kernel.rename_indexing(expr)), dtype)

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()
        var = V.kernel.cse.newvar()
        assert isinstance(other, float)
        if other == float("-inf"):
            code.writeline(f"float {var} = -std::numeric_limits<float>::infinity();")
        else:
            assert False
        code.writeline(f"if({mask})")
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f"{var} = {result};")
        V.kernel.compute.splice(code)
        return var

    @staticmethod
    def and_(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_not(a):
        return f"!{a}"


class JiteratorKernel(Kernel):
    overrides = JiteratorOverrides
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args):
        super(JiteratorKernel, self).__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_vars = {}

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        return self.cse.generate(self.loads, f"{var}")

    def store(self, name, index, value):
        assert "buf" in name
        var = self.args.output(name)
        self.stores.writeline(f"return {value};")

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy.Symbol(f"i{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def codegen_loops(self, code):
        with contextlib.ExitStack():
            code.splice(self.loads)
            code.splice(self.compute)
            code.splice(self.stores)

    @classmethod
    def codegen(cls, wrapper):
        def codegen_extern_call(node: ir.ExternKernel):
            nonlocal kernel_group
            kernel_group.codegen_define_and_call(wrapper)
            node.codegen(wrapper)
            scheduler.barrier()
            kernel_group = KernelGroup()

        scheduler = Scheduler(tuple, V.graph.buffers)
        kernel_group = KernelGroup()

        for group, reduction_group in scheduler.iter_runable_groups(
            codegen_extern_call
        ):
            if reduction_group:
                assert False, "Jiteartor doesn't support reduction yet"
            with scheduler.kernel(kernel_group.new_kernel()) as kernel:
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)

                # first any pointwise sharing same loops
                for node in scheduler.pop_group((group + reduction_group, ())):
                    node.run(vars, reduction_vars)
                    node.mark_fusable()

            kernel_group.finalize_kernel(kernel, scheduler)

        kernel_group.codegen_define_and_call(wrapper)

class JiteratorKernelArgs:
    @staticmethod
    def _lookup(prefix, odict, name):
        assert isinstance(name, (str, sympy.Symbol))
        name = str(name)
        if name not in odict:
            odict[name] = f"{prefix}{len(odict)}"
        return odict[name]

    def __init__(self, sizevars=None):
        self.input_buffers = collections.OrderedDict()
        self.output_buffers = collections.OrderedDict()
        self.sizevars = sizevars or collections.OrderedDict()

    def input(self, name):
        assert name not in V.graph.removed_buffers, name
        if name in self.output_buffers:
            return self.output_buffers[name]
        return self._lookup("in", self.input_buffers, name)

    def output(self, name):
        assert name not in V.graph.removed_buffers, name
        assert name not in self.input_buffers, name
        return self._lookup("out", self.output_buffers, name)

    def size(self, name):
        return self._lookup("ks", self.sizevars, name)

    def call_names(self):
        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )

    def argdefs(self):
        # from .cpp import DTYPE_TO_CPP
        # from .cpp import INDEX_TYPE
        argdefs = []
        for outer, inner in self.input_buffers.items():
            # dtype = buffer_types[outer]
            # argdefs.append(f"const {DTYPE_TO_CPP[dtype]} {inner}")
            argdefs.append(f"const T {inner}")
        return argdefs

class KernelGroup:
    def __init__(self):
        super().__init__()
        self.args = JiteratorKernelArgs()
        self.loops_code = BracesBuffer()
        self.stack = contextlib.ExitStack()
        self.count = 0

    def new_kernel(self):
        return JiteratorKernel(self.args)

    def finalize_kernel(self, new_kernel, scheduler):
        self.count += 1
        code = self.loops_code
        new_kernel.codegen_loops(code)

        if not scheduler.runable_nodes and scheduler.pending_buffer_names:
            scheduler.barrier()

    def codegen_define_and_call(self, wrapper):
        self.stack.close()
        if self.count == 0:
            return

        argdefs = ",\n".ljust(25).join(self.args.argdefs())
        code = BracesBuffer()
        code.writelines([f'template <typename T> T kernel({argdefs})'])

        with code.indent():
            code.splice(self.loops_code)

        codecache_def = IndentedBuffer()
        codecache_def.writeline("'''")
        codecache_def.splice(code)
        codecache_def.writeline("'''")

        kernel_name = wrapper.next_kernel_name()
        wrapper.define_kernel(kernel_name, codecache_def.getvalue())

        # generate the code to call this
        args = self.args
        call_args = []
        for name in args.input_buffers.keys():
            call_args.append(f"{name}")

        output_args = []
        for name in args.output_buffers.keys():
            output_args.append(f"{name}")

        wrapper.body.writeline(
            f"jitted_{kernel_name} = torch.cuda.jiterator._create_jit_fn({kernel_name})",
        )

        wrapper.body.writeline(
            "{} = jitted_{}({})".format(output_args[0], kernel_name, ", ".join(call_args)),
        )
