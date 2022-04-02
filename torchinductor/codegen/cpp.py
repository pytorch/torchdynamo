import contextlib
import functools
import textwrap
from itertools import chain

import torch

from .. import codecache
from .common import BracesBuffer
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import OpOverrides
from .common import PointwiseKernel


@functools.lru_cache()
def cpp_prefix():
    _, filename = codecache.write(
        textwrap.dedent(
            """
            #include <cmath>
            #include <cstdlib>
            """
        ),
        "h",
    )
    return f'#include "{filename}"'


class CppPrinter(ExprPrinter):
    pass


cexpr = CppPrinter().doprint


class CppOverrides(OpOverrides):
    @staticmethod
    def to_dtype(x, dtype):
        return f"static_cast<{CppPointwiseKernel.dtype_to_cpp[dtype]}>({x})"

    @staticmethod
    def abs(x):
        return f"std::abs({x})"


class CppPointwiseKernel(PointwiseKernel):
    overrides = CppOverrides
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"
    load_format = "{var}[{index}]"
    store_format = "{var}[{index}] = {value}"
    index_type = "long"
    dtype_to_cpp = {
        torch.float32: "float",
        torch.float64: "double",
        torch.int64: "long",
        torch.int32: "int",
        torch.int16: "short",
        torch.int8: "signed char",
        torch.uint8: "unsigned char",
    }

    def generate(self, graph):
        args = []
        for outer, inner in self.args.input_buffers.items():
            dtype = graph.graph_inputs[outer].get_dtype()
            args.append(f"const {self.dtype_to_cpp[dtype]}* __restrict__ {inner}")
        for outer, inner in self.args.output_buffers.items():
            dtype = graph.graph_outputs[outer].get_dtype()
            args.append(f"{self.dtype_to_cpp[dtype]}* __restrict__ {inner}")
        for outer, inner in self.args.sizevars.items():
            args.append(f"const {self.index_type} {inner}")

        code = BracesBuffer()
        with contextlib.ExitStack() as stack:
            fargs = ",\n".ljust(25).join(args)
            code.writelines([cpp_prefix(), "" f'extern "C" void kernel({fargs})'])
            stack.enter_context(code.indent())
            code.writeline(self.omp_parallel_pragma(graph))
            for var, size in zip(self.itervars, self.ranges):
                # code.writeline("#pragma GCC ivdep")
                code.writeline(
                    f"for({self.index_type} {var}=0; {var}<{cexpr(size)}; ++{var})"
                )
                stack.enter_context(code.indent())
            code.splice(self.loads.getvalue())
            code.splice(self.compute.getvalue())
            code.splice(self.stores.getvalue())

        wrapper = IndentedBuffer()
        wrapper.writeline("CppCodeCache.load('''")
        wrapper.splice(code.getvalue())
        wrapper.writeline("''').kernel")
        return wrapper.getvalue()

    def omp_parallel_pragma(self, graph):
        """
        A hacky heuristic to decide what openmp pragma to add.
        """
        seq_work = (
            self.loads.getvalue().count("\n")
            + self.compute.getvalue().count("\n")
            + self.stores.getvalue().count("\n")
        )

        for expr in self.call_ranges:
            seq_work *= graph.sizevars.size_hint(expr)

        par_work = 1
        depth = 0

        for expr in self.call_ranges:
            # TODO(jansel): these constants are total guesses without tuning
            hint = graph.sizevars.size_hint(expr)
            if par_work >= 128:
                break
            if (seq_work / hint) <= 512:
                break
            depth += 1
            par_work *= hint
            seq_work /= hint

        if depth == 0:
            return ""
        if depth == 1:
            return "#pragma omp parallel for"
        return f"#pragma omp parallel for collapse({depth})"

    def call_kernel(self, schedule, code: IndentedBuffer, kernel_name: str):
        call_args = []
        for name in chain(
            self.args.input_buffers.keys(), self.args.output_buffers.keys()
        ):
            call_args.append(f"{name}_ptr")
        for name in self.args.sizevars.keys():
            call_args.append(f"c_long({name})")
        code.writeline(
            "{}({})".format(kernel_name, ", ".join(call_args)),
        )
