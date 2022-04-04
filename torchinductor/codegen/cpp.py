import collections
import contextlib
import dataclasses
import functools
import multiprocessing
import textwrap
from itertools import chain

import sympy
import torch

from .. import codecache
from .. import config
from .common import BracesBuffer
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import KernelArgs
from .common import OpOverrides
from .common import PointwiseKernel
from .common import product

DTYPE_TO_CPP = {
    torch.float32: "float",
    torch.float64: "double",
    torch.int64: "long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
}
INDEX_TYPE = "long"


@functools.lru_cache()
def cpp_prefix():
    _, filename = codecache.write(
        textwrap.dedent(
            """
            #include <cmath>
            #include <cstdlib>
            #include <algorithm>
            """
        ),
        "h",
    )
    return f'#include "{filename}"'


class CppPrinter(ExprPrinter):
    pass


cexpr = CppPrinter().doprint


class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def to_dtype(x, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def abs(x):
        return f"std::abs({x})"

    @staticmethod
    def minimum(a, b):
        return f"std::min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"std::max({a}, {b})"


class CppPointwiseKernel(PointwiseKernel):
    overrides = CppOverrides
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"
    load_format = "{var}[{index}]"
    store_format = "{var}[{index}] = {value}"

    def __init__(self, args):
        super(CppPointwiseKernel, self).__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None

    def set_ranges(self, lengths):
        assert not self.ranges
        self.call_ranges = lengths
        self.ranges = [self.rename_indexing(x) for x in lengths]
        self.itervars = [sympy.Symbol(f"i{n}") for n in range(len(self.ranges))]
        return self.itervars

    def size_hint(self, graph):
        return graph.sizevars.size_hint(product(self.call_ranges))

    @classmethod
    def codegen(cls, graph, outputs, schedule, threads=None):
        if threads is None:
            threads = config.cpp.threads
        if threads < 1:
            threads = multiprocessing.cpu_count()

        args = KernelArgs()
        kernels = []

        # group by common loop nests
        loop_nests = collections.defaultdict(list)
        for output_name, node in outputs.items():
            loop_nests[tuple(node.get_size())].append((output_name, node))

        for sizes, name_nodes in loop_nests.items():
            with CppPointwiseKernel(args) as kernel:
                vars = kernel.set_ranges(sizes)
                # TODO(jansel): make the layout per-tensor
                for output_name, node in name_nodes:
                    node.store_output(output_name, vars)
                kernels.append(kernel)

        if (
            len(kernels) == 1
            and kernels[0].size_hint(graph) // threads < config.cpp.min_chunk_size
        ):
            # not enough work to go multithreaded
            threads = 1

        argdefs = ",\n".ljust(25).join(args.cpp_argdefs(graph))
        code = BracesBuffer()
        code.writelines([cpp_prefix(), "" f'extern "C" void kernel({argdefs})'])
        with contextlib.ExitStack() as stack:
            stack.enter_context(code.indent())
            if threads > 1:
                code.writeline(f"#pragma omp parallel num_threads({threads})")
                stack.enter_context(code.indent())
            for kernel in kernels:
                kernel.codegen_loops(graph, code, threads)

        wrapper = IndentedBuffer()
        wrapper.writeline("CppCodeCache.load('''")
        wrapper.splice(code.getvalue())
        wrapper.writeline("''').kernel")

        kernel_name = schedule.next_kernel_name()
        schedule.define_kernel(kernel_name, wrapper.getvalue())

        # generate the code to call this
        call_args = []
        for name in chain(args.input_buffers.keys(), args.output_buffers.keys()):
            call_args.append(f"{name}_ptr")
        for name in args.sizevars.keys():
            call_args.append(f"c_long({name})")
        schedule.body.writeline(
            "{}({})".format(kernel_name, ", ".join(call_args)),
        )

    def codegen_loops(self, graph, code, threads):
        seq = self.size_hint(graph)
        par = 1
        loops = [LoopLevel(var, size) for var, size in zip(self.itervars, self.ranges)]
        depth = 0

        for expr in self.call_ranges:
            hint = graph.sizevars.size_hint(expr)
            if par >= 2 * threads or par == threads:
                break
            if seq // threads < config.cpp.min_chunk_size:
                # not enough work
                break
            depth += 1
            par *= hint
            seq /= hint

        if config.cpp.simdlen:
            # TODO(jansel): detect the best axis to run SIMD
            loops[-1].simd = True

        with contextlib.ExitStack() as stack:
            if depth and threads > 1:
                loops[0].parallel = depth
                for i in range(1, depth):
                    loops[i].collapsed = True
                loops[0].simd = loops[depth - 1].simd
            elif threads > 1:
                code.writeline("#pragma omp single nowait")
                stack.enter_context(code.indent())
            for loop in loops:
                code.writelines(loop.lines())
                stack.enter_context(code.indent())

            code.splice(self.loads.getvalue())
            code.splice(self.compute.getvalue())
            code.splice(self.stores.getvalue())


@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    parallel: int = 0
    simd: bool = False
    collapsed: bool = False

    def lines(self):
        simd = f"simd simdlen({config.cpp.simdlen})"
        if self.parallel:
            # TODO(jansel): look into chunk size and other schedules
            line1 = "#pragma omp for nowait"
            if self.parallel > 1:
                line1 += f" collapse({self.parallel})"
            if self.simd:
                line1 = line1.replace(" for ", f" for {simd}")
        elif self.simd:
            line1 = f"#pragma omp {simd}"
        else:
            line1 = "#pragma GCC ivdep"
        line2 = f"for({INDEX_TYPE} {self.var}=0; {self.var}<{cexpr(self.size)}; ++{self.var})"
        if self.collapsed or not line1:
            return [line2]
        return [line1, line2]
