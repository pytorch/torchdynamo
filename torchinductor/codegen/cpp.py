import contextlib
import dataclasses
import functools
import multiprocessing
import textwrap
from itertools import chain
from typing import Dict
from typing import List

import sympy
import torch

from .. import codecache
from .. import config
from ..scheduler import Scheduler
from ..virtualized import graph
from .common import BracesBuffer
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import Kernel
from .common import KernelArgs
from .common import OpOverrides
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


def reduction_init(reduction_type, dtype):
    if reduction_type == "sum":
        return 0
    # TODO(jansel): infinity for floats?
    if reduction_type == "max":
        return f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()"
    if reduction_type == "min":
        return f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()"
    assert False, reduction_type


def reduction_combine(reduction_type, var, next_value):
    if reduction_type == "sum":
        return f"{var} += {next_value}"
    return f"{var} = std::{reduction_type}({var}, {next_value})"


@functools.lru_cache()
def cpp_prefix():
    _, filename = codecache.write(
        textwrap.dedent(
            """
            #include <algorithm>
            #include <cmath>
            #include <cstdlib>
            #include <limits>
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


class CppKernel(Kernel):
    overrides = CppOverrides
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args):
        super(CppKernel, self).__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.reduction_vars = {}

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        return self.cse.generate(self.loads, f"{var}[{cexpr(index)}]")

    def store(self, name, index, value):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        self.stores.writeline(f"{var}[{cexpr(index)}] = {value};")

    def reduction(self, name, dtype, reduction_type, index, value):
        var = self.args.output(name)
        index = self.rename_indexing(index)
        tmpvar = self.cse.generate(self.loads, f"{var}[{cexpr(index)}]", write=False)
        self.reduction_vars[tmpvar] = reduction_type
        self.reduction_prefix.writeline(
            f"{DTYPE_TO_CPP[dtype]} {tmpvar} = {reduction_init(reduction_type, dtype)};"
        )
        self.stores.writeline(f"{reduction_combine(reduction_type, tmpvar, value)};")
        self.reduction_suffix.writeline(f"{var}[{cexpr(index)}] = {tmpvar};")

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

    def size_hint(self):
        return graph.sizevars.size_hint(product(self.call_ranges))

    @classmethod
    def codegen(cls, wrapper, threads=None):
        args = KernelArgs()
        loops_code = BracesBuffer()
        with WorkSharing(loops_code) as ws:
            cls.schedule_kernels(loops_code, args, ws)
        cls.codegen_define_and_call(loops_code, wrapper, args)

    @classmethod
    def codegen_define_and_call(cls, loops_code, wrapper, args):
        argdefs = ",\n".ljust(25).join(args.cpp_argdefs(graph))
        code = BracesBuffer()
        code.writelines([cpp_prefix(), "" f'extern "C" void kernel({argdefs})'])
        with code.indent():
            code.splice(loops_code)

        codecache_def = IndentedBuffer()
        codecache_def.writeline("CppCodeCache.load('''")
        codecache_def.splice(code)
        codecache_def.writeline("''').kernel")

        kernel_name = wrapper.next_kernel_name()
        wrapper.define_kernel(kernel_name, codecache_def.getvalue())

        # generate the code to call this
        call_args = []
        for name in chain(args.input_buffers.keys(), args.output_buffers.keys()):
            call_args.append(f"{name}_ptr")
        for name in args.sizevars.keys():
            call_args.append(f"c_long({name})")
        wrapper.body.writeline(
            "{}({})".format(kernel_name, ", ".join(call_args)),
        )

    def codegen_loops(self, code, worksharing):
        threads = config.cpp.threads
        if threads < 1:
            threads = multiprocessing.cpu_count()

        loops = [LoopLevel(var, size) for var, size in zip(self.itervars, self.ranges)]
        loops, reductions = LoopNest(loops[: self.reduction_depth]), LoopNest(
            loops[self.reduction_depth :]
        )
        reductions.mark_reduction(self.reduction_vars)

        if config.cpp.simdlen:
            # TODO(jansel): detect stride-1 dimension and vectorize that
            if reductions:
                reductions.loops[-1].simd = True
            else:
                loops.loops[-1].simd = True

        par_depth = 0
        reduction_par_depth = 0
        if loops:
            par_depth = self.decide_parallel_depth(
                self.call_ranges[: self.reduction_depth], threads
            )
        else:
            reduction_par_depth = self.decide_parallel_depth(
                self.call_ranges[self.reduction_depth :], threads
            )

        with contextlib.ExitStack() as stack:
            if par_depth:
                worksharing.parallel(threads)
                loops.mark_parallel(par_depth)
            elif reduction_par_depth:
                # need to close the worksharing scope to define reduction vars outside it
                worksharing.close()
                reductions.mark_parallel(reduction_par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            loops.codegen(code, stack)

            code.splice(self.reduction_prefix)

            if reduction_par_depth:
                worksharing.parallel(threads)

            with contextlib.ExitStack() as stack:
                reductions.codegen(code, stack)
                code.splice(self.loads)
                code.splice(self.compute)
                code.splice(self.stores)

            if reduction_par_depth:
                worksharing.close()

            code.splice(self.reduction_suffix)

    def decide_parallel_depth(self, ranges, threads):
        if threads == 1:
            return 0
        seq = self.size_hint()
        par = 1
        depth = 0
        for expr in ranges:
            hint = graph.sizevars.size_hint(expr)
            if par >= 2 * threads or par == threads:
                break
            if seq // threads < config.cpp.min_chunk_size:
                # not enough work
                break
            depth += 1
            par *= hint
            seq /= hint
        return depth

    @contextlib.contextmanager
    def write_to_suffix(self):
        prior = (self.loads, self.compute, self.stores)
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        yield
        self.reduction_suffix.splice(self.loads)
        self.reduction_suffix.splice(self.compute)
        self.reduction_suffix.splice(self.stores)
        (self.loads, self.compute, self.stores) = prior

    @classmethod
    def schedule_kernels(
        cls, code: BracesBuffer, args: KernelArgs, ws: "WorkSharing"
    ) -> Scheduler:
        scheduler = Scheduler(tuple, graph.buffers)

        for group, reduction_group in scheduler.iter_runable_groups():
            with scheduler.kernel(CppKernel(args)) as kernel:
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)

                # first any pointwise sharing same loops
                for node in scheduler.pop_group((group + reduction_group, ())):
                    node.run(vars, reduction_vars)
                    node.mark_fusable()

                if reduction_group:
                    # reductions
                    reduction_nodes = list(
                        scheduler.pop_group((group, reduction_group))
                    )
                    for node in reduction_nodes:
                        node.run(vars, reduction_vars)
                    # can't fuse reduction into reduction
                    for node in reduction_nodes:
                        node.mark_fusable()

                    # we can fuse in some extra pointwise into the suffix
                    with kernel.write_to_suffix():
                        for node in scheduler.pop_group((group, ())):
                            node.run(vars, ())
                            node.mark_fusable()
            kernel.codegen_loops(code, ws)
            if not ws.in_parallel:
                scheduler.barrier()
            if not scheduler.runable_nodes and scheduler.pending_buffer_names:
                ws.barrier()
                scheduler.barrier()

        return scheduler


class WorkSharing:
    def __init__(self, code):
        self.code = code
        self.in_parallel = False
        self.need_barrier = False
        self.stack = contextlib.ExitStack()

    def parallel(self, threads):
        if not self.in_parallel:
            self.in_parallel = True
            self.code.writeline(f"#pragma omp parallel num_threads({threads})")
            self.stack.enter_context(self.code.indent())
        elif self.need_barrier:
            self.code.writeline("#pragma omp barrier")
        self.need_barrier = False

    def single(self):
        if self.in_parallel:
            self.code.writeline("#pragma omp single nowait")
        return self.in_parallel

    def barrier(self):
        self.need_barrier = True

    def close(self):
        self.stack.close()
        self.in_parallel = False

    def __enter__(self):
        self.stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.__exit__(exc_type, exc_val, exc_tb)


@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    parallel: int = 0
    simd: bool = False
    collapsed: bool = False
    reduction_vars: Dict[str, str] = None

    def lines(self):
        if self.reduction_vars:
            lookup = {"sum": "+", "min": "min", "max": "max"}
            reduction = " " + " ".join(
                f"reduction({lookup[rtype]}:{var})"
                for var, rtype in self.reduction_vars.items()
            )
        else:
            reduction = ""
        simd = f"simd simdlen({config.cpp.simdlen})"
        if self.parallel:
            # TODO(jansel): look into chunk size and other schedules
            line1 = f"#pragma omp for nowait{reduction}"
            if self.parallel > 1:
                line1 += f" collapse({self.parallel})"
            if self.simd:
                line1 = line1.replace(" for ", f" for {simd}")
        elif self.simd:
            line1 = f"#pragma omp {simd}{reduction}"
        elif not self.reduction_vars:
            line1 = "#pragma GCC ivdep"
        else:
            line1 = ""
        line2 = f"for({INDEX_TYPE} {self.var}=0; {self.var}<{cexpr(self.size)}; ++{self.var})"
        if self.collapsed or not line1:
            return [line2]
        return [line1, line2]


@dataclasses.dataclass
class LoopNest:
    loops: List[LoopLevel]

    def __bool__(self):
        return bool(self.loops)

    def mark_reduction(self, reduction_vars):
        for loop in self.loops:
            loop.reduction_vars = reduction_vars

    def mark_parallel(self, par_depth):
        loops = self.loops
        loops[0].parallel = par_depth
        for i in range(1, par_depth):
            loops[i].collapsed = True
        loops[0].simd = loops[par_depth - 1].simd

    def codegen(self, code, stack):
        for loop in self.loops:
            code.writelines(loop.lines())
            stack.enter_context(code.indent())
