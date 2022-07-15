import contextlib
import dataclasses
import functools
import multiprocessing
import textwrap
from typing import Dict
from typing import List

import sympy
import torch

from .. import codecache
from .. import config
from ..utils import sympy_product
from ..virtualized import V
from ..virtualized import ops
from .common import BracesBuffer
from .common import DeferredIndentedBuffer
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import Kernel
from .common import KernelArgs
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


def reduction_init(reduction_type, dtype):
    if reduction_type in ("sum", "any"):
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
    if reduction_type == "any":
        return f"{var} = {var} || {next_value}"
    return f"{var} = std::{reduction_type}({var}, {next_value})"


@functools.lru_cache()
def cpp_prefix():
    _, filename = codecache.write(
        textwrap.dedent(
            """
            #include <algorithm>
            #include <atomic>
            #include <cmath>
            #include <cstdlib>
            #include <iostream>
            #include <limits>
            #include <random>
            #include <omp.h>

            template<typename T>
            inline T mod(T a, T b) { return a % b; }
            template<>
            inline float mod(float a, float b) { return std::fmod(a, b); }
            template<>
            inline double mod(double a, double b) { return std::fmod(a, b); }

            float rand_cpu(unsigned int seed) {
                static thread_local std::mt19937 gen(seed ^ omp_get_thread_num());
                static_assert(std::mt19937::min() == 0);
                return gen() * static_cast<float>(1.0 / (std::mt19937::max() + 1.0));
            }
            """
        ),
        "h",
    )
    return f'#include "{filename}"'


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


class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def to_dtype(x, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def abs(x):
        return f"std::abs({x})"

    @staticmethod
    def sin(x):
        return f"std::sin({x})"

    @staticmethod
    def cos(x):
        return f"std::cos({x})"

    @staticmethod
    def exp(x):
        # return f"Sleef_expf_u10({x})"
        return f"std::exp({x})"

    @staticmethod
    def sqrt(x):
        return f"std::sqrt({x})"

    @staticmethod
    def log(x):
        return f"std::log({x})"

    @staticmethod
    def round(x):
        return f"std::nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"std::floor({x})"

    @staticmethod
    def trunc(x):
        return f"std::trunc({x})"

    @staticmethod
    def isinf(x):
        return f"std::isinf({x})"

    @staticmethod
    def isnan(x):
        return f"std::isnan({x})"

    @staticmethod
    def relu(x):
        return f"{x} * ({x}>0)"

    @staticmethod
    def minimum(a, b):
        return f"std::min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"std::max({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"{a} ? {b} : {c}"

    @staticmethod
    def mod(a, b):
        return f"mod({a}, {b})"

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
        elif other == float("inf"):
            code.writeline(f"float {var} = std::numeric_limits<float>::infinity();")
        else:
            code.writeline(f"float {var} = {other!r};")
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
    def rand_cpu(seed: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(rand_cpu({seed}));"


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
        self.reduction_suffix = DeferredIndentedBuffer()
        self.reduction_vars = {}

    def load(self, name: str, index: sympy.Expr, upcast: bool = False):
        # upcast argument is ignored on cpu
        var = self.args.input(name)
        index = self.rename_indexing(index)
        return self.cse.generate(self.loads, f"{var}[{cexpr(index)}]")

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        if mode is None:
            line = f"{var}[{cexpr(index)}] = {value};"
        elif mode == "atomic_add":
            if config.cpp.threads == 1:
                line = f"{var}[{cexpr(index)}] += {value};"
            else:
                # Note atomic_ref requires C++20 and certain processor features
                self.stores.writeline(
                    name,
                    "static_assert(std::atomic_ref<"
                    + f"std::remove_pointer_t<decltype({var})>"
                    + ">::is_always_lock_free);",
                )
                line = f"std::atomic_ref({var}[{cexpr(index)}]) += {value};"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(name, line)

    def reduction(self, name, dtype, reduction_type, index, value):
        tmpvar = self.cse.generate(
            self.loads, f"reduction {name} {cexpr(index)}", write=False
        )
        index = self.rename_indexing(index)
        self.reduction_vars[tmpvar] = reduction_type
        self.reduction_prefix.writeline(
            f"{DTYPE_TO_CPP[dtype]} {tmpvar} = {reduction_init(reduction_type, dtype)};"
        )
        self.stores.writeline(
            None, f"{reduction_combine(reduction_type, tmpvar, value)};"
        )
        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            self.reduction_suffix.writeline(name, f"{var}[{cexpr(index)}] = {tmpvar};")
        self.cse.store_cache[name] = tmpvar

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
        return V.graph.sizevars.size_hint(sympy_product(self.call_ranges))

    def codegen_loops(self, code, worksharing):
        threads = config.cpp.threads
        if threads < 1:
            threads = multiprocessing.cpu_count() // 2

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

        # limit thread count for smaller sizes
        work = self.size_hint()
        if work < 2**18:
            threads = min(threads, 4)
        elif work < 2**22:
            threads = min(threads, 8)

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

            with contextlib.ExitStack() as stack_outer:
                if self.reduction_prefix:
                    stack_outer.enter_context(code.indent())
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
            hint = V.graph.sizevars.size_hint(expr)
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
        prior = (self.loads, self.compute, self.stores, self.cse)
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = DeferredIndentedBuffer()
        self.cse = self.cse.clone()
        yield
        self.reduction_suffix.splice(self.loads)
        self.reduction_suffix.splice(self.compute)
        self.reduction_suffix.splice(self.stores)
        (self.loads, self.compute, self.stores, self.cse) = prior


class CppScheduling:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.kernel_group = KernelGroup()

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen(self, *groups):
        group, reduction_group = groups

        kernel_group = self.kernel_group
        scheduler = self.scheduler
        with scheduler.kernel(kernel_group.new_kernel()) as kernel:
            vars, reduction_vars = kernel.set_ranges(group, reduction_group)

            # first any pointwise sharing same loops
            for node in scheduler.pop_group((group + reduction_group, ())):
                node.run(vars, reduction_vars)
                node.mark_fusable()

            if reduction_group:
                # reductions
                reduction_nodes = list(scheduler.pop_group((group, reduction_group)))
                for node in reduction_nodes:
                    node.run(vars, reduction_vars)

                # can't fuse reduction into reduction, so do in seperate loop
                for node in reduction_nodes:
                    node.mark_fusable()

                # we can fuse in some extra pointwise into the suffix
                with kernel.write_to_suffix():
                    for node in scheduler.pop_group((group, ())):
                        node.run(vars, ())
                        node.mark_fusable()

        kernel_group.finalize_kernel(kernel, scheduler)

    def flush(self):
        self.kernel_group.codegen_define_and_call(V.graph.wrapper_code)
        self.kernel_group = KernelGroup()


class KernelGroup:
    def __init__(self):
        super().__init__()
        self.args = KernelArgs()
        self.loops_code = BracesBuffer()
        self.ws = WorkSharing(self.loops_code)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(self.ws)
        self.count = 0

    def new_kernel(self):
        return CppKernel(self.args)

    def finalize_kernel(self, new_kernel, scheduler):
        self.count += 1
        code = self.loops_code
        ws = self.ws
        new_kernel.codegen_loops(code, ws)
        if not ws.in_parallel:
            scheduler.barrier()
        if not scheduler.runable_nodes and scheduler.pending_buffer_names:
            ws.barrier()
            scheduler.barrier()

    def codegen_define_and_call(self, wrapper):
        self.stack.close()
        if self.count == 0:
            return

        arg_defs, call_args = self.args.cpp_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code = BracesBuffer()
        code.writelines([cpp_prefix(), "" f'extern "C" void kernel({arg_defs})'])
        with code.indent():
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            code.splice(self.loops_code)

        codecache_def = IndentedBuffer()
        codecache_def.writeline("CppCodeCache.load('''")
        codecache_def.splice(code)
        codecache_def.writeline("''').kernel")

        kernel_name = wrapper.next_kernel_name()
        wrapper.define_kernel(kernel_name, codecache_def.getvalue())

        # generate the code to call this
        wrapper.writeline(
            "{}({})".format(kernel_name, ", ".join(call_args)),
        )


class WorkSharing:
    def __init__(self, code):
        self.code = code
        self.in_parallel = False
        self.num_threads = None
        self.need_barrier = False
        self.stack = contextlib.ExitStack()

    def parallel(self, threads):
        if self.in_parallel and threads != self.num_threads:
            # wrong number of threads
            self.close()
        if not self.in_parallel:
            self.num_threads = threads
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
            lookup = {"sum": "+", "min": "min", "max": "max", "any": "||"}
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
        elif not self.reduction_vars and codecache.is_gcc():
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
