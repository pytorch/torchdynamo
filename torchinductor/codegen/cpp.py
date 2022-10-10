import contextlib
import dataclasses
import functools
from pathlib import Path
from typing import Dict
from typing import List

import sympy
import torch
from torch._prims_common import is_float_dtype

from .. import codecache
from .. import config
from .. import ir
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
    torch.float16: "half",
    torch.int64: "long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
    torch.bfloat16: "bfloat16",
}
INDEX_TYPE = "long"

RTYPE_TO_CPP = {
    "sum": "+",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "any": "||",
}


def reduction_init(reduction_type, dtype):
    if reduction_type in ("sum", "any"):
        return 0
    if reduction_type in {"max", "argmax"}:
        return (
            f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()"
        )
    if reduction_type in {"min", "argmin"}:
        return (
            f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()"
        )
    assert False, reduction_type


def reduction_combine(reduction_type, var, next_value):
    if reduction_type == "sum":
        return f"{var} += {next_value}"
    if reduction_type == "any":
        return f"{var} = {var} || {next_value}"
    return f"{var} = std::{reduction_type}({var}, {next_value})"


index_value_name_counter = 1


def argmax_argmin_prefix(reduction_type, src_dtype, tmpvar):
    global index_value_name_counter
    struct_name = f"IndexValue_{index_value_name_counter}"
    index_value_name_counter += 1

    # A small annoyance, due to it being a little cumbersome to just throw {} into strings
    prefix = [
        f"struct {struct_name} {{size_t index; {DTYPE_TO_CPP[src_dtype]} value;}};",
        f"{struct_name} {tmpvar}{{0, {reduction_init(reduction_type, src_dtype)}}};",
    ]
    if reduction_type == "argmax":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmax : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value < omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value < omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    elif reduction_type == "argmin":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmin : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value > omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value > omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    return prefix


def float16_reduction_prefix(rtype):
    # TODO: This user-defined reduction uses float16 accumulation for sum. To reduce numerical
    # errors, float32 accumulation should be used instead.
    assert rtype in (
        "sum",
        "any",
    ), f"float16 user-defined reduction only supports 'sum' and 'any' but got {rtype}"
    prefix = [
        f"#pragma omp declare reduction({RTYPE_TO_CPP[rtype]}:{DTYPE_TO_CPP[torch.float16]}:"
        + f"omp_out = omp_out {RTYPE_TO_CPP[rtype]} omp_in)"
    ]
    return prefix


@functools.lru_cache()
def cpp_prefix():
    path = Path(__file__).parent / "cpp_prefix.h"
    with path.open() as f:
        _, filename = codecache.write(
            f.read(),
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


class CppFastVecOverrides(OpOverrides):
    """Map element-wise ops to aten vectorization C++"""

    @staticmethod
    def abs(x):
        return f"{x}.abs()"

    @staticmethod
    def sin(x):
        return f"{x}.sin()"

    @staticmethod
    def cos(x):
        return f"{x}.cos()"

    @staticmethod
    def exp(x):
        return f"{x}.exp()"

    @staticmethod
    def sqrt(x):
        return f"{x}.sqrt()"

    @staticmethod
    def rsqrt(x):
        return f"{x}.rsqrt()"

    @staticmethod
    def pow(a, b):
        return f"{a}.pow({b})"

    @staticmethod
    def log(x):
        return f"{x}.log()"

    @staticmethod
    def round(x):
        return f"{x}.round()"

    @staticmethod
    def floor(x):
        return f"{x}.floor()"

    @staticmethod
    def ceil(x):
        return f"{x}.ceil()"

    @staticmethod
    def trunc(x):
        return f"{x}.trunc()"

    @staticmethod
    def add(x, y):
        return f"{x} + {y}"

    @staticmethod
    def fmod(a, b):
        return f"{a}.fmod({b})"

    @staticmethod
    def lgamma(x):
        return f"{x}.lgamma()"

    @staticmethod
    def relu(x):
        dtype_str = DTYPE_TO_CPP[V.kernel.cse.get_dtype(x)]
        quot = f"at::vec::Vectorized<{dtype_str}>(0)"
        return f"at::vec::Vectorized<{dtype_str}>::clamp_min({x}, {quot})"

    @staticmethod
    def minimum(a, b):
        dtype_str = DTYPE_TO_CPP[V.kernel.cse.get_dtype(a)]
        assert dtype_str == DTYPE_TO_CPP[V.graph.get_dtype(b)]
        return f"at::vec::Vectorized<{dtype_str}>::minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        dtype_str = DTYPE_TO_CPP[V.kernel.cse.get_dtype(a)]
        assert dtype_str == DTYPE_TO_CPP[V.kernel.cse.get_dtype(b)]
        return f"at::vec::Vectorized<{dtype_str}>::maximum({a}, {b})"

    @staticmethod
    def constant(val, dtype):
        dtype_str = DTYPE_TO_CPP[dtype]
        if val == float("inf"):
            quot = f"std::numeric_limits<{dtype_str}>::infinity()"
            return f"at::vec::Vectorized<{dtype_str}>({quot})"
        elif val == float("-inf"):
            quot = f"-std::numeric_limits<{dtype_str}>::infinity()"
            return f"at::vec::Vectorized<{dtype_str}>({quot})"
        elif val is True or val is False:
            # TODO: It will trigger not-implemented error
            return ops.to_dtype(str(val).lower(), dtype)
        return ops.to_dtype(repr(val), dtype)

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"


class SimdVecChecker:
    def __init__(self):
        super().__init__()
        self.passed = True
        self.fast_vec_list = ["load", "store"]
        for dict_obj in CppFastVecOverrides.__dict__:
            if isinstance(CppFastVecOverrides.__dict__[dict_obj], staticmethod):
                self.fast_vec_list.append(dict_obj)

        self.exit_stack = contextlib.ExitStack()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def __enter__(self):
        class SimdVecCheckerProxy:
            @staticmethod
            def __getattr__(name):
                def inner(*args, **kwargs):
                    if not (name in self.fast_vec_list):
                        self.passed = False
                    return self.passed

                return inner

        self.exit_stack.enter_context(V.set_ops_handler(SimdVecCheckerProxy()))


class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def to_dtype(x, dtype):
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
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
    def rsqrt(x):
        return f"1 / std::sqrt({x})"

    @staticmethod
    def signbit(x):
        return f"std::signbit({x})"

    @staticmethod
    def pow(a, b):
        return f"std::pow({a}, {b})"

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
    def floordiv(a, b):
        # a and b are integer type
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})"

    @staticmethod
    def ceil(x):
        return f"std::ceil({x})"

    @staticmethod
    def trunc(x):
        return f"std::trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        return f"{a} / {b}"

    @staticmethod
    def fmod(a, b):
        return f"std::fmod({a}, {b})"

    @staticmethod
    def isinf(x):
        return f"std::isinf({x})"

    @staticmethod
    def isnan(x):
        return f"std::isnan({x})"

    @staticmethod
    def lgamma(x):
        return f"std::lgamma({x})"

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
        elif val is True or val is False:
            return ops.to_dtype(str(val).lower(), dtype)
        return ops.to_dtype(repr(val), dtype)

    @staticmethod
    def index_expr(expr, dtype):
        return ops.to_dtype(cexpr(V.kernel.rename_indexing(expr)), dtype)

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()
        var = V.kernel.cse.newvar()
        if other == float("-inf"):
            code.writeline(f"float {var} = -std::numeric_limits<float>::infinity();")
        elif other == float("inf"):
            code.writeline(f"float {var} = std::numeric_limits<float>::infinity();")
        else:
            code.writeline(f"auto {var} = {other!r};")
        code.writeline(f"if({mask})")
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f"{var} = {result};")
        V.kernel.compute.splice(code)
        return var

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"

    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(normalized_rand_cpu({seed}, {offset}));"

    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(randn_cpu({seed}, {offset}));"


class CppOverridesProxy(OpOverrides):
    def __init__(self, parent):
        super().__init__(parent)
        self.cpp_overrides = CppOverrides(parent)
        self.cpp_fast_vec_overrides = CppFastVecOverrides(parent)

    def __getattr__(self, __name: str):
        assert isinstance(V.kernel, CppKernel)
        if V.kernel.simd_vec:
            return getattr(self.cpp_fast_vec_overrides, __name)
        else:
            return getattr(self.cpp_overrides, __name)


class CppKernel(Kernel):
    overrides = CppOverridesProxy
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args, num_threads):
        super(CppKernel, self).__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = DeferredIndentedBuffer()
        self.reduction_vars = {}
        self.num_threads = num_threads  # num_threads the kernel specialized for
        self.simd_len = config.cpp.simdlen
        self.simd_vec = False

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)

        if self.simd_len and self.simd_vec and V.graph.get_dtype(name) is torch.float:
            line = f"at::vec::Vectorized<float>::loadu({var} + {cexpr(index * self.simd_len)})"
        else:
            line = f"{var}[{cexpr(index)}]"

        if V.graph.get_dtype(name) in (torch.float16, torch.bfloat16):
            line = f"static_cast<float>({line})"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        if mode is None:
            if self.simd_len and self.simd_vec:
                line = f"{value}.store({var} + {cexpr(index * self.simd_len)});"
            else:
                line = f"{var}[{cexpr(index)}] = {value};"
        elif mode == "atomic_add":
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                line = f"{var}[{cexpr(index)}] += {value};"
            else:
                line = f"atomic_add(&{var}[{cexpr(index)}], {value});"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(name, line)

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        tmpvar = self.cse.generate(
            self.loads, f"reduction {name} {cexpr(index)}", write=False
        )
        index = self.rename_indexing(index)
        self.reduction_vars[tmpvar] = reduction_type
        if argmax_or_argmin:
            self.reduction_prefix.writelines(
                argmax_argmin_prefix(reduction_type, src_dtype, tmpvar)
            )
            compare_op = "<" if reduction_type == "argmax" else ">"
            self.stores.writelines(
                None,
                [
                    f"if ({tmpvar}.value {compare_op} {value}) {{",
                    f"    {tmpvar}.index = {self.itervars[-1]}; {tmpvar}.value = {value};",
                    "}",
                ],
            )
        else:
            if dtype == torch.float16:
                self.reduction_prefix.writelines(
                    float16_reduction_prefix(reduction_type)
                )
            self.reduction_prefix.writeline(
                f"{DTYPE_TO_CPP[dtype]} {tmpvar} = {reduction_init(reduction_type, dtype)};"
            )
            self.stores.writeline(
                None, f"{reduction_combine(reduction_type, tmpvar, value)};"
            )

        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            member_name = ".index" if argmax_or_argmin else ""
            self.reduction_suffix.writeline(
                name, f"{var}[{cexpr(index)}] = {tmpvar}{member_name};"
            )
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

    def codegen_loops(self, code, simd_vec_kernel, worksharing):
        threads = config.cpp.threads
        if threads < 1:
            threads = torch.get_num_threads()

        loops = [LoopLevel(var, size) for var, size in zip(self.itervars, self.ranges)]
        loops_nest_non_reduc, reductions = LoopNest(
            loops[: self.reduction_depth]
        ), LoopNest(loops[self.reduction_depth :])
        reductions.mark_reduction(self.reduction_vars)

        if config.cpp.simdlen:
            # TODO(jansel): detect stride-1 dimension and vectorize that
            if reductions:
                reductions.loops[-1].simd_omp = True
            else:
                loops_nest_non_reduc.loops[-1].simd_omp = True

        par_depth = 0
        reduction_par_depth = 0
        if loops_nest_non_reduc:
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
                loops_nest_non_reduc.mark_parallel(par_depth)
            elif reduction_par_depth:
                # need to close the worksharing scope to define reduction vars outside it
                worksharing.close()
                reductions.mark_parallel(reduction_par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            simd_omp_kernel = self
            loops_nest_non_reduc.vectorize_most_inner_loop(
                simd_vec_kernel, simd_omp_kernel
            )
            loops_nest_non_reduc.codegen(code, stack)

            if isinstance(loops_nest_non_reduc.loops[-1], LoopLevelWithTail):
                pass
            else:
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
        # if we assume thread number is dynamic, make sure we
        # have at least one parallel scope and let OMP runtime
        # to manage the serial vs. parallel.
        if config.cpp.dynamic_threads and depth == 0 and len(ranges) > 0:
            depth = 1
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

    @staticmethod
    def can_fuse_horizontal(node1, node2):
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group
        if vars1 == vars2 and reduce1 == reduce2:
            return True
        if reduce1 == () and vars1 == vars2 + reduce2:
            return True
        # TODO(jansel): allow fusion pointwise (vars1, ()) suffix?
        return False

    @classmethod
    def can_fuse_vertical(cls, node1, node2):
        return cls.can_fuse_horizontal(node1, node2) and not node1.is_reduction()

    def check_simd_vec(self, nodes):
        res = False
        with SimdVecChecker():

            def get_index(*args, **kwargs):
                return sympy.Symbol("dummy")

            _submodules = {"get_index": get_index}

            class PatchSubmodule:
                def __init__(self, node) -> None:
                    self.org_sub_moduel = node._body.submodules

                def __exit__(self, exc_type, exc_val, exc_tb):
                    node._body.submodules = self.org_sub_moduel

                def __enter__(self):
                    node._body.submodules = _submodules

            for node in nodes:
                with PatchSubmodule(node=node):
                    res = node._body.root_block()
                    if not res:
                        return res
        return res

    def _codegen_nodes_impl(self, nodes, is_simd_vec=False):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group
        in_suffix = False

        with kernel_group.new_kernel() as kernel:
            vars, reduction_vars = kernel.set_ranges(group, reduction_group)
            kernel.simd_vec = is_simd_vec

            for node in nodes:
                if node.group[1] in [
                    (group, reduction_group),
                    (group + reduction_group, ()),
                ]:
                    assert not in_suffix
                    node.run(vars, reduction_vars)
                else:
                    in_suffix = True
                    assert node.group[1] == (
                        group,
                        (),
                    ), f"unexpected group: {node.group[1]} != {group}, {reduction_group}"
                    # we can fuse in some extra pointwise into the suffix
                    with kernel.write_to_suffix():
                        node.run(vars, ())

        return kernel

    def codegen_nodes(self, nodes):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group

        can_be_simd_vec = self.check_simd_vec(nodes)
        simd_vec_kernel = (
            self._codegen_nodes_impl(nodes, can_be_simd_vec)
            if can_be_simd_vec
            else None
        )
        scalar_kernel = self._codegen_nodes_impl(nodes)

        kernel_group.finalize_kernel(scalar_kernel, simd_vec_kernel, None)

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
        return CppKernel(self.args, self.ws.num_threads)

    def finalize_kernel(self, scalar_kernel, simd_vec_kernel, scheduler):
        self.count += 1
        code = self.loops_code
        ws = self.ws
        scalar_kernel.codegen_loops(code, simd_vec_kernel, ws)

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
        codecache_def.writeline("async_compile.cpp('''")
        codecache_def.splice(code)
        codecache_def.writeline("''')")

        kernel_name = wrapper.next_kernel_name()
        codecache_str = codecache_def.getvalue()
        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        codecache_str = codecache_str.replace("#pragma CMT", "//")
        wrapper.define_kernel(kernel_name, codecache_str)

        # generate the code to call this
        wrapper.writeline(
            "{}({})".format(kernel_name, ", ".join(call_args)),
        )


class WorkSharing:
    def __init__(self, code):
        self.code = code
        self.in_parallel = False
        self.num_threads = None
        self.stack = contextlib.ExitStack()

    def parallel(self, threads):
        if self.in_parallel and threads != self.num_threads:
            # wrong number of threads
            self.close()
        if not self.in_parallel:
            self.num_threads = threads
            self.in_parallel = True
            if config.cpp.dynamic_threads:
                self.code.writeline("#pragma omp parallel")
            else:
                self.code.writeline(f"#pragma omp parallel num_threads({threads})")
            self.stack.enter_context(self.code.indent())

    def single(self):
        if self.in_parallel:
            self.code.writeline("#pragma omp single")
        return self.in_parallel

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
    var: sympy.Expr = None
    size: sympy.Expr = None
    offset: sympy.Expr = sympy.Integer(0)
    parallel: int = 0
    simd_omp: bool = False
    simd_len: int = config.cpp.simdlen
    simd_vec: bool = False
    collapsed: bool = False
    reduction_vars: Dict[str, str] = None

    def lines(self):
        if self.reduction_vars:
            reduction = " " + " ".join(
                f"reduction({RTYPE_TO_CPP[rtype]}:{var})"
                for var, rtype in self.reduction_vars.items()
            )
        else:
            reduction = ""
        simd = f"simd simdlen({self.simd_len})" if self.simd_omp else ""
        if self.parallel:
            # TODO(jansel): look into chunk size and other schedules
            line1 = f"#pragma omp for{reduction} "
            if self.parallel > 1:
                line1 += f" collapse({self.parallel})"
            if self.simd_omp:
                line1 = line1.replace(" for ", f" for {simd}")
        elif self.simd_vec:
            line1 = f""
        elif self.simd_omp:
            line1 = f"#pragma omp {simd}{reduction}"
        elif not self.reduction_vars and codecache.is_gcc():
            line1 = "#pragma GCC ivdep"
        else:
            line1 = ""
        line2 = f"for({INDEX_TYPE} {self.var}={cexpr(self.offset)}; {self.var}<{cexpr(self.size)}; ++{self.var})"
        if self.collapsed or not line1:
            return [line2]
        return [line1, line2]


class LoopLevelWithTail(LoopLevel):
    def __init__(self, main_loop: LoopLevel, tail_loop: LoopLevel):
        super().__init__()
        self.main_loop = main_loop
        self.tail_loop = tail_loop
        self.main_loop_body = None
        self.tail_loop_body = None

    def lines(self):
        assert False

    def code_gen(self, code):
        assert self.main_loop_body
        assert self.tail_loop_body
        assert isinstance(self.main_loop_body, Kernel)
        assert isinstance(self.tail_loop_body, Kernel)
        for loop, kernel in (
            (self.main_loop, self.main_loop_body),
            (self.tail_loop, self.tail_loop_body),
        ):
            with contextlib.ExitStack() as stack:
                code.writelines(loop.lines())
                stack.enter_context(code.indent())
                with contextlib.ExitStack() as stack:
                    code.splice(kernel.loads)
                    code.splice(kernel.compute)
                    code.splice(kernel.stores)


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
        loops[0].simd_omp = loops[par_depth - 1].simd_omp

    def split_most_inner_loop(self, factor):
        sympy_factor = sympy.Integer(factor)

        most_inner_loop = self.loops[-1]
        main_loop_range = ir.IndexingDiv(most_inner_loop.size, sympy_factor)

        main_loop = LoopLevel(most_inner_loop.var, main_loop_range)
        main_loop.parallel = most_inner_loop.parallel

        offset = main_loop_range * sympy_factor
        tail_loop = LoopLevel(most_inner_loop.var, most_inner_loop.size)
        tail_loop.offset = offset
        tail_loop.parallel = most_inner_loop.parallel

        loop_with_tail = LoopLevelWithTail(main_loop, tail_loop)
        self.loops[-1] = loop_with_tail

    def vectorize_most_inner_loop(self, simd_vec_kernel, simd_omp_kernel):
        if simd_vec_kernel is not None and simd_vec_kernel.simd_vec:
            self.split_most_inner_loop(config.cpp.simdlen)
            loop_with_tail = self.loops[-1]
            assert isinstance(loop_with_tail, LoopLevelWithTail)

            simd_vec_kernel.simd = False
            simd_vec_kernel.fast_vec = True

            loop_with_tail.tail_loop.simd_omp = True
            loop_with_tail.tail_loop.simd_len = int(config.cpp.simdlen / 2)
            loop_with_tail.tail_loop.simd_vec = False

            loop_with_tail.main_loop_body = simd_vec_kernel
            loop_with_tail.tail_loop_body = simd_omp_kernel

    def codegen(self, code, stack):
        for loop in self.loops:
            if isinstance(loop, LoopLevelWithTail):
                assert self.loops[-1] == loop
                loop.code_gen(code)
            else:
                code.writelines(loop.lines())
                stack.enter_context(code.indent())
        else:
            stack.enter_context(code.indent())
