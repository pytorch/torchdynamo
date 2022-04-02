import collections
import contextlib
import functools
import itertools
import operator
import re
import textwrap
from io import StringIO
from itertools import chain

import sympy
import torch
from sympy.printing.printer import Printer

from . import codecache
from . import virtualized

product = functools.partial(functools.reduce, operator.mul)


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


class ExprPrinter(Printer):
    @staticmethod
    def paren(string):
        if re.match(r"^[a-z0-9_.]+$", string, re.I):
            return string
        return f"({string})"

    def _print_Pow(self, expr):
        # Pow() confuses triton
        base, exp = expr.args
        base = self._print(base)
        assert exp.is_integer
        exp = int(exp)
        return "*".join([self.paren(base)] * exp)

    def _print_Mul(self, expr):
        return "*".join(map(self.paren, map(self._print, expr.args)))

    def _print_Add(self, expr):
        return " + ".join(map(self.paren, map(self._print, expr.args)))

    def _print_Mod(self, expr):
        return " % ".join(map(self.paren, map(self._print, expr.args)))


class TritonPrinter(ExprPrinter):
    pass


class CppPrinter(ExprPrinter):
    pass


texpr = TritonPrinter().doprint
cexpr = CppPrinter().doprint


class PrimOverrides:
    def __init__(self, parent):
        super().__init__()
        self._parent = parent

    def __getattr__(self, item):
        return getattr(self._parent, item)

    @staticmethod
    def constant(value, dtype):
        return repr(value)


class CppOverrides(PrimOverrides):
    @staticmethod
    def to_dtype(x, dtype):
        return f"static_cast<{CppPointwiseKernel.dtype_to_cpp[dtype]}>({x})"

    @staticmethod
    def abs(x):
        return f"std::abs({x})"


class TritonOverrides(PrimOverrides):
    @staticmethod
    def to_dtype(x, dtype: torch.dtype):
        triton_type_name = str(dtype).split(".")[-1]
        return f"{x}.to(tl.{triton_type_name})"

    @staticmethod
    def abs(x):
        return f"tl.abs({x})"


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self.contents = StringIO()
        self._indent = initial_indent
        self.getvalue = self.contents.getvalue

    def prefix(self):
        return " " * (self._indent * self.tabwidth)

    def writeline(self, line):
        self.contents.write(self.prefix())
        self.contents.write(line)
        self.contents.write("\n")

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            yield
            self._indent -= offset

        return ctx()

    def splice(self, other_code):
        other_code = textwrap.dedent(other_code)
        self.contents.write(textwrap.indent(other_code, self.prefix()))


class BracesBuffer(IndentedBuffer):
    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            self.writeline("{")
            self._indent += offset
            yield
            self._indent -= offset
            self.writeline("}")

        return ctx()


class KernelArgs:
    @staticmethod
    def _lookup(prefix, odict, name):
        assert isinstance(name, (str, sympy.Symbol))
        name = str(name)
        if name not in odict:
            odict[name] = f"{prefix}{len(odict)}"
        return odict[name]

    def __init__(self):
        self.input_buffers = collections.OrderedDict()
        self.output_buffers = collections.OrderedDict()
        self.sizevars = collections.OrderedDict()

    def input(self, name):
        return self._lookup("in_ptr", self.input_buffers, name)

    def output(self, name):
        return self._lookup("out_ptr", self.output_buffers, name)

    def size(self, name):
        return self._lookup("ks", self.sizevars, name)

    def call_names(self):
        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )


class CSE:
    """Common subexpression elimination"""

    def __init__(self, prefix="", suffix=""):
        self.prefix = prefix
        self.suffix = suffix
        self.cache = {}

    def generate(self, buffer: IndentedBuffer, expr: str):
        if expr not in self.cache:
            var = f"tmp{len(self.cache)}"
            self.cache[expr] = var
            buffer.writeline(f"{self.prefix}{var} = {expr}{self.suffix}")
        return self.cache[expr]


class CodeGen:
    def __init__(self):
        super().__init__()
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        self.exit_stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)


class PointwiseKernel(CodeGen):
    newvar_prefix = ""
    suffix = ""

    def __init__(self):
        super().__init__()
        self.args = KernelArgs()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.cse = CSE(self.newvar_prefix, self.suffix)
        self.ranges = None
        self.itervars = None

    def __enter__(self):
        class CSEProxy:
            @staticmethod
            def __getattr__(name):
                def inner(*args, **kwargs):
                    return self.cse.generate(
                        self.compute, getattr(parent_handler, name)(*args, **kwargs)
                    )

                return inner

            @staticmethod
            def load(name: str, index: sympy.Expr):
                var = self.args.input(name)
                index = self.rename_indexing(index, load_store=True)
                return self.cse.generate(
                    self.loads,
                    self.load_format.format(var=var, index=self.sexpr(index)),
                )

            @staticmethod
            def store(name, index, value):
                var = self.args.output(name)
                index = self.rename_indexing(index, load_store=True)
                self.stores.writeline(
                    self.store_format.format(
                        var=var, index=self.sexpr(index), value=value
                    )
                    + self.suffix
                )

        super().__enter__()
        parent_handler = self.prims(virtualized.prim.get_handler())
        self.exit_stack.enter_context(virtualized.prim.set_handler(CSEProxy()))
        return self

    def rename_indexing(self, index, load_store=False):
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]
        subs = {
            x: self.args.size(x) for x in index.free_symbols if str(x).startswith("s")
        }
        return index.subs(subs)

    def set_ranges(self, lengths):
        assert not self.ranges
        self.call_ranges = lengths
        self.ranges = [self.rename_indexing(x) for x in lengths]
        self.itervars = [sympy.Symbol(f"i{n}") for n in range(len(self.ranges))]
        return self.itervars


class CppPointwiseKernel(PointwiseKernel):
    prims = CppOverrides
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


class TritonPointwiseKernel(PointwiseKernel):
    prims = TritonOverrides
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

    def call_kernel(self, schedule: "ScheduleCodeGen", code: IndentedBuffer, name: str):
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


class ScheduleCodeGen(CodeGen):
    """
    The outer wrapper that calls the kernels above.
    """

    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.header = IndentedBuffer()
        self.body = IndentedBuffer(initial_indent=1)
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                from torch import empty, empty_like
                # TODO(jansel): handle triton missing
                import triton
                from triton import cdiv
                import triton.language as tl
                from {codecache.__name__} import CppCodeCache, TritonCodeCache, grid
            """
        )
        with self.body.indent(-1):
            self.body.writelines(
                [f"def call({', '.join(self.graph.graph_inputs.keys())}):"]
            )
        self.graph.sizevars.codegen(self.body, self.graph.graph_inputs)
        self.codegen_outputs()

    def codegen_outputs(self):
        code = self.body
        empty_like_cache = dict()
        for name, value in self.graph.graph_inputs.items():
            device = value.get_device()
            dtype = value.get_dtype()
            shape = tuple(value.get_size())
            stride = tuple(value.get_stride())
            empty_like_cache.setdefault((device, dtype, shape, stride), name)

        for name, value in self.graph.graph_outputs.items():
            device = value.get_device()
            dtype = value.get_dtype()
            shape = tuple(value.get_size())
            stride = tuple(value.get_stride())
            key = (device, dtype, shape, stride)
            # TODO(jansel): strides?
            if key in empty_like_cache:
                code.writeline(f"{name} = empty_like({empty_like_cache[key]})")
            else:
                code.writeline(
                    f"{name} = empty([{', '.join(map(str, shape))}], device='{device.type}', dtype={dtype})"
                )
        for name, value in chain(
            self.graph.graph_inputs.items(), self.graph.graph_outputs.items()
        ):
            device = value.get_device()
            if device.type == "cpu":
                code.writeline(f"{name}_ptr = c_void_p({name}.data_ptr())")

    def generate(self):
        self.body.writeline(
            "return (" + ", ".join(self.graph.graph_outputs.keys()) + ", )"
        )
        return f"{self.header.getvalue()}\n\n{self.body.getvalue()}"

    def define_kernel(self, name: str, kernel: PointwiseKernel):
        self.header.splice(f"\n\n{name} = {kernel.generate(self.graph)}")

    def call_kernel(self, name: str, kernel: PointwiseKernel):
        kernel.call_kernel(self, self.body, name)
