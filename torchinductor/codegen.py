import code
import collections
import contextlib
import itertools
import textwrap
from io import StringIO
from itertools import chain

import sympy

from . import virtualized


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
        self.contents.write(textwrap.indent(textwrap.dedent(other_code), self.prefix()))


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
    def _lookup(odict, name):
        assert isinstance(name, (str, sympy.Symbol))
        name = str(name)
        if name not in odict:
            odict[name] = name
        return odict[name]

    def __init__(self):
        self.input_buffers = collections.OrderedDict()
        self.output_buffers = collections.OrderedDict()
        self.sizevars = collections.OrderedDict()

    def input(self, name):
        return self._lookup(self.input_buffers, name)

    def output(self, name):
        return self._lookup(self.output_buffers, name)

    def size(self, name):
        return self._lookup(self.sizevars, name)

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


class PointwiseKernel:
    newvar_prefix = ""
    suffix = ""

    def __init__(self):
        super().__init__()
        self.args = KernelArgs()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.exit_stack = contextlib.ExitStack()
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
                index = self.rename_indexing(index)
                return self.cse.generate(
                    self.loads, self.load_format.format(var=var, index=index)
                )

            @staticmethod
            def store(name, index, value):
                var = self.args.output(name)
                index = self.rename_indexing(index)
                self.stores.writeline(
                    self.store_format.format(var=var, index=index, value=value)
                    + self.suffix
                )

        self.exit_stack.__enter__()
        parent_handler = virtualized.prim.get_handler()
        self.exit_stack.enter_context(virtualized.prim.set_handler(CSEProxy()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def rename_indexing(self, index):
        subs = {
            x: self.args.size(x) for x in index.free_symbols if str(x).startswith("s")
        }
        return index.subs(subs)

    def set_ranges(self, lengths):
        assert not self.ranges
        self.ranges = [self.rename_indexing(x) for x in lengths]
        self.itervars = [sympy.Symbol(f"i{n}") for n in range(len(self.ranges))]
        return self.itervars


class CppPointwiseKernel(PointwiseKernel):
    newvar_prefix = "auto "
    suffix = ";"
    load_format = "{var}[{index}]"
    store_format = "{var}[{index}] = {value}"

    def getvalue(self):
        code = BracesBuffer()
        with contextlib.ExitStack() as stack:
            code.writeline(f"void kernel({', '.join(self.args.call_names())})")
            stack.enter_context(code.indent())
            for var, size in zip(self.itervars, self.ranges):
                # TODO(jansel): add parallel
                code.writeline(f"#pragma GCC ivdep")
                code.writeline(f"for(int {var}=0; {var}<{size}; ++{var})")
                stack.enter_context(code.indent())
            code.splice(self.loads.getvalue())
            code.splice(self.compute.getvalue())
            code.splice(self.stores.getvalue())
        return code.getvalue()


class TritonPointwiseKernel(PointwiseKernel):
    load_format = "tl.load({var} + {index}, mask=mask)"
    store_format = "tl.store({var} + {index}, {value}, mask=mask)"

    def set_ranges(self, lengths):
        super().set_ranges(lengths)
        self.blockvars = [sympy.Symbol(f"BLOCK{n}") for n in range(len(self.ranges))]
        return self.itervars

    def getvalue(self):
        blockargs = [f"{var}: tl.constexpr" for var in self.blockvars]
        code = IndentedBuffer()
        code.writeline("import triton")
        code.writeline("import triton.language as tl")
        code.writeline("")
        code.writeline("@triton.jit")
        code.writeline(
            f"def kernel({', '.join(chain(self.args.call_names(), blockargs))}):"
        )
        with code.indent():
            for axis, var, size, block_size in zip(
                itertools.count(), self.itervars, self.ranges, self.blockvars
            ):
                code.writeline(
                    f"{var} = tl.program_id(axis={axis}) * {block_size} + tl.arange(0, {block_size})"
                )
                code.writeline(f"mask{axis} = {var} < {size}")
            code.writeline(
                "mask = " + " & ".join(f"mask{i}" for i in range(len(self.itervars)))
            )
            code.splice(self.loads.getvalue())
            code.splice(self.compute.getvalue())
            code.splice(self.stores.getvalue())
        return code.getvalue()

    pass
