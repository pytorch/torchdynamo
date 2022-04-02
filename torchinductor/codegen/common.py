import collections
import contextlib
import re
import textwrap
from io import StringIO
from itertools import chain

import sympy
from sympy.printing.printer import Printer

from ..virtualized_ops import ops


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


class OpOverrides:
    def __init__(self, parent):
        super().__init__()
        self._parent = parent

    def __getattr__(self, item):
        return getattr(self._parent, item)

    @staticmethod
    def constant(value, dtype):
        return repr(value)


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
    overrides = None
    load_format = None
    store_format = None

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
        parent_handler = self.overrides(ops.get_handler())
        self.exit_stack.enter_context(ops.set_handler(CSEProxy()))
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
