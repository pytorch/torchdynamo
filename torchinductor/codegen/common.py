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
from sympy.printing.printer import Printer

from ..virtualized import V
from ..virtualized import ops


def product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


class ExprPrinter(Printer):
    @staticmethod
    def paren(string):
        if (
            re.match(r"^[a-z0-9_.]+$", string, re.I)
            or re.match(r"^\([^)]*\)$", string, re.I)
            or string == ""
        ):
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

    def _print_CleanDiv(self, expr):
        return self._print_IndexingDiv(expr)


class OpOverrides:
    def __init__(self, parent):
        super().__init__()
        self._parent = parent

    def __getattr__(self, item):
        return getattr(self._parent, item)

    @staticmethod
    def constant(value, dtype):
        return repr(value)

    @staticmethod
    def sigmoid(x):
        x = ops.exp(f"-{x}")
        return f"1 / (1 + {x})"

    @staticmethod
    def silu(x):
        return f"{x} * {ops.sigmoid(x)}"

    @staticmethod
    def reciprocal(x):
        return ops.div("1", x)

    @staticmethod
    def square(x):
        return ops.mul(x, x)

    @staticmethod
    def sign(x):
        return ops.where(f"{x} < 0", "-1", "1")


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self.contents = StringIO()
        self._indent = initial_indent
        self.getvalue = self.contents.getvalue

    def __bool__(self):
        return len(self.getvalue()) > 0

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

    def splice(self, other_code, strip=False):
        if isinstance(other_code, IndentedBuffer):
            other_code = other_code.getvalue()
        other_code = textwrap.dedent(other_code)
        if strip:
            other_code = other_code.lstrip()
        if not other_code:
            return
        assert other_code.endswith("\n")
        self.contents.write(textwrap.indent(other_code, self.prefix()))


class BracesBuffer(IndentedBuffer):
    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                self._indent -= 1
                self.writeline("}")
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                self._indent -= 1
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

    def __init__(self, sizevars=None):
        self.input_buffers = collections.OrderedDict()
        self.output_buffers = collections.OrderedDict()
        self.sizevars = sizevars or collections.OrderedDict()

    def input(self, name):
        assert name not in V.graph.removed_buffers, name
        if name in self.output_buffers:
            return self.output_buffers[name]
        return self._lookup("in_ptr", self.input_buffers, name)

    def output(self, name):
        assert name not in V.graph.removed_buffers, name
        assert name not in self.input_buffers, name
        return self._lookup("out_ptr", self.output_buffers, name)

    def size(self, name):
        return self._lookup("ks", self.sizevars, name)

    def call_names(self):
        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )

    def cpp_argdefs(self):
        from .cpp import DTYPE_TO_CPP
        from .cpp import INDEX_TYPE

        # TODO(jansel): replace this with data from scheduler
        buffer_types = {x.get_name(): x.get_dtype() for x in V.graph.buffers}
        buffer_types.update(
            {name: val.get_dtype() for name, val in V.graph.graph_inputs.items()}
        )

        argdefs = []
        for outer, inner in self.input_buffers.items():
            dtype = buffer_types[outer]
            argdefs.append(f"const {DTYPE_TO_CPP[dtype]}* __restrict__ {inner}")
        for outer, inner in self.output_buffers.items():
            dtype = buffer_types[outer]
            argdefs.append(f"{DTYPE_TO_CPP[dtype]}* __restrict__ {inner}")
        for outer, inner in self.sizevars.items():
            argdefs.append(f"const {INDEX_TYPE} {inner}")
        return argdefs


class CSE:
    """Common subexpression elimination"""

    def __init__(
        self,
        prefix="",
        suffix="",
        name_prefix="tmp",
        iter_buffers=None,
        store_cache=None,
    ):
        self.prefix = prefix
        self.suffix = suffix
        self.cache = {}
        self.name_prefix = name_prefix
        self.store_cache = store_cache or {}
        self.iter_buffer_ids = iter_buffers or itertools.count()

    def clone(self):
        return CSE(
            self.prefix,
            self.suffix,
            self.name_prefix,
            self.iter_buffer_ids,
            self.store_cache,
        )

    def generate(self, buffer: IndentedBuffer, expr: str, write=True):
        if expr.startswith(self.name_prefix) and re.match(r"^[a-z0-9]+$", expr):
            return expr
        if expr not in self.cache:
            var = self.newvar()
            self.cache[expr] = var
            if write:
                buffer.writeline(f"{self.prefix}{var} = {expr}{self.suffix}")
        return self.cache[expr]

    def newvar(self):
        return f"{self.name_prefix}{next(self.iter_buffer_ids)}"


class CodeGen:
    def __init__(self):
        super().__init__()
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        self.exit_stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)


class Kernel(CodeGen):
    newvar_prefix = ""
    suffix = ""
    overrides = None
    load_format = None
    store_format = None

    def __init__(self, args=None):
        super().__init__()
        self.args = args or KernelArgs()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.cse = CSE(self.newvar_prefix, self.suffix)

    @contextlib.contextmanager
    def swap_buffers(self, lb, cb=None, sb=None):
        if cb is None:
            cb = lb
        loads = self.loads
        compute = self.compute
        stores = self.stores
        cse = self.cse
        self.loads = lb
        self.compute = cb
        self.stores = sb
        self.cse = cse.clone()
        yield
        self.loads = loads
        self.compute = compute
        self.stores = stores
        self.cse = cse

    def load(self, name: str, index: sympy.Expr):
        raise NotImplementedError()

    def indirect_load(self, name: str, index: sympy.Expr):
        """A load the depends on an index we have read"""
        prior = self.loads
        try:
            # put the load in the compute section as it might have deps
            self.loads = self.compute
            return self.load(name, index)
        finally:
            self.loads = prior

    def store(self, name, index, value):
        raise NotImplementedError()

    def reduction(self, name, dtype, reduction_type, index, value):
        raise NotImplementedError()

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
                if "tmp" in str(index):
                    return self.indirect_load(name, index)
                store_cache = self.cse.store_cache
                if (name, index) in store_cache:
                    return store_cache[(name, index)]
                if (name, self.rename_indexing(index)) in store_cache:
                    # TODO(jansel): figure out why we need this second case
                    return store_cache[(name, self.rename_indexing(index))]
                return self.load(name, index)

            @staticmethod
            def store(name, index, value):
                self.cse.store_cache[(name, index)] = value
                if name not in V.graph.removed_buffers:
                    return self.store(name, index, value)

            @staticmethod
            def reduction(name, dtype, reduction_type, index, value):
                return self.reduction(name, dtype, reduction_type, index, value)

        super().__enter__()
        parent_handler = self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(CSEProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def rename_indexing(self, index) -> sympy.Expr:
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]
        index = sympy.simplify(index.subs(V.graph.sizevars.replacements))
        subs = {
            x: self.args.size(x) for x in index.free_symbols if str(x).startswith("s")
        }
        return index.subs(subs)
