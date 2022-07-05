import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
from typing import Dict
from typing import List

import sympy
import torch

from .. import codecache
from .. import config
from .. import ir
from ..utils import sympy_product
from ..virtualized import V
from ..virtualized import ops
from .common import DeferredLine
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import Kernel
from .common import OpOverrides

log = logging.getLogger(__name__)


class TritonPrinter(ExprPrinter):
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} // {div})"
        return f"{x} % {mod}"

    def _print_IndexingDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} // {div})"


texpr = TritonPrinter().doprint


def triton_compute_type(dtype):
    triton_type_name = str(dtype).split(".")[-1]
    if triton_type_name == "bool":
        triton_type_name = "int1"
    if triton_type_name in ("float16", "bfloat16"):
        # float16 math is done in float32 inside the kernel
        triton_type_name = "float32"
    return f"tl.{triton_type_name}"


def triton_constant(value):
    if value == float("inf"):
        return 'float("inf")'
    elif value == float("-inf"):
        return 'float("-inf")'
    elif math.isnan(value):
        return 'float("nan")'
    return repr(value)


class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    def to_dtype(x, dtype: torch.dtype):
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"{x}.to({triton_compute_type(dtype)})"

    @staticmethod
    def constant(value, dtype):
        return triton_constant(value)

    @staticmethod
    def abs(x):
        return f"tl.abs({x})"

    @staticmethod
    def exp(x):
        return f"tl.exp({x})"

    @staticmethod
    def sqrt(x):
        return f"tl.sqrt({x})"

    @staticmethod
    def log(x):
        # workaround https://github.com/openai/triton/issues/543
        return f"tl.log({x}.to(tl.float32))"

    @staticmethod
    def isinf(x):
        return f"{x}+1 == {x}"

    @staticmethod
    def isnan(x):
        return f"{x} != {x}"

    @staticmethod
    def relu(x):
        return ops.maximum("0", x)

    @staticmethod
    def round(x):
        return f"tl.where({x}<0, {x}-0.5, {x}+0.5).to(tl.int32).to(tl.float32)"

    @staticmethod
    def floor(x):
        tmp = ops.trunc(x)
        return f"tl.where({tmp}>{x}, {tmp}-1, {tmp})"

    @staticmethod
    def trunc(x):
        return f"{x}.to(tl.int32).to(tl.float32)"

    @staticmethod
    def minimum(a, b):
        return f"tl.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"tl.maximum({a}, {b})"

    @staticmethod
    def where(a, b, c):
        # wonkyness to work around https://github.com/openai/triton/issues/532
        # identity calls to force new triton variables (and get access to .shape/.dtype/.numel
        a = ops.identity(a)
        b = ops.identity(b)
        c = ops.identity(c)
        a = ops.identity(
            f"{a} | tl.zeros({b}.shape, {a}.dtype) if {b}.numel > 1 else {a}"
        )
        a = ops.identity(
            f"{a} | tl.zeros({c}.shape, {a}.dtype) if {c}.numel > 1 else {a}"
        )
        return f"tl.where({a}, {b}, {c})"

    @staticmethod
    def cos(x):
        return f"tl.cos({x})"

    @staticmethod
    def index_expr(expr, dtype):
        return V.kernel.indexing(expr)[0]

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask) as new_mask:
            result = body()
        return ops.where(
            new_mask, result, TritonOverrides.constant(other, torch.float32)
        )

    @staticmethod
    def rand(seed, offset):
        return f"tl.rand({seed}, {offset})"


@dataclasses.dataclass
class RangeTree:
    """
    Each range tree represents multiple sets of iteration indexing
    in a single tiled dimension in the output kernel.

    If you have two loops ranges one (4, 3, 2) and another (4, 6),
    then the range tree will be:
            4 (i0)
        3 (i1)  6 (i3)
        2 (i2)
    Where i0 is shared between both loops, but then the split into
    different indexing vars.  All loop ranges must iterate over
    the same number of elements.
    """

    def __init__(
        self,
        name: str,
        var_list: List[sympy.Symbol],
        var_ranges: Dict[sympy.Symbol, sympy.Expr],
        numel: sympy.Expr,
        prefix: str,
        depth=0,
        length=sympy.Integer(1),
    ):
        super(RangeTree, self).__init__()
        self.name = name
        self.children: Dict[sympy.Expr, RangeTreeEntry] = {}
        self.var_list = var_list
        self.var_ranges = var_ranges
        self.numel = numel
        self.prefix = prefix
        self.depth = depth
        self.length = length

    def cache_clear(self):
        for child in self.children.values():
            child.cache_clear()

    def is_loop(self):
        return self.prefix == "r"

    def child_node(self, length):
        if length not in self.children:
            node = RangeTreeEntry(
                f"{self.prefix}{next(V.kernel.iter_vars_count)}", length, self
            )
            self.children[length] = node
            V.kernel.range_tree_nodes[node.symbol()] = node
            self.var_list.append(node.symbol())
            self.var_ranges[node.symbol()] = length
        else:
            node = self.children[length]
        return node


class RangeTreeRoot(RangeTree):
    def __init__(
        self, name: str, numel: sympy.Expr, prefix: str, index: int, kernel: "Kernel"
    ):
        super(RangeTreeRoot, self).__init__(
            name=name,
            var_list=[],
            var_ranges={},
            numel=numel,
            prefix=prefix,
        )
        self.index = index
        self.kernel = kernel

    def codegen_next(self):
        return self.name

    def simplify(self, expr, nodes):
        return expr

    def construct(self, lengths):
        node = self
        itervars = []
        for sv in reversed(lengths):
            node = node.child_node(sv)
            itervars.append(node.symbol())
        return list(reversed(itervars))

    def ranges_code(self):
        size = self.kernel.reshape_size_str(self.index, self.prefix)
        return f"tl.reshape(tl.arange(0, {self.prefix.upper()}BLOCK), {size})"

    def codegen_header(self, code):
        x = self.prefix
        if self.is_loop():
            code.writeline(f"{self.name} = {x}offset + {x}base")
        else:
            code.writelines(
                [
                    f"{x}offset = tl.program_id({self.index}) * {x.upper()}BLOCK",
                    f"{self.name} = {x}offset + {self.ranges_code()}",
                ]
            )
        code.writeline(f"{x}mask = {self.name} < {x}numel")


class RangeTreeEntry(RangeTree):
    def __init__(self, name: str, length: sympy.Expr, parent: RangeTree):
        super(RangeTreeEntry, self).__init__(
            name=name,
            numel=parent.numel / length,
            var_list=parent.var_list,
            var_ranges=parent.var_ranges,
            prefix=parent.prefix,
            length=length,
            depth=parent.depth + 1,
        )
        self.parent = parent
        self.codegen = functools.lru_cache(None)(self._codegen)
        self.codegen_next = functools.lru_cache(None)(self._codegen_next)

    def cache_clear(self):
        self.codegen.cache_clear()
        self.codegen_next.cache_clear()
        super().cache_clear()

    def writeline(self, line):
        if self.is_loop():
            V.kernel.indexing_code.writeline(line)
        else:
            # lift non-reduction stores outside loop
            V.kernel.body.writeline(line)

    def _codegen_next(self):
        denom = texpr(V.kernel.rename_indexing(self.length))
        self.writeline(
            f"{self.name}_next = {self.parent.codegen_next()} // {TritonPrinter.paren(denom)}"
        )
        return f"{self.name}_next"

    def _codegen(self):
        denom = texpr(V.kernel.rename_indexing(self.length))
        if self.numel == 1:
            line = f"{self.name} = {self.parent.codegen_next()}"
        else:
            line = f"{self.name} = {self.parent.codegen_next()} % {TritonPrinter.paren(denom)}"
        self.writeline(line)
        return self.name

    def symbol(self):
        return sympy.Symbol(self.name)

    def next_symbol(self):
        return sympy.Symbol(f"{self.name}_next")

    def simplify(self, expr: sympy.Expr):
        """
        Merge the indexing math for contiguous dimensions
        """
        if isinstance(self.parent, RangeTreeRoot):
            return expr

        v1 = sympy.Symbol("subs_var1")
        v2 = sympy.Symbol("subs_var2")
        pl = self.parent.length
        test_expr = sympy.expand(
            expr.subs({self.name: 0, self.parent.name: (v1 * pl + v2)}).subs(
                {
                    v1: self.name,
                    v2: self.parent.name,
                }
            )
        )
        if test_expr == sympy.expand(expr):
            # we can compact this dimension into a a single one
            node = self.parent.parent.child_node(self.length * self.parent.length)
            new_expr = expr.subs({self.name: 0, self.parent.name: node.symbol()})
            return node.simplify(new_expr)

        return expr

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


def zero_vars(it):
    return {k: 0 for k in it}


class TritonKernel(Kernel):
    overrides = TritonOverrides
    sexpr = texpr

    def __init__(self, *groups):
        super(TritonKernel, self).__init__()
        self.numels = [V.graph.sizevars.simplify(s) for s in groups]
        self.range_trees = []
        self.range_tree_nodes = {}
        self.iter_vars_count = itertools.count()
        self.inside_reduction = self.numels[-1] != 1
        self._load_mask = None
        self.body = IndentedBuffer()
        self.indexing_code = IndentedBuffer()
        self.suffix = IndentedBuffer()
        self.outside_loop_vars = set()
        self.initialize_range_tree()

    def initialize_range_tree(self):
        names = ["xindex", "yindex", "zindex"][: len(self.numels) - 1] + ["rindex"]
        for i in range(len(self.numels)):
            self.range_trees.append(
                RangeTreeRoot(names[i], self.numels[i], names[i][0], i, self)
            )
        for tree in self.range_trees:
            # reduction indexing goes inside a loop
            if tree.prefix != "r":
                tree.codegen_header(self.body)
        if self.inside_reduction and self.range_trees[-1].is_loop():
            # workaround for this issue:
            # https://gist.github.com/jansel/6527126f781559095c5531f98a4235a7
            self.body.writeline(f"rbase = {self.range_trees[-1].ranges_code()}")

    def disable_reduction(self):
        @contextlib.contextmanager
        def ctx():
            if not self.inside_reduction:
                yield
                return
            # calling codegen_body() will flush all the pending buffers
            # and write out a reduction loop
            self.codegen_body()
            self.inside_reduction = False
            yield
            self.inside_reduction = True

        return ctx()

    def set_ranges(self, *lengths):
        assert len(lengths) == len(self.range_trees)
        return [
            ranges.construct(length)
            for length, ranges in zip(lengths, self.range_trees)
        ]

    def split_and_set_ranges(self, lengths: List[List[sympy.Expr]]):
        """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
        sv = V.graph.sizevars
        new_ranges = [[] for _ in self.range_trees]
        remaining = [sv.simplify(rt.numel) for rt in self.range_trees]
        var_count = itertools.count()

        def add_range(i, expr):
            expr = sv.simplify(expr)
            if not sv.maybe_guard_multiple_of(remaining[i], expr):
                raise CantSplit()
            # guard on the last item out
            sv.maybe_guard_equals(remaining[i], expr)
            remaining[i] = ir.IndexingDiv(remaining[i], expr)
            new_ranges[i].append(expr)
            return next(var_count)

        def make_combined(size, idx1, idx2):
            def getter(flat_vars):
                return size * flat_vars[idx1] + flat_vars[idx2]

            return getter

        return_getters_groups = []
        current_group = 0
        for length_group in lengths:
            return_getters = []
            for size in length_group:
                while (
                    current_group < len(remaining)
                    and sv.size_hint(remaining[current_group]) == 1
                ):
                    # scroll to next group with remaining elements
                    current_group += 1

                if sv.size_hint(size) > sv.size_hint(remaining[current_group]):
                    # need to break size in two
                    if not sv.maybe_guard_multiple_of(size, remaining[current_group]):
                        raise CantSplit()
                    size1 = remaining[current_group]
                    size2 = ir.IndexingDiv(size, remaining[current_group])
                    return_getters.append(
                        make_combined(
                            size2,
                            add_range(current_group, size1),
                            add_range(current_group + 1, size2),
                        )
                    )
                else:
                    return_getters.append(
                        operator.itemgetter(add_range(current_group, size))
                    )

            return_getters_groups.append(return_getters)

        assert all(s == 1 for s in remaining)
        itervars = list(itertools.chain(*self.set_ranges(*new_ranges)))
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    def indexing(self, index: sympy.Expr, copy_shape=None):
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        index_vars = set(index.free_symbols)
        index_str = texpr(self.rename_indexing(self.simplify_indexing(index)))

        need_dense = (
            config.triton.dense_indexing
            or any(
                # tmpX  means indirect indexing
                str(v).startswith("tmp")
                for v in index_vars
            )
        ) and index != 0
        have_dense = True
        have_loop_vars = False
        mask = []
        dense_mask = []

        for tree in self.range_trees:
            if tree.prefix == "r" and not self.inside_reduction:
                continue
            if index_vars.intersection(tree.var_list):
                have_loop_vars = True
                have_dense = False
                mask.append(f"{tree.prefix}mask")
            dense_mask.append(f"{tree.prefix}mask")

        if need_dense and not have_dense:
            mask = dense_mask
            index_str = f"{index_str} + tl.zeros({self.dense_size_str()}, tl.int32)"
        elif not have_loop_vars and copy_shape:
            mask = dense_mask
            index_str = f"{index_str} + tl.zeros({copy_shape}.shape, tl.int32)"

        if self._load_mask:
            mask.append(self._load_mask)
        elif not mask:
            mask = ["None"]

        return index_str, " & ".join(mask)

    def var_ranges(self):
        return (
            dict(
                itertools.chain.from_iterable(
                    tree.var_ranges.items() for tree in self.range_trees
                )
            ),
        )

    def simplify_indexing(self, expr: sympy.Expr):
        expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
        nodes = [
            self.range_tree_nodes[sym]
            for sym in expr.free_symbols
            if sym in self.range_tree_nodes
        ]
        if nodes:
            nodes.sort(key=lambda x: x.depth)
            expr = nodes[-1].simplify(expr)

        for sym in expr.free_symbols:
            if sym in self.range_tree_nodes:
                self.range_tree_nodes[sym].codegen()

        return expr

    @contextlib.contextmanager
    def mask_loads(self, mask):
        """Context manager to add an additional mask to tl.load/store"""
        assert self._load_mask is None, "TODO: nesting"
        prior = self._load_mask
        self._load_mask = mask
        with self.swap_buffers(self.compute, self.compute):
            # TODO(jansel): do we need a reshape here?
            yield mask
        self._load_mask = prior

    def load(self, name: str, index: sympy.Expr, upcast: bool = False):
        var = self.args.input(name)
        index, mask = self.indexing(index)
        line = f"tl.load({var} + {index}, {mask})"
        if upcast:
            line += ".to(tl.float32)"

        if self.inside_reduction and "rmask" not in mask:
            # can lift a common load outside of reduction loop
            tmp = self.cse.generate(self.body, line)
        else:
            tmp = self.cse.generate(self.loads, line)

        if not self.inside_reduction or "rmask" not in mask:
            self.outside_loop_vars.add(tmp)

        return tmp

    def store(self, name, index, value, mode=None):
        var = self.args.output(name)
        index, mask = self.indexing(index, value)
        if mode is None:
            line = f"tl.store({var} + {index}, {value}, {mask})"
        elif mode == "atomic_add":
            line = f"tl.atomic_add({var} + {index}, {value}, {mask})"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(name, line)
        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

    def reduction(self, name, dtype, reduction_type, index, value):
        assert self.inside_reduction
        default = triton_constant(ir.Reduction.default_value(reduction_type, dtype))
        masks = [f"{tree.prefix}mask" for tree in self.range_trees]
        if self._load_mask:
            masks.append(self._load_mask)
        sizes = [f"{tree.prefix.upper()}BLOCK" for tree in self.range_trees]
        sizes[-1] = "1"
        if reduction_type == "any":
            reduction_type = "max"

        dim = len(self.range_trees) - 1
        result_var = self.cse.newvar()
        if (dtype, reduction_type, value) not in self.cse.reduction_cache:
            self.cse.reduction_cache[(dtype, reduction_type, value)] = result_var
            accumulator = f"_{result_var}"
            self.body.writeline(
                f"{accumulator} = tl.zeros({self.dense_size_str()}, {triton_compute_type(dtype)}) + {default}"
            )

            updated = value
            if reduction_type == "min":
                masks.append(f"({accumulator} > {value})")
            elif reduction_type == "max":
                masks.append(f"({accumulator} < {value})")
            elif reduction_type == "sum":
                updated = f"{accumulator} + {value}"
            else:
                raise NotImplementedError(f"reduction_type {reduction_type}")

            cond = " & ".join(masks)
            self.compute.writeline(
                f"{accumulator} = tl.where({cond}, {updated}, {accumulator})"
            )

            self.suffix.writeline(
                f"{result_var} = tl.reshape(tl.{reduction_type}({accumulator}, {dim}), [{', '.join(sizes)}])"
            )
        else:
            var_name = self.cse.reduction_cache[(dtype, reduction_type, value)]
            self.suffix.writeline(f"{result_var} = {var_name}")
        self.inside_reduction = False
        index, mask = self.indexing(index, result_var)
        assert "rmask" not in index
        self.inside_reduction = True
        self.outside_loop_vars.add(result_var)
        self.cse.store_cache[name] = result_var
        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            self.suffix.writeline(
                DeferredLine(name, f"tl.store({var} + {index}, {result_var}, {mask})")
            )

    def codegen_body(self):
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
        if not (
            self.indexing_code
            or self.loads
            or self.stores
            or self.compute
            or self.suffix
        ):
            return

        if self.inside_reduction:
            self.body.writeline("for roffset in range(0, rnumel, RBLOCK):")
            with self.body.indent():
                # last range tree is always reduction
                self.range_trees[-1].codegen_header(self.body)
                self.body.splice(self.indexing_code)
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)

            # invalidate any caches that came from inside the reduction loop
            self.cse.invalidate(self.outside_loop_vars)
            self.range_trees[-1].cache_clear()
        else:
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.body.splice(self.suffix)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.suffix.clear()

    def codegen_kernel(self, name=None):
        from triton import next_power_of_2

        code = IndentedBuffer()
        size_hints = [
            next_power_of_2(V.graph.sizevars.size_hint(numel)) for numel in self.numels
        ]
        if self.inside_reduction:
            heuristics = "reduction_heuristics"
        else:
            heuristics = "pointwise_heuristics"
            size_hints = size_hints[:-1]

        if name is None:
            code.splice(
                f"""
                    import triton
                    import triton.language as tl
                    from {codecache.__name__} import {heuristics}

                """
            )

        code.splice(
            f"""
                @{heuristics}(size_hints={size_hints!r})
                @triton.jit
            """
        )

        argdefs, _ = self.args.python_argdefs()

        if config.dynamic_shapes:
            maybe_const = ""
        else:
            maybe_const = ": tl.constexpr"

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                argdefs.append(f"{tree.prefix}numel{maybe_const}")

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        code.writeline(f"def {name or 'kernel'}({', '.join(argdefs)}):")
        self.codegen_body()
        with code.indent():
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if name is not None:
            return code.getvalue()

        wrapper = IndentedBuffer()
        wrapper.writeline("TritonCodeCache.load('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''').kernel")
        return wrapper.getvalue()

    def reshape_size_str(self, i=None, x=None):
        sizes = ["1"] * (len(self.range_trees) - int(self.numels[-1] == 1))
        if i is not None:
            sizes[i] = f"{x.upper()}BLOCK"
        return f"[{', '.join(sizes)}]"

    def dense_size_str(self):
        sizes = []
        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                sizes.append(f"{tree.prefix.upper()}BLOCK")
            elif tree.prefix == "r" and tree.numel != 1:
                sizes.append("1")
        return f"[{', '.join(sizes)}]"

    def call_kernel(self, code, name: str):
        _, call_args = self.args.python_argdefs()
        grid = []
        # TODO(jansel): if there are constants, we shouldn't bother passing them as args
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = texpr(tree.numel)
            else:
                expr = f"{name}_{tree.prefix}numel"
                code.writeline(f"{expr} = {texpr(tree.numel)}")
            if tree.prefix != "r" or self.inside_reduction:
                call_args.append(expr)
            if tree.prefix != "r":
                grid.append(expr)
        call_args = ", ".join(call_args)
        code.writeline(f"{name}[grid({', '.join(grid)})]({call_args})")


class TritonScheduling:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    def codegen(self, *groups):
        wrapper = V.graph.wrapper_code
        scheduler = self.scheduler

        reduction_nodes = []
        reschedule = []
        with scheduler.kernel(TritonKernel(*groups)) as kernel:
            for _ in scheduler.iter_fixed_point():
                # scheduler.pop_group will keep iterating all reachable fusable nodes
                for node in scheduler.pop_group(groups):
                    node.run(*kernel.set_ranges(*node.get_ranges()))
                    if kernel.inside_reduction:
                        reduction_nodes.append(node)
                    else:
                        node.mark_fusable()

                # the rest of this function could be correctly removed
                # it is various cases of horizonal fusions
                if kernel.inside_reduction:
                    # TODO(jansel): rewrite this to support tiled reductions
                    group, reduction_group = groups

                    # Add pointwise with compatible dimensions
                    for node in scheduler.pop_group(
                        (group * reduction_group, sympy.Integer(1)),
                    ):
                        try:
                            node.run(*kernel.split_and_set_ranges(node.get_ranges()))
                            node.mark_fusable()
                        except CantSplit:
                            reschedule.append(node)

                    # we mark reductions fusable here as they rely on the loop break below
                    for node in reduction_nodes:
                        node.mark_fusable(broadcast_after_reduce=True)
                    reduction_nodes.clear()

                    # Add more pointwise with fewer dimensions
                    # disable_reduction() will close the current reduction loop
                    with kernel.disable_reduction():
                        for node in scheduler.pop_group((group, sympy.Integer(1))):
                            node.run(*kernel.set_ranges(*node.get_ranges()))
                            node.mark_fusable()

                elif len(groups) == 3:
                    tile1, tile2, _ = groups
                    # Add pointwise with compatible dimensions
                    for node in scheduler.pop_group(
                        (tile1 * tile2, sympy.Integer(1)),
                    ):
                        try:
                            node.run(*kernel.split_and_set_ranges(node.get_ranges()))
                            node.mark_fusable()
                        except CantSplit:
                            reschedule.append(node)

        kernel_name = wrapper.next_kernel_name()
        if config.triton.many_files:
            wrapper.define_kernel(kernel_name, kernel.codegen_kernel())
        else:
            wrapper.header.splice(kernel.codegen_kernel(kernel_name))
        kernel.call_kernel(wrapper, kernel_name)

        scheduler.enqueue(reschedule)
        scheduler.barrier()
        scheduler.maybe_free_buffers()

    def flush(self):
        pass


class CantSplit(Exception):
    pass
