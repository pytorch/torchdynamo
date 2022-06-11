import contextlib
import dataclasses
import functools
import itertools
from typing import Dict
from typing import List

import sympy
import torch

from .. import codecache
from .. import config
from .. import ir
from ..virtualized import V
from ..virtualized import ops
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import Kernel
from .common import OpOverrides
from .common import product


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


class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    def to_dtype(x, dtype: torch.dtype):
        triton_type_name = str(dtype).split(".")[-1]
        if triton_type_name == "bool":
            triton_type_name = "int1"
        if triton_type_name in ("float16", "bfloat16"):
            triton_type_name = "float32"
        return f"{x}.to(tl.{triton_type_name})"

    @staticmethod
    def constant(value, dtype):
        if value == float("inf"):
            return 'float("inf")'
        elif value == float("-inf"):
            return 'float("-inf")'
        return OpOverrides.constant(value, dtype)

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
    def index_expr(expr, dtype):
        return V.kernel.indexing(expr)[0]

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask) as new_mask:
            result = body()
        return ops.where(
            new_mask, result, TritonOverrides.constant(other, torch.float32)
        )


@dataclasses.dataclass
class RangeTree:
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
    def __init__(self, name: str, numel: sympy.Expr, prefix: str):
        super(RangeTreeRoot, self).__init__(
            name=name,
            var_list=[],
            var_ranges={},
            numel=numel,
            prefix=prefix,
        )

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

    def _codegen_next(self):
        denom = texpr(V.kernel.rename_indexing(self.length))
        V.kernel.indexing_code.writeline(
            f"{self.name}_next = {self.parent.codegen_next()} // {TritonPrinter.paren(denom)}"
        )
        return f"{self.name}_next"

    def _codegen(self):
        denom = texpr(V.kernel.rename_indexing(self.length))
        if self.numel == 1:
            line = f"{self.name} = {self.parent.codegen_next()}"
        else:
            line = f"{self.name} = {self.parent.codegen_next()} % {TritonPrinter.paren(denom)}"
        V.kernel.indexing_code.writeline(line)
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
        names = ["xindex", "yindex", "zindex"][: len(self.numels) - 1] + ["rindex"]
        for i in range(len(self.numels)):
            self.range_trees.append(
                RangeTreeRoot(names[i], self.numels[i], names[i][0])
            )
        self.range_tree_nodes = {}
        self.iter_vars_count = itertools.count()
        self.indexing_code = IndentedBuffer()
        self.inside_reduction = self.numels[-1] != 1
        self.disabled_reduction_stores = {}
        self._load_mask = None

    def disable_reduction(self):
        @contextlib.contextmanager
        def ctx():
            if not self.inside_reduction:
                yield
                return
            prior = self.cse.store_cache
            self.cse.store_cache = self.disabled_reduction_stores
            self.inside_reduction = False
            yield
            self.inside_reduction = True
            self.cse.store_cache = prior

        return ctx()

    def set_ranges(self, *lengths):
        assert len(lengths) == len(self.range_trees)
        return [
            ranges.construct(length)
            for length, ranges in zip(lengths, self.range_trees)
        ]

    def indexing(self, index: sympy.Expr, copy_shape=None):
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        all_vars = list(
            itertools.chain.from_iterable(tree.var_list for tree in self.range_trees)
        )
        offset = index.subs(zero_vars(all_vars))
        parts = []
        for i in range(len(self.range_trees)):
            other_vars = list(
                itertools.chain.from_iterable(
                    self.range_trees[j].var_list
                    for j in range(len(self.range_trees))
                    if i != j
                )
            )
            parts.append(index.subs(zero_vars(other_vars)) - offset)

        assert index == offset + sum(
            parts
        ), f"failed to split indexing into base+reduction {index}"

        offset = self.rename_indexing(self.simplify_indexing(offset))
        parts = [self.rename_indexing(self.simplify_indexing(part)) for part in parts]

        mask = []
        addr = []

        have_dense = True
        need_dense = config.triton.dense_indexing or "tmp" in str(index)

        for part, tree in zip(parts, self.range_trees):
            if part != 0:
                addr.append(texpr(part))
                mask.append(f"{tree.prefix}mask")
            elif tree.prefix == "r" and not self.inside_reduction:
                pass
            else:
                have_dense = False

        if need_dense and not have_dense:
            mask = [
                f"{tree.prefix}mask"
                for tree in self.range_trees
                if tree.prefix != "r" or self.inside_reduction
            ]
            addr.append(f"tl.zeros({self.dense_size_str()}, tl.int32)")

        if offset != 0:
            addr.append(texpr(offset))

        if not addr:
            if copy_shape:
                addr.append(f"tl.zeros({copy_shape}.shape, tl.int32)")
                mask.extend([f"{tree.prefix}mask" for tree in self.range_trees[:-1]])
            else:
                addr.append("0")

        if self._load_mask:
            mask.append(self._load_mask)

        if not mask:
            mask = ["None"]

        return " + ".join(addr), " & ".join(mask)

    def simplify_indexing(self, expr: sympy.Expr):
        expr = V.graph.sizevars.simplify_with_ranges(
            expr,
            dict(
                itertools.chain.from_iterable(
                    tree.var_ranges.items() for tree in self.range_trees
                )
            ),
        )

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
        if (name, index) in self.disabled_reduction_stores:
            return self.disabled_reduction_stores[(name, index)]
        var = self.args.input(name)
        index, mask = self.indexing(index)
        line = f"tl.load({var} + {index}, {mask})"
        if upcast:
            line += ".to(tl.float32)"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value):
        var = self.args.output(name)
        index, mask = self.indexing(index, value)
        line = f"tl.store({var} + {index}, {value}, {mask})"
        self.stores.writeline(line)

    def reduction(self, name, dtype, reduction_type, index, value):
        default = ops.constant(ir.Reduction.default_value(reduction_type, dtype), dtype)
        masks = [f"{tree.prefix}mask" for tree in self.range_trees]
        if self._load_mask:
            masks.append(self._load_mask)
        res = self.cse.generate(
            self.compute,
            f"tl.where({' & '.join(masks)}, {value}, {default})",
        )
        sizes = [f"{tree.prefix.upper()}BLOCK" for tree in self.range_trees]
        sizes[-1] = "1"
        if reduction_type == "any":
            reduction_type = "max"
        res = self.cse.generate(
            self.compute,
            f"tl.reshape(tl.{reduction_type}({res}, {len(self.range_trees) - 1}), [{', '.join(sizes)}])",
        )
        assert self.inside_reduction
        with self.disable_reduction():
            ops.store(name, index, res)

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
        with code.indent():
            for i, tree in enumerate(self.range_trees):
                x = tree.prefix
                if x == "r":
                    if not self.inside_reduction:
                        continue
                    # TODO(jansel): for now reduction must be in a single block
                    code.writeline(f"{x}offset = 0")
                else:
                    code.writeline(f"{x}offset = tl.program_id({i}) * {x.upper()}BLOCK")
                code.writeline(
                    f"{tree.name} = {x}offset + tl.reshape(tl.arange(0, {x.upper()}BLOCK), "
                    f"{self.reshape_size_str(i, x)})"
                )
                code.writeline(f"{x}mask = {tree.name} < {x}numel")

            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")

            code.splice(self.indexing_code)
            code.splice(self.loads)
            code.splice(self.compute)
            code.splice(self.stores)

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
        return tuple(V.graph.sizevars.simplify(product(s)) for s in sizes)

    def codegen(self, *groups):
        wrapper = V.graph.wrapper_code
        scheduler = self.scheduler

        def is_group_matching(other_node):
            other_groups = other_node.group
            if groups == other_groups:
                return True
            if len(groups) == 2 and groups[-1] != 1:
                group, reduction_group = groups
                if other_groups == (group * reduction_group, sympy.Integer(1)):
                    sizes, _ = node.get_ranges()
                    split = split_sizes(sizes, group, reduction_group)
                    return split is not None
                return other_groups == (group, sympy.Integer(1))
            elif len(groups) == 3:
                tile1, tile2, _ = groups
                if other_groups == (tile1 * tile2, sympy.Integer(1)):
                    sizes, _ = node.get_ranges()
                    split = split_sizes(sizes, tile1, tile2)
                    return split is not None

            return False

        reschedule = []
        with scheduler.kernel(TritonKernel(*groups)) as kernel:
            for _ in scheduler.iter_fixed_point():
                for node in scheduler.pop_group(groups):
                    scheduler.maybe_remove_buffer(node, check_group=is_group_matching)
                    node.run(*kernel.set_ranges(*node.get_ranges()))
                    node.mark_fusable(broadcast_after_reduce=True)

                # the rest of this function could be correctly removed
                # it is various cases of horizonal fusions
                if kernel.inside_reduction:
                    # TODO(jansel): rewrite this to support tiled reductions
                    group, reduction_group = groups

                    # Add pointwise with compatible dimensions
                    for node in scheduler.pop_group(
                        (group * reduction_group, sympy.Integer(1)),
                    ):
                        sizes, _ = node.get_ranges()
                        split = split_sizes(sizes, group, reduction_group)
                        if split is None:
                            reschedule.append(node)
                        else:
                            scheduler.maybe_remove_buffer(
                                node, check_group=is_group_matching
                            )
                            node.run(*kernel.set_ranges(sizes[:split], sizes[split:]))
                            node.mark_fusable()

                    # Add more pointwise with fewer dimensions
                    with kernel.disable_reduction():
                        for node in scheduler.pop_group((group, sympy.Integer(1))):
                            scheduler.maybe_remove_buffer(
                                node, check_group=is_group_matching
                            )
                            node.run(*kernel.set_ranges(*node.get_ranges()))
                            node.mark_fusable()
                elif len(groups) == 3:
                    tile1, tile2, _ = groups
                    # Add pointwise with compatible dimensions
                    for node in scheduler.pop_group(
                        (tile1 * tile2, sympy.Integer(1)),
                    ):
                        sizes, _ = node.get_ranges()
                        split = split_sizes(sizes, tile1, tile2)
                        if split is None:
                            reschedule.append(node)
                        else:
                            scheduler.maybe_remove_buffer(
                                node, check_group=is_group_matching
                            )
                            node.run(
                                *kernel.set_ranges(sizes[:split], sizes[split:], [])
                            )
                            node.mark_fusable()

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


def split_sizes(sizes, prod1, prod2):
    for i in range(len(sizes)):
        if product(sizes[:i]) == prod1 and product(sizes[i:]) == prod2:
            return i
    return None
