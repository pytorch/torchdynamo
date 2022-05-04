import contextlib
import dataclasses
import functools
import itertools
from itertools import chain
from typing import Dict
from typing import List

import sympy
import torch

from .. import codecache
from .. import config
from .. import ir
from ..scheduler import Scheduler
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
        return f"{x}.to(tl.{triton_type_name})"

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
        return f"tl.log({x})"

    @staticmethod
    def relu(x):
        return ops.maximum("0", x)

    @staticmethod
    def minimum(a, b):
        return f"tl.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"tl.maximum({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"tl.where({a}, {b}, {c})"

    @staticmethod
    def index_expr(expr, dtype):
        return V.kernel.indexing(expr)

    @staticmethod
    def masked(mask, body, other):
        if other == float("-inf"):
            other = 'float("-inf")'
        else:
            assert False, other
        with V.kernel.mask_loads(mask) as new_mask:
            result = body()
        return ops.where(new_mask, result, other)

    @staticmethod
    def logical_not(a):
        return f"~{a}"


@dataclasses.dataclass
class RangeTree:
    def __init__(
        self,
        name: str,
        var_list: List[sympy.Symbol],
        numel: sympy.Expr,
        prefix: str,
        depth=0,
        length=sympy.Integer(1),
    ):
        super(RangeTree, self).__init__()
        self.name = name
        self.children: Dict[sympy.Expr, RangeTreeEntry] = dict()
        self.var_list = var_list
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
        else:
            node = self.children[length]
        return node


class RangeTreeRoot(RangeTree):
    def __init__(self, name: str, numel: sympy.Expr, prefix: str):
        super(RangeTreeRoot, self).__init__(
            name=name,
            var_list=[],
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


class TritonKernel(Kernel):
    overrides = TritonOverrides
    sexpr = texpr

    def __init__(self, numel, reduction_numel):
        super(TritonKernel, self).__init__()
        if reduction_numel is None:
            reduction_numel = sympy.Integer(1)
        self.numel = numel
        self.reduction_numel = reduction_numel
        self.iter_range_tree = RangeTreeRoot("indices", numel, "i")
        self.reduction_range_tree = RangeTreeRoot("reduction", reduction_numel, "r")
        self.range_tree_nodes = dict()
        self.iter_vars_count = itertools.count()
        self.indexing_code = IndentedBuffer()
        self.inside_reduction = reduction_numel != 1
        self.disabled_reduction_stores = dict()
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

    def set_ranges(self, lengths, reduction_lengths):
        return self.iter_range_tree.construct(
            lengths
        ), self.reduction_range_tree.construct(reduction_lengths)

    def indexing(self, index: sympy.Expr):
        iter_vars = self.iter_range_tree.var_list
        reduction_vars = self.reduction_range_tree.var_list

        offset = index.subs({v: 0 for v in chain(iter_vars, reduction_vars)})
        base_part = index.subs({v: 0 for v in reduction_vars}) - offset
        reduction_part = index.subs({v: 0 for v in iter_vars}) - offset

        assert index == offset + base_part + reduction_part

        offset = self.rename_indexing(offset)
        base_part = self.rename_indexing(self.simplify_indexing(base_part))
        reduction_part = self.rename_indexing(self.simplify_indexing(reduction_part))

        addr = []
        if offset != 0:
            addr.append(texpr(offset))

        if base_part != 0:
            addr.append(texpr(base_part))
        else:
            addr.append("tl.zeros((BLOCK_SIZE, ), tl.int32)")

        if self.inside_reduction:
            addr[-1] = f"tl.reshape({addr[-1]}, (BLOCK_SIZE, 1))"

            if reduction_part != 0:
                addr.append(texpr(reduction_part))
            else:
                addr.append("tl.zeros((REDUCTION_SIZE, ), tl.int32)")

            addr[-1] = f"tl.reshape({addr[-1]}, (1, REDUCTION_SIZE))"
        else:
            assert reduction_part == 0

        return " + ".join(addr)

    def simplify_indexing(self, expr: sympy.Expr):
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

    def mask(self, reductions=True):
        if self._load_mask:
            return self._load_mask
        return (
            "mask=mask_reduction"
            if (self.inside_reduction and reductions)
            else "mask=mask"
        )

    @contextlib.contextmanager
    def mask_loads(self, mask):
        assert self._load_mask is None, "TODO: nesting"
        var = self.cse.newvar()
        if self.inside_reduction:
            old_mask = "mask_reduction"
        else:
            old_mask = "mask"
        self.compute.splice(
            f"""
                if NEED_MASK:
                    {var} = {old_mask} & {mask}
                else:
                    {var} = {mask}
            """
        )
        prior = self._load_mask
        self._load_mask = var
        with self.swap_buffers(self.compute, self.compute):
            yield var
        self._load_mask = prior

    def load(self, name: str, index: sympy.Expr):
        if (name, index) in self.disabled_reduction_stores:
            tmpvar = self.disabled_reduction_stores[(name, index)]
            return f"tl.reshape({tmpvar}, (BLOCK_SIZE, 1))"
        var = self.args.input(name)
        line = f"tl.load({var} + {self.indexing(index)}, {self.mask()})"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value):
        var = self.args.output(name)
        line = f"tl.store({var} + {self.indexing(index)}, {value}, {self.mask()})"
        self.stores.writeline(line)

    def reduction(self, name, dtype, reduction_type, index, value):
        default = {"sum": 0, "max": "float('-inf')", "min": "float('inf')"}
        res = self.cse.generate(
            self.compute,
            f"tl.where(mask_reduction, {value}, "
            f"{default[reduction_type]}) if NEED_MASK else {value}",
        )
        res = self.cse.generate(self.compute, f"tl.{reduction_type}({res}, 1)")
        assert self.inside_reduction
        with self.disable_reduction():
            ops.store(name, index, res)

    def codegen_kernel(self):
        code = IndentedBuffer()
        heuristics = (
            "reduction_heuristics" if self.inside_reduction else "pointwise_heuristics"
        )
        code.splice(
            f"""
                import triton
                import triton.language as tl
                from {codecache.__name__} import reduction_heuristics, pointwise_heuristics

                @triton.heuristics({heuristics}())
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

        if config.dynamic_shapes:
            maybe_const = ""
        else:
            maybe_const = ": tl.constexpr"

        if self.inside_reduction:
            argdefs += [
                f"numel{maybe_const}",
                f"reduction_numel{maybe_const}",
                "BLOCK_SIZE: tl.constexpr",
                "REDUCTION_SIZE: tl.constexpr",
                "NEED_MASK: tl.constexpr",
            ]

        else:
            argdefs += [
                f"numel{maybe_const}",
                "BLOCK_SIZE: tl.constexpr",
                "NEED_MASK: tl.constexpr",
            ]

        code.writeline(f"def kernel({', '.join(argdefs)}):")
        with code.indent():
            code.splice(
                """
                    offset = tl.program_id(0) * BLOCK_SIZE
                    indices = offset + tl.arange(0, BLOCK_SIZE)
                """,
                strip=True,
            )

            if self.inside_reduction:
                code.splice(
                    """
                        reduction = tl.arange(0, REDUCTION_SIZE)
                        if NEED_MASK:
                            mask = indices < numel
                            mask_reduction = (tl.reshape(mask, (BLOCK_SIZE, 1)) &
                                              tl.reshape(reduction < reduction_numel, (1, REDUCTION_SIZE)))
                        else:
                            mask : tl.constexpr = None
                            mask_reduction : tl.constexpr = None
                    """,
                    strip=True,
                )
            else:
                code.splice(
                    """
                    if NEED_MASK:
                        mask = indices < numel
                    else:
                        mask : tl.constexpr = None
                """,
                    strip=True,
                )
            code.splice(self.indexing_code)
            code.splice(self.loads)
            code.splice(self.compute)
            code.splice(self.stores)

        wrapper = IndentedBuffer()
        wrapper.writeline("TritonCodeCache.load('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''').kernel")
        return wrapper.getvalue()

    def call_kernel(self, schedule, name: str):
        code = schedule.body
        call_args = list(
            chain(
                self.args.input_buffers.keys(),
                self.args.output_buffers.keys(),
                self.args.sizevars.keys(),
            )
        )
        code.writeline(f"{name}_numel = {texpr(self.numel)}")
        call_args.append(f"{name}_numel")
        if self.inside_reduction:
            code.writeline(f"{name}_reduction_numel = {texpr(self.reduction_numel)}")
            call_args.append(f"{name}_reduction_numel")
        code.writeline(f"{name}[grid({name}_numel)](")
        with code.indent():
            code.writeline(", ".join(call_args))
        code.writeline(")")

    @classmethod
    def codegen(cls, wrapper):
        def codegen_extern_call(node):
            assert isinstance(node, ir.ExternKernel)
            node.codegen(wrapper)
            scheduler.barrier()

        scheduler = Scheduler(product, V.graph.buffers)

        for group, reduction_group in scheduler.iter_runable_groups(
            codegen_extern_call
        ):
            reschedule = []
            with scheduler.kernel(TritonKernel(group, reduction_group)) as kernel:
                for _ in scheduler.iter_fixed_point():
                    for node in scheduler.pop_group(
                        (group, reduction_group),
                    ):
                        scheduler.maybe_remove_buffer(node, broadcast_after_reduce=True)
                        node.run(*kernel.set_ranges(*node.get_ranges()))
                        node.mark_fusable(broadcast_after_reduce=True)

                    if kernel.inside_reduction:
                        # Add pointwise with compatible dimensions
                        for node in scheduler.pop_group(
                            (group * reduction_group, sympy.Integer(1)),
                        ):
                            sizes, _ = node.get_ranges()
                            split = split_sizes(sizes, group, reduction_group)
                            if split is None:
                                reschedule.append(node)
                            else:
                                node.run(
                                    *kernel.set_ranges(sizes[:split], sizes[split:])
                                )
                                node.mark_fusable()

                        # Add more pointwise with fewer dimensions
                        with kernel.disable_reduction():
                            for node in scheduler.pop_group((group, sympy.Integer(1))):
                                node.run(*kernel.set_ranges(*node.get_ranges()))
                                node.mark_fusable()

            kernel_name = wrapper.next_kernel_name()
            wrapper.define_kernel(kernel_name, kernel.codegen_kernel())
            kernel.call_kernel(wrapper, kernel_name)

            scheduler.enqueue(reschedule)
            scheduler.barrier()


def split_sizes(sizes, prod1, prod2):
    for i in range(len(sizes)):
        if product(sizes[:i]) == prod1 and product(sizes[i:]) == prod2:
            return i
    return None
