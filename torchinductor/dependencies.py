import collections
import dataclasses
import itertools
import typing
from typing import List
from typing import Set

import sympy

from .codegen.common import _simplify_loops
from .virtualized import V


class MemoryDep(typing.NamedTuple):
    name: str
    index: sympy.Expr
    size: List[sympy.Expr]

    def broadcast_extend_sizes(self, extra_sizes):
        size = (*self.size, *[x for x in extra_sizes if x != 1])
        return MemoryDep(self.name, self.index, size)

    def maybe_swap_sizes(self):
        # swap only in simple cases where index is trivial and
        # there are just 2 sizes
        if (
            len(self.size) == 2
            and len(self.index.args) == 0
            and self.index.name == canonicalization_prefix() + "0"
        ):
            c = canonicalization_prefix()
            size = (self.size[1], self.size[0])
            s0 = sympy.Symbol(c + "0", is_integer=True)
            s1 = sympy.Symbol(c + "1", is_integer=True)
            index = self.index.subs(s0, s1)
            return MemoryDep(self.name, index, size)
        else:
            return self

    def strip_last_size(self):
        nsizes = len(self.size)
        if nsizes >= 1 and len(self.index.args) <= nsizes - 1:
            # make sure last dim index is not used
            prefix = canonicalization_prefix()
            len_prefix = len(prefix)
            prefixes = [fs.name[:len_prefix] for fs in self.index.free_symbols]
            assert (
                len(prefixes) == 0 or prefix in prefixes
            ), "index expression should contain canonicalized symbols"
            last_index = f"{prefix}{len(self.size)-1}"
            if last_index not in self.index.free_symbols:
                size = self.size[:-1]
                return MemoryDep(self.name, self.index, size)
            else:
                return self

    def rename(self, renames):
        if self.name in renames:
            return MemoryDep(renames[self.name], self.index, self.size)
        return self


class StarDep(typing.NamedTuple):
    # depends on the entire buffer
    name: str

    def rename(self, renames):
        if self.name in renames:
            return StarDep(renames[self.name])
        return self


class IndexExprDep(typing.NamedTuple):
    index: sympy.Expr
    size: List[sympy.Expr]


@dataclasses.dataclass
class ReadWrites:
    reads: Set[MemoryDep]
    writes: Set[MemoryDep]
    index_exprs: Set[IndexExprDep]

    def rename(self, renames: typing.Dict[str, str]):
        return ReadWrites(
            {dep.rename(renames) for dep in self.reads},
            {dep.rename(renames) for dep in self.writes},
            self.index_exprs,
        )

    def with_read(self, name: str):
        assert isinstance(name, str)
        return ReadWrites(
            set.union(self.reads, {StarDep(name)}),
            self.writes,
            self.index_exprs,
        )


class RecordLoadStore(V.MockHandler):
    def __init__(self, var_ranges, normalize):
        super(RecordLoadStore, self).__init__()
        self._reads = set()
        self._writes = set()
        self._index_exprs = set()
        self._var_ranges = var_ranges
        self._normalize = normalize

    def canonicalize(self, index):
        sizes = list(self._var_ranges.values())
        if not self._normalize:
            return index, tuple([x for x in sizes if x != 1])

        # Try to further simplify the indexes even if simplify_loops didn't
        # convert it to the simpliest form because of the interference from
        # different indexing formulas.
        index_vars = list(self._var_ranges.keys())
        new_sizes, reindex, prune = _simplify_loops(index_vars, sizes, [index])

        # assign new variables each dimension to deal with numbering mismatches
        # d0, d1, d2 could become d0, d2 -- which won't match d0, d1
        _, add_var = var_builder(canonicalization_prefix())
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))

        index = sympy.expand(index).subs(replacement)
        return index, tuple(new_sizes)

    def load(self, name: str, index: sympy.Expr, upcast: bool = False):
        canonicalized_index, canonicalized_size = self.canonicalize(index)
        self._reads.add(MemoryDep(name, canonicalized_index, canonicalized_size))
        return f"load({name}, {index}, {upcast})"

    def store(self, name, index, value, mode=None):
        canonicalized_index, canonicalized_size = self.canonicalize(index)
        self._writes.add(MemoryDep(name, canonicalized_index, canonicalized_size))
        return f"store({name}, {index}, {value}, {mode})"

    def reduction(self, name, dtype, reduction_type, index, value):
        return self.store(name, index, f"reduce_{reduction_type})({value})")

    def index_expr(self, index, dtype):
        canonicalized_index, canonicalized_size = self.canonicalize(index)
        self._index_exprs.add(IndexExprDep(canonicalized_index, canonicalized_size))
        return f"index_expr({index}, {dtype})"


def var_builder(prefix):
    cnt = itertools.count()
    var_ranges = collections.OrderedDict()

    def add_var(length):
        v = sympy.Symbol(f"{prefix}{next(cnt)}", is_integer=True)
        var_ranges[v] = length
        return v

    return var_ranges, add_var


def index_vars_no_squeeze(*argsizes, prefix):
    var_ranges, add_var = var_builder(prefix)
    args = []
    for size in argsizes:
        args.append(list(map(add_var, size)))
    return args, var_ranges


def index_vars_squeeze(*argsizes, prefix="d"):
    from torchinductor.ir import SqueezeView

    var_ranges, add_var = var_builder(prefix)
    args = []
    new_sizes = []
    for size in argsizes:
        new_size, reindex = SqueezeView.squeezer(size)
        new_sizes.append(new_size)
        args.append(reindex(list(map(add_var, new_size))))
    return new_sizes, args, var_ranges


def extract_read_writes(fn, *argsizes, normalize=False):
    _, args, var_ranges = index_vars_squeeze(*argsizes)
    rw = RecordLoadStore(var_ranges, normalize=normalize)
    with V.set_ops_handler(rw):
        fn(*args)
    return ReadWrites(rw._reads, rw._writes, rw._index_exprs)


def canonicalization_prefix():
    return "c"
