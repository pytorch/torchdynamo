import dataclasses
import itertools
import typing
from typing import List
from typing import Set

import sympy

from .virtualized import V


class MemoryDep(typing.NamedTuple):
    name: str
    index: sympy.Expr
    size: List[sympy.Expr]

    def broadcast_extend_sizes(self, extra_sizes):
        size = (*self.size, *[x for x in extra_sizes if x != 1])
        return MemoryDep(self.name, self.index, size)

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


@dataclasses.dataclass
class ReadWrites:
    reads: Set[MemoryDep]
    writes: Set[MemoryDep]

    def rename(self, renames: typing.Dict[str, str]):
        return ReadWrites(
            {dep.rename(renames) for dep in self.reads},
            {dep.rename(renames) for dep in self.writes},
        )

    def with_read(self, name: str):
        assert isinstance(name, str)
        return ReadWrites(
            set.union(self.reads, {StarDep(name)}),
            self.writes,
        )


class RecordLoadStore(V.MockHandler):
    def __init__(self, size):
        super(RecordLoadStore, self).__init__()
        self._reads = set()
        self._writes = set()
        self._size = tuple([x for x in size if x != 1])

    def load(self, name: str, index: sympy.Expr):
        self._reads.add(MemoryDep(name, index, self._size))
        return f"load({name}, {index})"

    def store(self, name, index, value):
        self._writes.add(MemoryDep(name, index, self._size))
        return f"store({name}, {index}, {value})"

    def reduction(self, name, dtype, reduction_type, index, value):
        return self.store(name, index, f"reduce_{reduction_type})({value})")


def index_vars(*argsizes):
    from .ir import SqueezeView

    args = []
    new_sizes = []
    cnt = itertools.count()
    for size in argsizes:
        new_size, reindex = SqueezeView.squeezer(size)
        new_sizes.append(new_size)
        args.append(reindex([sympy.Symbol(f"d{next(cnt)}") for _ in new_size]))
    return new_sizes, args


def extract_read_writes(fn, *argsizes):
    new_sizes, args = index_vars(*argsizes)
    rw = RecordLoadStore(new_sizes[0])
    with V.set_ops_handler(rw):
        fn(*args)
    return ReadWrites(rw._reads, rw._writes)
