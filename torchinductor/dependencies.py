import dataclasses
import itertools
import typing
from typing import List
from typing import Set

import sympy

from .virtualized import MockHandler
from .virtualized import ops


class MemoryDep(typing.NamedTuple):
    name: str
    index: sympy.Expr
    size: List[sympy.Expr]


class StarDep(typing.NamedTuple):
    # depends on the entire buffer
    name: str


@dataclasses.dataclass
class ReadWrites:
    reads: Set[MemoryDep]
    writes: Set[MemoryDep]


class RecordLoadStore(MockHandler):
    def __init__(self, size):
        super(RecordLoadStore, self).__init__()
        self._reads = set()
        self._writes = set()
        self._size = tuple(size)

    def load(self, name: str, index: sympy.Expr):
        self._reads.add(MemoryDep(name, index, self._size))
        return f"load({name}, {index})"

    def store(self, name, index, value):
        self._writes.add(MemoryDep(name, index, self._size))
        return f"store({name}, {index}, {value})"

    def reduction(self, name, dtype, reduction_type, index, value):
        return self.store(name, index, f"reduce_{reduction_type})({value})")


def extract_read_writes(fn, *argsizes):
    from .ir import SqueezeView

    args = []
    new_sizes = []
    cnt = itertools.count()
    for size in argsizes:
        new_size, reindex = SqueezeView.squeezer(size)
        new_sizes.append(new_size)
        args.append(reindex([sympy.Symbol(f"d{next(cnt)}") for _ in new_size]))
    rw = RecordLoadStore(new_sizes[0])
    with ops.set_handler(rw):
        fn(*args)
    return ReadWrites(rw._reads, rw._writes)
