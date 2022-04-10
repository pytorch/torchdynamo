import dataclasses
import itertools
import typing
from typing import Set

import sympy

from .virtualized import MockHandler
from .virtualized import ops


class MemoryDep(typing.NamedTuple):
    name: str
    index: sympy.Expr


@dataclasses.dataclass
class ReadWrites:
    reads: Set[MemoryDep]
    writes: Set[MemoryDep]


class RecordLoadStore(MockHandler):
    def __init__(self):
        super(RecordLoadStore, self).__init__()
        self._reads = set()
        self._writes = set()

    def load(self, name: str, index: sympy.Expr):
        self._reads.add(MemoryDep(name, index))
        return f"load({name}, {index})"

    def store(self, name, index, value):
        self._writes.add(MemoryDep(name, index))
        return f"store({name}, {index}, {value})"

    def reduction(self, name, dtype, reduction_type, index, value):
        return self.store(name, index, f"reduce_{reduction_type})({value})")


def extract_read_writes(fn, *argcounts):
    cnt = itertools.count()
    args = [[sympy.Symbol(f"d{next(cnt)}") for _ in range(n)] for n in argcounts]
    rw = RecordLoadStore()
    with ops.set_handler(rw):
        fn(*args)
    return ReadWrites(rw._reads, rw._writes)
