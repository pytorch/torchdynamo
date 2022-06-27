import collections
import dataclasses
from typing import Any

from . import utils
from .bytecode_transformation import create_instruction
from .guards import Guard
from .guards import GuardSource
from .utils import rename_implicit

_GUARD_SOURCE_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_NN_MODULE,
    GuardSource.LOCAL_NN_MODULE: GuardSource.LOCAL_NN_MODULE,
    GuardSource.GLOBAL_NN_MODULE: GuardSource.GLOBAL_NN_MODULE,
}

_GUARD_SOURCE_NOT_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL,
    GuardSource.GLOBAL: GuardSource.GLOBAL,
    GuardSource.LOCAL_NN_MODULE: GuardSource.LOCAL,
    GuardSource.GLOBAL_NN_MODULE: GuardSource.GLOBAL,
}


@dataclasses.dataclass
class Source:
    def create_guard(self, fn):
        return Guard(self.name(), self.guard_source(), fn)

    def reconstruct(self, codegen):
        raise NotImplementedError()

    def guard_source(self):
        raise NotImplementedError()

    def name(self):
        raise NotImplementedError()

    def make_guard(self, fn):
        return Guard(self.name(), self.guard_source(), fn)

    def is_nn_module(self):
        return self.guard_source() in (
            GuardSource.LOCAL_NN_MODULE,
            GuardSource.GLOBAL_NN_MODULE,
        )


@dataclasses.dataclass
class LocalSource(Source):
    local_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load(self.local_name)]

    def guard_source(self):
        return GuardSource.LOCAL

    def name(self):
        return rename_implicit(self.local_name)


@dataclasses.dataclass
class RandomValueSource(Source):
    random_call_index: int

    def reconstruct(self, codegen):
        return [
            codegen.create_load(codegen.tx.output.random_values_var),
            codegen.create_load_const(self.random_call_index),
            create_instruction("BINARY_SUBSCR"),
        ]

    def name(self):
        return rename_implicit(f"random_value_{self.random_call_index}")


@dataclasses.dataclass
class GlobalSource(Source):
    global_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load_global(self.global_name, add=True)]

    def guard_source(self):
        return GuardSource.GLOBAL

    def name(self):
        return self.global_name


@dataclasses.dataclass
class AttrSource(Source):
    base: Source
    member: str

    def __init__(self, base, member):
        super().__init__()
        if "." in member:
            member_parts = member.split(".")
            self.base = AttrSource(base, ".".join(member_parts[:-1]))
            self.member = member_parts[-1]
        else:
            self.base = base
            self.member = member

    def reconstruct(self, codegen):
        return self.base.reconstruct(codegen) + codegen.create_load_attrs(self.member)

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        if self.member.isnumeric():
            return f"getattr({self.base.name()}, {self.member!r})"
        return f"{self.base.name()}.{self.member}"


@dataclasses.dataclass
class GetItemSource(Source):
    base: Source
    index: Any

    def reconstruct(self, codegen):
        return self.base.reconstruct(codegen) + [
            codegen.create_load_const(self.index),
            create_instruction("BINARY_SUBSCR"),
        ]

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        return f"{self.base.name()}[{self.index!r}]"


@dataclasses.dataclass
class TupleIteratorGetItemSource(GetItemSource):
    def reconstruct(self, codegen):
        codegen.load_import_from(utils.__name__, "tuple_iterator_getitem")
        return self.base.reconstruct(codegen) + [
            codegen.create_load_const(self.index),
            create_instruction("CALL_FUNCTION", 2),
        ]

    def name(self):
        return f"___tuple_iterator_getitem({self.base.name()}, {self.index!r})"


@dataclasses.dataclass
class TypeSource(Source):
    base: Source

    def reconstruct(self, codegen):
        codegen.load_import_from("builtins", "type")
        return self.base.reconstruct(codegen) + [create_instruction("CALL_FUNCTION", 1)]

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        return f"type({self.base.name()})"


@dataclasses.dataclass
class ODictGetItemSource(Source):
    base: Source
    index: Any

    def reconstruct(self, codegen):
        return (
            [codegen._create_load_const(collections.OrderedDict.__getitem__)]
            + self.base.reconstruct(codegen)
            + [
                codegen.create_load_const(self.index),
                create_instruction("CALL_FUNCTION", 2),
            ]
        )

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        return f"___odict_getitem({self.base.name()}, {self.index!r})"


@dataclasses.dataclass
class NNModuleSource(Source):
    inner: Source

    def reconstruct(self, codegen):
        return self.inner.reconstruct(codegen)

    def guard_source(self):
        return _GUARD_SOURCE_NN_MODULE[self.inner.guard_source()]

    def name(self):
        return self.inner.name()


class NotNNModuleSource(NNModuleSource):
    def guard_source(self):
        return _GUARD_SOURCE_NOT_NN_MODULE[self.inner.guard_source()]
