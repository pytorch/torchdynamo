import dataclasses
from typing import Any

from torchdynamo.bytecode_transformation import create_instruction
from torchdynamo.guards import Guard
from torchdynamo.guards import GuardSource


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


@dataclasses.dataclass
class LocalSource(Source):
    local_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load(self.local_name)]

    def guard_source(self):
        return GuardSource.LOCAL

    def name(self):
        return self.local_name


@dataclasses.dataclass
class GlobalSource(Source):
    global_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load_global(self.global_name)]

    def guard_source(self):
        return GuardSource.GLOBAL

    def name(self):
        return self.global_name


@dataclasses.dataclass
class AttrSource(Source):
    base: Source
    member: str

    def reconstruct(self, codegen):
        return self.base.reconstruct(codegen) + [codegen.create_load_attr(self.member)]

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
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
        return f"{self.base.name()}[{self.index}]"
