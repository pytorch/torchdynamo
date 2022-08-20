import dataclasses
from dataclasses import field
from dis import Instruction
from types import CodeType
from types import ModuleType
from typing import Any
from typing import List

import dill


@dataclasses.dataclass
class ModuleRecord:
    module: ModuleType
    accessed_attrs: dict[str, Any] = field(default_factory=dict)


@dataclasses.dataclass
class DummyModule:
    name: str


@dataclasses.dataclass
class ExecutionRecord:
    code: CodeType
    instrs: List[Instruction] = field(default_factory=list)
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)

    def dump(self, f):
        dill.dump(self, f)

    @classmethod
    def load(cls, f):
        return dill.load(f)


@dataclasses.dataclass
class ExecutionRecorder:
    code: CodeType
    instrs: List[Instruction] = field(default_factory=list)
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)
    name_to_modrec: dict[str, Any] = field(default_factory=dict)

    def add_global_var(self, name, var):
        if isinstance(var, ModuleType):
            var = ModuleRecord(var)
            self.name_to_modrec[name] = var

        self.globals[name] = var

    def add_module_access(self, mod, name, val):
        self.name_to_modrec[mod.__name__].accessed_attrs[name] = val

    def get_record(self):
        return ExecutionRecord(
            self.code,
            self.instrs.copy(),
            ExecutionRecorder._resolve_modules(self.globals),
            ExecutionRecorder._resolve_modules(self.locals),
            self.builtins.copy(),
            self.code_options.copy(),
        )

    # Convert ModuleRecords -> DummyModule tree
    @classmethod
    def _resolve_modules(cls, vars):
        def resolve_module(var):
            if not isinstance(var, ModuleRecord):
                return var

            dummy_mod = DummyModule(var.module.__name__)
            for attr_name, attr_value in var.accessed_attrs.items():
                attr_value = resolve_module(attr_value)
                dummy_mod.__setattr__(attr_name, attr_value)

            return dummy_mod

        return {k: resolve_module(v) for k, v in vars.items()}
