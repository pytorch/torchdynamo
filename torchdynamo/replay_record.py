import dataclasses
from dataclasses import field
from types import CodeType
from types import ModuleType
from typing import Any

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
    MOD_EXCLUDES = ["torch"]
    LOCAL_MOD_PREFIX = "___local_mod_"

    code: CodeType
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)
    name_to_modrec: dict[str, Any] = field(default_factory=dict)

    def add_local_var(self, name, var):
        self.locals[name] = var

    def add_global_var(self, name, var):
        if isinstance(var, ModuleType):
            if self._is_excl(var):
                return
            mod_rec = ModuleRecord(var)
            self.name_to_modrec[var.__name__] = mod_rec
            self.globals[name] = mod_rec
        else:
            self.globals[name] = var

    def add_local_mod(self, name, mod):
        assert isinstance(mod, ModuleType)
        if self._is_excl(mod):
            return

        self.add_global_var(self.LOCAL_MOD_PREFIX + name, mod)

    def record_module_access(self, mod, name, val):
        if self._is_excl(mod):
            return
        # check local mods first
        local_mod_name = self.LOCAL_MOD_PREFIX + mod.__name__
        if local_mod_name in self.name_to_modrec:
            self.name_to_modrec[local_mod_name].accessed_attrs[name] = val
        else:
            self.name_to_modrec[mod.__name__].accessed_attrs[name] = val

    def get_record(self):
        return ExecutionRecord(
            self.code,
            ExecutionRecorder._resolve_modules(self.globals),
            ExecutionRecorder._resolve_modules(self.locals),
            self.builtins.copy(),
            self.code_options.copy(),
        )

    @classmethod
    def _is_excl(cls, mod):
        return any([mod.__name__ == excl for excl in cls.MOD_EXCLUDES])

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
