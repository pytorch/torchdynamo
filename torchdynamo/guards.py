import dataclasses
import enum
import itertools
import types
import weakref
from typing import Optional, Set, Dict


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1


class GuardRequirement(enum.Enum):
    TYPE_MATCH = 0
    VALUE_MATCH = 1
    FUNCTION_MATCH = 2  # e.q. "from torch import add"
    FIXED_TENSOR_LIST = 3


@dataclasses.dataclass
class Guard:
    name: str
    source: GuardSource
    requirement: GuardRequirement

    def __hash__(self):
        return hash((self.name, self.source, self.requirement))


class GuardBuilder:
    def __init__(self, id_ref, scope):
        self.id_ref = id_ref
        self.argnames = []
        self.code = []
        self.scope = scope

    def arg_ref(self, guard: Guard):
        if guard.name not in self.argnames:
            self.argnames.append(guard.name)
        return guard.name

    def TYPE_MATCH(self, guard: Guard):
        self.code.append(f"id(type({self.arg_ref(guard)})) == {self.id_ref(type(self.scope[guard.name]))}")

    def VALUE_MATCH(self, guard: Guard):
        self.code.append(f"id({self.arg_ref(guard)}) == {self.id_ref(self.scope[guard.name])}")

    def FUNCTION_MATCH(self, guard: Guard):
        pass  # should we add more checks here?

    def FIXED_TENSOR_LIST(self, guard: Guard):
        ref = self.arg_ref(guard)
        value = self.scope[guard.name]
        assert len(value) > 0
        tensor_type = self.id_ref(type(value[0]))
        assert all([id(type(v)) == tensor_type for v in value])
        self.code.append(f"id(type({ref})) == {self.id_ref(type(value))}")
        self.code.append(f"len({ref}) == {len(value)}")
        for i in range(len(value)):
            self.code.append(f"id(type({ref}[{i}])) == {tensor_type}")


class GuardedCode:
    identifier = itertools.count()

    def __init__(self,
                 code: types.CodeType,
                 guards: Optional[Set[Guard]] = None,
                 f_locals: Optional[Dict] = None,
                 f_globals: Optional[Dict] = None):
        self.code = code
        self.valid = True
        self._weakrefs = []
        self._seen_ids = set()

        local_builder = GuardBuilder(self.id_ref, f_locals)
        global_builder = GuardBuilder(self.id_ref, f_globals)
        for guard in (guards or []):
            if guard.source == GuardSource.LOCAL:
                getattr(local_builder, guard.requirement.name)(guard)
            else:
                getattr(global_builder, guard.requirement.name)(guard)
        self.check_fn = self.compile_check_fn(local_builder, global_builder)

    def compile_check_fn(self, local_builder, global_builder):
        local_builder.argnames.append("**___kwargs_ignored")
        assert not (set(local_builder.argnames) & set(global_builder.argnames))
        code = [f"__guarded_code.valid"] + local_builder.code + global_builder.code
        py_code = f"lambda __guarded_code: lambda {','.join(local_builder.argnames)}: ({' and '.join(code)})"
        return eval(py_code, global_builder.scope)(self)

    def invalidate(self, ref):
        # A weakref is no longer valid
        self.valid = False

    def id_ref(self, obj):
        """ add a weakref, return the id """
        try:
            if id(obj) not in self._seen_ids:
                self._weakrefs.append(weakref.ref(obj, self.invalidate))
                self._seen_ids.add(id(obj))
        except TypeError:
            pass  # cannot weakref bool object
        return id(obj)
