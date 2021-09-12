import dataclasses
import enum
import types
import weakref
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1

    def select(self, locals_, globals_):
        if self == GuardSource.LOCAL:
            return locals_
        if self == GuardSource.GLOBAL:
            return globals_


@dataclasses.dataclass
class Guard:
    name: str
    source: GuardSource
    create_fn: Callable

    def __hash__(self):
        return hash((self.name, self.source, id(self.create_fn)))

    def create(self, local_builder: "GuardBuilder", global_builder: "GuardBuilder"):
        return self.create_fn(self.source.select(local_builder, global_builder), self)


class GuardBuilder:
    def __init__(self, id_ref: Callable, scope: Dict[str, Any]):
        self.id_ref = id_ref
        self.scope = scope
        self.argnames: List[str] = []
        # Code is python expression strings generated for each guard
        self.code: List[str] = []

    def arg_ref(self, guard: Guard):
        if guard.name not in self.argnames:
            self.argnames.append(guard.name)
        return guard.name

    def TYPE_MATCH(self, guard: Guard):
        self.code.append(
            f"id(type({self.arg_ref(guard)})) == {self.id_ref(type(self.scope[guard.name]))}"
        )

    def VALUE_MATCH(self, guard: Guard):
        self.code.append(
            f"id({self.arg_ref(guard)}) == {self.id_ref(self.scope[guard.name])}"
        )

    def FUNCTION_MATCH(self, guard: Guard):
        """things like torch.add and user defined functions"""
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
    def __init__(
        self,
        code: types.CodeType,
        guards: Optional[Set[Guard]] = None,
        f_locals: Optional[Dict] = None,
        f_globals: Optional[Dict] = None,
    ):
        self.code = code
        self.valid = True
        self._weakrefs = []
        self._seen_ids = set()

        local_builder = GuardBuilder(self.id_ref, f_locals)
        global_builder = GuardBuilder(self.id_ref, f_globals)
        for guard in guards or []:
            guard.create(local_builder, global_builder)
        self.check_fn = self.compile_check_fn(local_builder, global_builder)
        self._seen_ids.clear()

    def compile_check_fn(self, local_builder, global_builder):
        assert not (set(local_builder.argnames) & set(global_builder.argnames))
        args = local_builder.argnames + ["**___kwargs_ignored"]
        code = ["___guarded_code.valid"] + local_builder.code + global_builder.code
        py_code = (
            f"lambda ___guarded_code: lambda {','.join(args)}: ({' and '.join(code)})"
        )
        return eval(py_code, global_builder.scope)(self)

    def invalidate(self, ref):
        # A weakref is no longer valid, self.check_fn should return false
        self.valid = False

    def id_ref(self, obj):
        """add a weakref, return the id"""
        try:
            if id(obj) not in self._seen_ids:
                self._weakrefs.append(weakref.ref(obj, self.invalidate))
                self._seen_ids.add(id(obj))
        except TypeError:
            pass  # cannot weakref bool object
        return id(obj)
