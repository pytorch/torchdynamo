import dataclasses
import enum
import re
import textwrap
import types
import weakref
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import torch

from . import mutation_guard
from ._guards import TensorGuards
from ._guards import check_obj_id
from ._guards import check_type_id
from .utils import istype


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1
    LOCAL_NN_MODULE = 2
    GLOBAL_NN_MODULE = 3

    def select(self, locals_, globals_):
        if self in (GuardSource.LOCAL, GuardSource.LOCAL_NN_MODULE):
            return locals_
        if self in (GuardSource.GLOBAL, GuardSource.GLOBAL_NN_MODULE):
            return globals_
        raise NotImplementedError()

    def is_nn_module(self):
        return self in (GuardSource.GLOBAL_NN_MODULE, GuardSource.LOCAL_NN_MODULE)

    def is_local(self):
        return self in (GuardSource.LOCAL, GuardSource.LOCAL_NN_MODULE)


@dataclasses.dataclass
class Guard:
    name: str
    source: GuardSource
    create_fn: Callable

    def __hash__(self):
        return hash((self.name, self.source, id(self.create_fn)))

    def sort_key(self):
        return self.source.value, self.name, self.create_fn.__code__.co_firstlineno

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()

    def __str__(self):
        return f"{self.source.name.lower()} {repr(self.name)} {self.create_fn.__name__}"

    def create(self, local_builder: "GuardBuilder", global_builder: "GuardBuilder"):
        return self.create_fn(self.source.select(local_builder, global_builder), self)

    def is_nn_module(self):
        return self.source.is_nn_module()

    def is_local(self):
        return self.source.is_local()


class GuardBuilder:
    def __init__(self, id_ref: Callable, scope: Dict[str, Any], guarded_code):
        self.id_ref = id_ref
        self.scope = scope
        self.argnames: List[str] = []
        # Code is python expression strings generated for each guard
        self.code: List[str] = []
        self.tensor_check_names = []
        self.tensor_check_examples = []
        self.guarded_code = guarded_code

    def get(self, name: str):
        return eval(name, self.scope)

    def arg_ref(self, guard: Guard):
        name = guard.name
        base = re.split(r"[.\[]", name)[0]
        if base not in self.argnames:
            self.argnames.append(base)
        return name

    def TYPE_MATCH(self, guard: Guard):
        # ___check_type_id is same as `id(type(x)) == y`
        self.code.append(
            f"___check_type_id({self.arg_ref(guard)}, {self.id_ref(type(self.get(guard.name)))})"
        )

    def ID_MATCH(self, guard: Guard):
        # ___check_obj_id is same as `id(x) == y`
        self.code.append(
            f"___check_obj_id({self.arg_ref(guard)}, {self.id_ref(self.get(guard.name))})"
        )

    def EQUALS_MATCH(self, guard: Guard):
        val = self.get(guard.name)
        assert istype(
            val,
            (
                int,
                float,
                bool,
                type(None),
                str,
                type,
                list,
                tuple,
                set,
                frozenset,
                torch.Size,
            ),
        ), type(val).__name__
        if istype(val, torch.Size):
            val = tuple(val)
        self.code.append(f"{self.arg_ref(guard)} == {val!r}")

    def CONSTANT_MATCH(self, guard: Guard):
        val = self.get(guard.name)
        if istype(val, (bool, type(None))):
            self.ID_MATCH(guard)
        else:
            self.EQUALS_MATCH(guard)

    def NN_MODULE(self, guard: Guard):
        self.ID_MATCH(guard)

    def FUNCTION_MATCH(self, guard: Guard):
        """things like torch.add and user defined functions"""
        if guard.is_local():
            return self.ID_MATCH(guard)

    def TENSOR_MATCH(self, guard: Guard):
        if guard.is_nn_module():
            self.ID_MATCH(guard)
        else:
            self.tensor_check_names.append(self.arg_ref(guard))
            self.tensor_check_examples.append(self.get(guard.name))

    def LIST_LENGTH(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        self.code.append(f"___check_type_id({ref}, {self.id_ref(type(value))})")
        self.code.append(f"len({ref}) == {len(value)}")

    def DICT_KEYS(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        self.code.append(f"___check_type_id({ref}, {self.id_ref(type(value))})")
        self.code.append(f"{ref}.keys() == {set(value.keys())!r}")

    def OBJECT_MUTATION(self, guard: Guard):
        mutation_guard.watch(self.get(guard.name), self.guarded_code)


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

        local_builder = GuardBuilder(self.id_ref, f_locals, self)
        global_builder = GuardBuilder(self.id_ref, f_globals, self)
        for guard in sorted(guards or [], key=Guard.sort_key):
            guard.create(local_builder, global_builder)
        self.check_fn = self.compile_check_fn(local_builder, global_builder)
        self._seen_ids.clear()

    def compile_check_fn(self, local_builder, global_builder):
        assert not (set(local_builder.argnames) & set(global_builder.argnames))
        args = local_builder.argnames + ["**___kwargs_ignored"]
        args = ",".join(args)
        code = ["___guarded_code.valid"] + local_builder.code + global_builder.code

        tensor_check_names = (
            local_builder.tensor_check_names + global_builder.tensor_check_names
        )
        check_tensors_fn = None
        if tensor_check_names:
            tensor_check_examples = (
                local_builder.tensor_check_examples
                + global_builder.tensor_check_examples
            )
            check_tensors_fn = TensorGuards(*tensor_check_examples).check
            code.append(f"___check_tensors({', '.join(tensor_check_names)})")

        code = " and ".join(code)

        py_code = textwrap.dedent(
            f"""
            def ___make_guard_fn(___guarded_code, ___check_type_id, ___check_obj_id, ___check_tensors):
                return lambda {args}: {code}
            """
        )
        # print("GUARDS", code)
        out = dict()
        exec(py_code, global_builder.scope, out)
        return out["___make_guard_fn"](
            self, check_type_id, check_obj_id, check_tensors_fn
        )

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
