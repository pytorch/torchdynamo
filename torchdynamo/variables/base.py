import collections
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from .. import variables
from ..source import AttrSource
from ..source import Source
from ..utils import dict_values
from ..utils import identity
from ..utils import odict_values
from ..utils import unimplemented


class MutableLocal:
    """
    Marker used to indicate this (list, iter, etc) was constructed in
    local scope and can be mutated safely in analysis without leaking
    state.
    """

    pass


class VariableTracker:
    """
    Base class for tracked locals and stack values

    VariableTracker instances are immutable and should be copied in
    order to change them.
    """

    @staticmethod
    def propagate(*vars: List[List["VariableTracker"]]):
        """Combine the guards from many VariableTracker into **kwargs for a new instance"""
        guards = set()

        def visit(var):
            if type(var) in (list, tuple, dict_values, odict_values):
                for i in var:
                    visit(i)
            else:
                assert isinstance(var, VariableTracker), typestr(var)
                guards.update(var.guards)

        visit(vars)
        return {
            "guards": guards,
        }

    def clone(self, **kwargs):
        """Shallow copy with some (optional) changes"""
        args = dict(self.__dict__)
        args.update(kwargs)
        return self.__class__(**args)

    @classmethod
    def copy(cls, value):
        """Deeper (but not full) copy, leaving FX and user objects alone"""
        return cls.apply(identity, value)

    @classmethod
    def apply(
        cls, fn: Callable[["VariableTracker"], "VariableTracker"], value, cache=None
    ):
        """
        Walk this object and call fn on all the VariableTracker
        instances to produce a new VariableTracker with the results.
        """
        if cache is None:
            cache = dict()

        idx = id(value)
        if idx in cache:
            return cache[idx][0]

        if isinstance(value, VariableTracker):
            result = fn(value.clone(**cls.apply(fn, value.__dict__, cache)))
        elif isinstance(value, list):
            result = [cls.apply(fn, v, cache) for v in value]
        elif isinstance(value, tuple):
            result = tuple(cls.apply(fn, v, cache) for v in value)
        elif isinstance(value, collections.OrderedDict):
            result = collections.OrderedDict(
                cls.apply(fn, v, cache) for v in value.items()
            )
        elif isinstance(value, dict):
            result = {k: cls.apply(fn, v, cache) for k, v in value.items()}
        else:
            result = value

        # save `value` to keep it alive and ensure id() isn't reused
        cache[idx] = (result, value)
        return result

    def add_guard(self, guard):
        return self.clone(guards=set.union(self.guards, {guard}))

    def add_guards(self, guards):
        assert isinstance(guards, set)
        return self.clone(guards=set.union(self.guards, guards))

    def add_options(self, options, *more):
        if more:
            return self.add_options(options).add_options(*more)
        if isinstance(options, VariableTracker):
            return self.add_guards(options.guards)
        assert isinstance(options, dict)
        return self.add_guards(options.get("guards", set()))

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def __repr__(self):
        return str(self)

    def python_type(self):
        raise NotImplementedError(f"{self} has no type")

    def as_python_constant(self):
        """For constants"""
        raise NotImplementedError(f"{self} is not a constant")

    def is_python_constant(self):
        try:
            self.as_python_constant()
            return True
        except NotImplementedError:
            return False

    def can_create_guard(self):
        try:
            self.create_guard(None)
            return True
        except NotImplementedError:
            return False

    def create_guard(self, fn):
        if self.source:
            return self.source.create_guard(fn)
        raise NotImplementedError()

    def replace_guards(self, guards, *fns):
        name = self.source.name()
        new_guards = {g for g in (guards or []) if g.name != name}
        new_guards.update(self.source.create_guard(fn) for fn in fns)
        return new_guards

    def has_const_attr(self, tx, name):
        try:
            return variables.ConstantVariable.is_literal(self.get_const_attr(tx, name))
        except NotImplementedError:
            return False

    def get_const_attr(self, tx, name):
        raise NotImplementedError()

    def get_var_attr(self, tx, name):
        options = VariableTracker.propagate(self)
        if self.source:
            options["source"] = AttrSource(self.source, name)
        return variables.ConstantVariable(self.get_const_attr(tx, name), **options)

    def is_proxy(self):
        try:
            self.as_proxy()
            return True
        except NotImplementedError:
            return False

    def as_proxy(self):
        raise NotImplementedError(str(self))

    def reconstruct(self, codegen):
        raise NotImplementedError()

    def unpack_var_sequence(self, tx):
        raise NotImplementedError()

    def has_unpack_var_sequence(self, tx):
        try:
            self.unpack_var_sequence(tx)
            return True
        except NotImplementedError:
            return False

    def num_parameters(self):
        unimplemented(f"num_parameters: {self}")

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        unimplemented(f"hasattr: {self}")

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        raise unimplemented(f"call_function {self} {args} {kwargs}")

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__len__" and self.has_unpack_var_sequence(tx):
            assert not (args or kwargs)
            return variables.ConstantVariable(
                len(self.unpack_var_sequence(tx)), **VariableTracker.propagate(self)
            )
        raise unimplemented(f"call_method {self} {name} {args} {kwargs}")

    def __init__(
        self,
        guards: Optional[Set] = None,
        source: Source = None,
        mutable_local: MutableLocal = None,
    ):
        super(VariableTracker, self).__init__()
        self.guards = guards or set()
        self.source = source
        self.mutable_local = mutable_local


def typestr(*objs):
    if len(objs) == 1:
        (obj,) = objs
        if isinstance(obj, VariableTracker):
            return str(obj)
        else:
            return type(obj).__name__
    else:
        return " ".join(map(typestr, objs))
