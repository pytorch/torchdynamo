import enum
import functools
from typing import Callable, List, Set, Optional

import torch.fx


class TracingSupported(enum.Enum):
    UNKNOWN = 0
    YES = 1
    NO = 2

    @staticmethod
    def combine(a, b):
        return TracingSupported(max(a.value, b.value))


combine_states = functools.partial(functools.reduce, TracingSupported.combine)
combine_guards = functools.partial(functools.reduce, set.union)


class VariableTracker:
    """
    Base class for tracked locals and stack values

    VariableTracker instances are immutable and should be copied in
    order to change them.
    """

    @staticmethod
    def propagate(vars: List["VariableTracker"]):
        if len(vars) == 0:
            return {}
        assert all(isinstance(x, VariableTracker) for x in vars)
        return {
            "state": combine_states(v.state for v in vars),
            "guards": combine_guards(v.guards for v in vars),
        }

    def clone(self, **kwargs):
        """Shallow copy with some (optional) changes"""
        args = dict(self.__dict__)
        args.update(kwargs)
        return self.__class__(**args)

    def apply(self, fn: Callable[["VariableTracker"], "VariableTracker"]):
        """
        Walk this object and call fn on all the VariableTracker
        instances to produce a new VariableTracker with the results.
        """
        updates = {}
        for key, value in dict(self.__dict__).items():
            if isinstance(value, VariableTracker):
                updates[key] = value.apply(fn)
            elif isinstance(value, list):
                assert len(value) == 0 or isinstance(value[0], VariableTracker)
                updates[key] = [item.apply(fn) for item in value]
            elif isinstance(value, dict):
                assert len(value) == 0 or isinstance(
                    next(iter(value.values())), VariableTracker
                )
                updates[key] = {k: item.apply(fn) for k, item in value.items()}
        return fn(self.clone(**updates))

    def visit(self, fn: Callable[["VariableTracker"], None]):
        """Walk this object and call fn on all the VariableTracker instances"""

        def fn_(obj: VariableTracker):
            fn(obj)
            return obj

        self.apply(fn_)
        return

    def with_initial_name(self, name: str):
        """Shallow copy with a different value for self.initial_name"""
        return self.clone(initial_name=name)

    def __init__(
        self,
        state=TracingSupported.UNKNOWN,
        guards: Optional[Set] = None,
        initial_name=None,
    ):
        super(VariableTracker, self).__init__()
        self.state = state
        self.guards = guards or set()
        self.initial_name = initial_name


class TensorVariable(VariableTracker):
    """Points to a tensor"""

    def __init__(self, proxy: torch.fx.Proxy, **kwargs):
        super(TensorVariable, self).__init__(**kwargs)
        self.proxy = proxy

    def as_proxy(self):
        return self.proxy


class NNModuleVariable(VariableTracker):
    def __init__(self, key: str, **kwargs):
        super(NNModuleVariable, self).__init__(**kwargs)
        self.key = key


class ConstantVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super(ConstantVariable, self).__init__(**kwargs)
        self.value = value

    def as_proxy(self):
        return self.value


class BuiltinVariable(VariableTracker):
    def __init__(self, fn, **kwargs):
        super(BuiltinVariable, self).__init__(**kwargs)
        self.fn = fn


class ListIteratorVariable(VariableTracker):
    def __init__(self, items, index: int = 0, **kwargs):
        super(ListIteratorVariable, self).__init__(**kwargs)
        assert isinstance(items, list)
        self.items = items
        self.index = index

    def next_variables(self):
        if self.index >= len(self.items):
            raise StopIteration()
        return self.items[self.index], self.clone(index=self.index + 1)


class GetAttrVariable(VariableTracker):
    def __init__(self, obj, name, **kwargs):
        super(GetAttrVariable, self).__init__(**kwargs)
        assert isinstance(obj, VariableTracker)
        assert isinstance(name, str)
        self.obj = obj
        self.name = name

    def as_proxy(self):
        return getattr(self.obj.as_proxy(), self.name)


class BaseListVariable(VariableTracker):
    def __init__(self, items, **kwargs):
        super(BaseListVariable, self).__init__(**kwargs)
        assert isinstance(items, list)
        self.items = items

    def _as_proxy(self):
        return [x.as_proxy() for x in self.items]


class ListVariable(BaseListVariable):
    def as_proxy(self):
        return list(self._as_proxy())


class TupleVariable(BaseListVariable):
    def as_proxy(self):
        return tuple(self._as_proxy())


class SliceVariable(BaseListVariable):
    def as_proxy(self):
        return slice(*self._as_proxy())


class ConstDictVariable(VariableTracker):
    def __init__(self, items, **kwargs):
        super(ConstDictVariable, self).__init__(**kwargs)
        assert isinstance(items, dict)
        self.items = items


class UserFunctionVariable(VariableTracker):
    """Some unsupported user-defined global function"""

    def __init__(self, fn, **kwargs):
        super(UserFunctionVariable, self).__init__(**kwargs)
        self.fn = fn

    def self_args(self):
        return []


class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(self, fn, obj, **kwargs):
        super(UserMethodVariable, self).__init__(fn=fn, **kwargs)
        self.obj = obj

    def self_args(self):
        return [self.obj]


class AllowedFunctionOrModuleVariable(VariableTracker):
    """Points to a module or method in torch.*"""

    def __init__(self, value, **kwargs):
        super(AllowedFunctionOrModuleVariable, self).__init__(**kwargs)
        self.value = value

    def as_proxy(self):
        return self.value


class PythonModuleVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super(PythonModuleVariable, self).__init__(**kwargs)
        self.value = value
