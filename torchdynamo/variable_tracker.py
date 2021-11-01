import enum
import functools
import types
from typing import Callable
from typing import List
from typing import Optional
from typing import Set

import torch.fx

from torchdynamo import config
from torchdynamo.mutation_guard import real_type as type


class TracingSupported(enum.Enum):
    UNKNOWN = 0
    YES = 1
    NO = 2

    @staticmethod
    def combine(a, b):
        return TracingSupported(max(a.value, b.value))


combine_states = functools.partial(functools.reduce, TracingSupported.combine)
combine_guards = functools.partial(functools.reduce, set.union)


def identity(x):
    return x


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

    @classmethod
    def copy(cls, value):
        """Deeper (but not full) copy, leaving FX and user objects alone"""
        return cls.apply(identity, value)

    @classmethod
    def apply(cls, fn: Callable[["VariableTracker"], "VariableTracker"], value):
        """
        Walk this object and call fn on all the VariableTracker
        instances to produce a new VariableTracker with the results.
        """
        if isinstance(value, VariableTracker):
            return fn(value.clone(**cls.apply(fn, value.__dict__)))
        elif isinstance(value, list):
            return [cls.apply(fn, v) for v in value]
        elif isinstance(value, dict):
            return {k: cls.apply(fn, value[k]) for k in sorted(value.keys())}
        else:
            return value

    def add_guard(self, guard):
        return self.clone(guards=set.union(self.guards, {guard}))

    def get_key(self):
        return self.__class__

    def __str__(self):
        return f"{self.__class__.__name__}()"

    __repr__ = __str__

    def with_initial_name(self, name: str):
        """Shallow copy with a different value for self.initial_name"""
        return self.clone(initial_name=name)

    def python_type(self):
        raise NotImplementedError("No known type")

    def python_value(self):
        raise NotImplementedError(
            "Use of corresponding Python value is not supported yet"
        )

    def __init__(
        self,
        state=TracingSupported.UNKNOWN,
        guards: Optional[Set] = None,
        initial_name: Optional[str] = None,
        global_name: Optional[str] = None,
    ):
        super(VariableTracker, self).__init__()
        self.state = state
        self.guards = guards or set()
        self.initial_name = initial_name
        self.global_name = global_name


class TensorVariable(VariableTracker):
    """Points to a tensor"""

    def __init__(
        self,
        proxy: torch.fx.Proxy,
        dtype=None,
        device=None,
        ndim=None,
        size=None,
        stride=None,
        **kwargs,
    ):
        super(TensorVariable, self).__init__(**kwargs)
        self.proxy = proxy
        self.dtype = dtype
        self.device = device
        self.ndim = ndim
        self.size = size
        self.stride = stride

    def as_proxy(self):
        return self.proxy

    def python_type(self):
        return torch.Tensor

    @staticmethod
    def specialize(value: torch.Tensor):
        props = {
            "dtype": value.dtype,
            "device": value.device,
            "ndim": int(value.ndim),
        }
        if not config.dynamic_shapes:
            props["size"] = tuple(value.size())
            props["stride"] = tuple(value.stride())
        return props

    def const_attr(self, name):
        result = None
        wrapped = False
        options = VariableTracker.propagate([self])
        if name in ("ndim", "ndimension", "dim") and self.ndim is not None:
            wrapped = name != "ndim"
            result = ConstantVariable(self.ndim, **options)
        elif name == "dtype" and self.dtype is not None:
            result = AllowedFunctionOrModuleVariable(self.dtype, **options)
        elif name == "device" and self.device is not None:
            result = AllowedFunctionOrModuleVariable(self.device, **options)
        elif name == "is_cuda" and self.device is not None:
            result = ConstantVariable(self.device.type == "cuda", **options)
        elif name in ("size", "shape") and self.size is not None:
            wrapped = name == "size"
            result = ConstantVariable(self.size, **options)
        elif name == "stride" and self.stride is not None:
            wrapped = True
            result = ConstantVariable(self.stride, **options)
        if wrapped:
            result = FunctionConstantWrapper(result, **options)
        return result


class BasicTypeVariable(TensorVariable):
    """
    Points to a simple type, e.g. int, float, str. So far, we treat this
    the same as TensorVariable
    """

    def python_type(self):
        return self.proxy


class NNModuleVariable(VariableTracker):
    def __init__(self, module_key: str, **kwargs):
        super(NNModuleVariable, self).__init__(**kwargs)
        self.module_key = module_key

    def get_key(self):
        return self.__class__, self.module_key

    def python_type(self):
        return torch.nn.Module


class ConstantVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super(ConstantVariable, self).__init__(**kwargs)
        self.value = value

    def as_proxy(self):
        return self.value

    def get_key(self):
        return self.__class__, self.value

    def python_type(self):
        return type(self.value)

    def python_value(self):
        return self.value

    def getitem_const(self, arg):
        return ConstantVariable(
            self.value[arg.python_value()], **VariableTracker.propagate([self, arg])
        )

    @staticmethod
    def is_literal(obj):
        if type(obj) in (int, float, bool, type(None)):
            return True
        if type(obj) in (list, tuple):
            return all(ConstantVariable.is_literal(x) for x in obj)
        return False


class FunctionConstantWrapper(VariableTracker):
    def __init__(self, value, **kwargs):
        super(FunctionConstantWrapper, self).__init__(**kwargs)
        self.value = value

    def get_key(self):
        return self.__class__, self.value

    def call_const(self, args, kwargs):
        # this is used to implement Tensor.size(1)
        assert not kwargs
        if len(args) == 1:
            return self.value.getitem_const(args[0])
        elif args:
            return tuple(self.value.getitem_const(a) for a in args)
        return self.value


class BuiltinVariable(VariableTracker):
    def __init__(self, fn, **kwargs):
        super(BuiltinVariable, self).__init__(**kwargs)
        self.fn = fn

    def get_key(self):
        return self.__class__, id(self.fn)

    def python_type(self):
        return type(self.fn)

    def python_value(self):
        return self.fn


class ListIteratorVariable(VariableTracker):
    def __init__(self, items, index: int = 0, **kwargs):
        super(ListIteratorVariable, self).__init__(**kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        self.items = items
        self.index = index

    def next_variables(self):
        if self.index >= len(self.items):
            raise StopIteration()
        # Note this is the only mutation in VariableTracker so far
        item = self.items[self.index]
        self.index += 1
        self.initial_name = None
        return item, self

    def get_key(self):
        return self.__class__, id(self.index), tuple(v.get_key() for v in self.items)


class GetAttrVariable(VariableTracker):
    def __init__(self, obj, name, **kwargs):
        super(GetAttrVariable, self).__init__(**kwargs)
        assert isinstance(obj, VariableTracker)
        assert isinstance(name, str)
        self.obj = obj
        self.name = name

    def as_proxy(self):
        return getattr(self.obj.as_proxy(), self.name)

    def get_key(self):
        return self.__class__, self.name, self.obj.get_key()


class BaseListVariable(VariableTracker):
    def __init__(self, items, **kwargs):
        super(BaseListVariable, self).__init__(**kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        self.items = items

    def _as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def get_key(self):
        return self.__class__, tuple(v.get_key() for v in self.items)


class ListVariable(BaseListVariable):
    def as_proxy(self):
        return list(self._as_proxy())

    def python_type(self):
        return list


class TupleVariable(BaseListVariable):
    def as_proxy(self):
        return tuple(self._as_proxy())

    def python_type(self):
        return tuple


class SliceVariable(BaseListVariable):
    def as_proxy(self):
        return slice(*self._as_proxy())

    def python_type(self):
        return slice

    def python_value(self):
        return slice(*[x.python_value() for x in self.items])


class ConstDictVariable(VariableTracker):
    def __init__(self, items, **kwargs):
        super(ConstDictVariable, self).__init__(**kwargs)
        assert isinstance(items, dict)
        self.items = items

    def get_key(self):
        return self.__class__, tuple(
            (k, self.items[k].get_key()) for k in sorted(self.items.keys())
        )

    def as_proxy(self):
        return {k: v.as_proxy() for k, v in self.items.items()}

    def python_type(self):
        return dict


class UserFunctionVariable(VariableTracker):
    """Some unsupported user-defined global function"""

    def __init__(self, fn, **kwargs):
        super(UserFunctionVariable, self).__init__(**kwargs)
        self.fn = fn

    def self_args(self):
        return []

    def get_key(self):
        return self.__class__, id(self.fn)

    def python_type(self):
        return types.FunctionType


class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(self, fn, obj, **kwargs):
        super(UserMethodVariable, self).__init__(fn=fn, **kwargs)
        self.obj = obj

    def self_args(self):
        return [self.obj]

    def get_key(self):
        return self.__class__, id(self.fn), self.obj.get_key()

    def python_type(self):
        return types.MethodType


class AllowedFunctionOrModuleVariable(VariableTracker):
    """Points to a module or method in torch.*"""

    def __init__(self, value, **kwargs):
        super(AllowedFunctionOrModuleVariable, self).__init__(**kwargs)
        self.value = value

    def as_proxy(self):
        return self.value

    def get_key(self):
        return self.__class__, id(self.value)

    def python_type(self):
        if isinstance(self.value, (torch.Tensor, torch.nn.Module)):
            return type(self.value)
        return super().python_type()

    def python_value(self):
        if isinstance(self.value, (torch.Tensor, torch.nn.Module)):
            return self.value
        return super().python_value()

    def is_basic_math(self):
        return getattr(self.value, "__module__", None) == "math"


class PythonModuleVariable(VariableTracker):
    def __init__(self, value: types.ModuleType, **kwargs):
        super(PythonModuleVariable, self).__init__(**kwargs)
        self.value = value

    def get_key(self):
        return self.__class__, id(self.value)

    def python_type(self):
        return types.ModuleType


class UnsupportedVariable(VariableTracker):
    """
    Mostly objects of defined type.  Catch-all for something where we only know the type.
    """

    def __init__(self, value, value_type=None, **kwargs):
        super(UnsupportedVariable, self).__init__(**kwargs)
        self.value = value
        self.value_type = value_type or type(value)

    def __str__(self):
        return f"{self.__class__.__name__}({self.value_type.__name__})"

    def get_key(self):
        return self.__class__, id(self.value_type)

    def python_type(self):
        return self.value_type


class UnknownVariable(VariableTracker):
    """
    It could be anything!
    """

    pass
