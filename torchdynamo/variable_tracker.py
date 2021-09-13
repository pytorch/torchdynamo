import enum
import functools


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
    """Base class for tracked locals and stack values"""

    @staticmethod
    def propagate(vars):
        if len(vars) == 0:
            return {}
        assert all(isinstance(x, VariableTracker) for x in vars)
        return {
            "state": combine_states(v.state for v in vars),
            "guards": combine_guards(v.guards for v in vars),
        }

    @staticmethod
    def combine_type(vars):
        if len(vars) == 0:
            return ConstantVariable, {}
        vars = list(vars)
        priority = [
            TensorVariable,
            AllowedFunctionOrModuleVariable,
            NNModuleVariable,
            ConstantVariable,
            MethodNameVariable,
            GetAttrVariable,
            ListVariable,
            TupleVariable,
            SliceVariable,
        ]
        vars.sort(key=lambda v: priority.index(type(v)))
        return type(vars[0])

    def with_initial_name(self, name):
        args = dict(self.__dict__)
        args["initial_name"] = name
        return self.__class__(**args)

    def __init__(self, state=TracingSupported.UNKNOWN, guards=None, initial_name=None):
        super(VariableTracker, self).__init__()
        self.state = state
        self.guards = guards or set()
        self.initial_name = initial_name


class TensorVariable(VariableTracker):
    """Points to a tensor"""

    def __init__(self, proxy, **kwargs):
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


class IterVariable(VariableTracker):
    def __init__(self, it, **kwargs):
        super(IterVariable, self).__init__(**kwargs)
        self.it = it


class GetAttrVariable(VariableTracker):
    def __init__(self, obj, name, **kwargs):
        super(GetAttrVariable, self).__init__(**kwargs)
        self.obj = obj
        self.name = name

    def as_proxy(self):
        return getattr(self.obj.as_proxy(), self.name)


class ListVariable(VariableTracker):
    def __init__(self, items, **kwargs):
        super(ListVariable, self).__init__(**kwargs)
        self.items = items

    def as_proxy(self):
        return [x.as_proxy() for x in self.items]


class TupleVariable(ListVariable):
    def as_proxy(self):
        return tuple(super().as_proxy())


class SliceVariable(ListVariable):
    def as_proxy(self):
        return slice(*super().as_proxy())


class ConstDictVariable(VariableTracker):
    def __init__(self, items, **kwargs):
        super(ConstDictVariable, self).__init__(**kwargs)
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


class MethodNameVariable(VariableTracker):
    def __init__(self, name, **kwargs):
        super(MethodNameVariable, self).__init__(**kwargs)
        self.name = name
