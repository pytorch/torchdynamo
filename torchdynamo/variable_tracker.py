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
    """ Base class for tracked locals and stack values """

    @staticmethod
    def combine(vars):
        vars = list(vars)
        priority = [TensorVariable, AllowedFunctionOrModuleVariable, NNModuleVariable, ConstantVariable,
                    MethodNameVariable]
        vars.sort(key=lambda v: priority.index(type(v)))
        return type(vars[0]), {
            "state": combine_states(v.state for v in vars),
            "guards": combine_guards(v.guards for v in vars),
        }

    def __init__(self, state=TracingSupported.UNKNOWN, guards=None):
        super(VariableTracker, self).__init__()
        self.state = state
        self.guards = guards or set()


class TensorVariable(VariableTracker):
    """ Points to a tensor """

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


class AllowedFunctionOrModuleVariable(VariableTracker):
    """ Points to a module or method in torch.* """

    def __init__(self, value, **kwargs):
        super(AllowedFunctionOrModuleVariable, self).__init__(**kwargs)
        self.value = value


class MethodNameVariable(VariableTracker):
    def __init__(self, name, **kwargs):
        super(MethodNameVariable, self).__init__(**kwargs)
        self.name = name
