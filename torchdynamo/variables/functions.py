import inspect
import types
from typing import Dict
from typing import List

from .. import variables
from ..bytecode_transformation import create_instruction
from ..utils import make_cell
from .base import VariableTracker
from .base import typestr


class BaseUserFunctionVariable(VariableTracker):
    def get_filename(self):
        return self.get_code().co_filename

    def get_name(self):
        return self.get_code().co_name

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        return tx.inline_user_function_return(self, self.self_args() + args, kwargs)

    def num_parameters(self):
        return len(inspect.signature(self.get_function()).parameters)

    def closure_vars(self, tx):
        return {}


class UserFunctionVariable(BaseUserFunctionVariable):
    """Some unsupported user-defined global function"""

    def __init__(self, fn, **kwargs):
        super(UserFunctionVariable, self).__init__(**kwargs)
        assert isinstance(
            fn, types.FunctionType
        ), f"expected FunctionType {typestr(fn)} {fn}"
        self.fn: types.FunctionType = fn

    def self_args(self):
        return []

    def get_function(self):
        return self.fn

    def get_code(self):
        return self.fn.__code__

    def python_type(self):
        return types.FunctionType

    def has_closure(self):
        if getattr(self.fn, "__closure__", None) is not None:
            if len(self.fn.__closure__) == 1 and self.fn.__code__.co_freevars == (
                "__class__",
            ):
                # not a real closure, just a usage of `super()`
                return False
            return True
        return False

    def closure_vars(self, tx):
        if self.fn.__code__.co_freevars and "__class__" in self.fn.__code__.co_freevars:
            assert len(self.fn.__closure__) == len(self.fn.__code__.co_freevars)
            cls = self.fn.__closure__[
                self.fn.__code__.co_freevars.index("__class__")
            ].cell_contents
            return {
                "__class__": variables.UserDefinedClassVariable(cls).add_options(self)
            }
        return super().closure_vars(tx)

    def has_self(self):
        return getattr(self.fn, "__self__", None) is not None

    def get_globals(self):
        return self.fn.__globals__

    def bind_args(self, parent, args, kwargs):
        from . import BaseListVariable
        from . import ConstantVariable
        from . import ConstDictVariable

        options = VariableTracker.propagate([self])

        def wrap(val):
            if isinstance(val, dict):
                return ConstDictVariable(
                    {k: wrap(v) for k, v in val.items()}, **options
                )
            elif isinstance(val, (tuple, list)):
                cls = BaseListVariable.cls_for(type(val))
                return cls(list(map(wrap, val)), **options)
            elif ConstantVariable.is_literal(val):
                return ConstantVariable(val, **options)
            else:
                assert isinstance(val, VariableTracker), typestr(val)
                return val

        fn: types.FunctionType = self.fn
        fake_func = types.FunctionType(
            fn.__code__,
            fn.__globals__,
            fn.__name__,
            tuple(map(wrap, fn.__defaults__ or [])),
            fn.__closure__,
        )
        if fn.__kwdefaults__:
            fake_func.__kwdefaults__ = {
                k: wrap(v) for k, v in fn.__kwdefaults__.items()
            }

        bound = inspect.signature(fake_func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())

        for k, v in list(result.items()):
            if isinstance(v, (tuple, dict)):
                # args/kwargs
                result[k] = wrap(v)

        return result

    def export_freevars(self, parent, child):
        pass


class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(self, fn, obj, **kwargs):
        super(UserMethodVariable, self).__init__(fn=fn, **kwargs)
        self.obj = obj

    def __str__(self):
        return f"{self.__class__.__name__}({self.fn}, {self.obj})"

    def self_args(self):
        return [self.obj]

    def python_type(self):
        return types.MethodType

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        if isinstance(self.obj, variables.NNModuleVariable) and getattr(
            self.fn, "__module__", ""
        ).startswith("torch.nn."):
            return self.obj.call_method(tx, self.fn.__name__, args, kwargs).add_options(
                self
            )
        return super().call_function(tx, args, kwargs)

    def num_parameters(self):
        return super(UserMethodVariable, self).num_parameters() - 1


class NestedUserFunctionVariable(BaseUserFunctionVariable):
    def __init__(
        self,
        fn_name,
        code,
        f_globals,
        defaults,
        kwdefaults,
        annotations,
        closure,
        closure_scope,
        **kwargs,
    ):
        super(NestedUserFunctionVariable, self).__init__(**kwargs)
        assert isinstance(fn_name.as_python_constant(), str)
        assert isinstance(code.as_python_constant(), types.CodeType)
        assert isinstance(f_globals, dict)
        self.fn_name = fn_name
        self.code = code
        self.f_globals = f_globals
        self.defaults = defaults
        self.kwdefaults = kwdefaults
        self.annotations = annotations
        self.closure = closure
        if closure is None:
            closure_scope = None
        self.closure_scope = closure_scope

    def self_args(self):
        return []

    def get_code(self):
        return self.code.as_python_constant()

    def get_function(self):
        if self.closure:
            raise NotImplementedError()
        func = types.FunctionType(
            self.code.as_python_constant(),
            self.f_globals,
            self.fn_name.as_python_constant(),
        )
        if self.defaults:
            func.__defaults__ = self.defaults.as_python_constant()
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.as_python_constant()
        if self.annotations:
            func.__annotations__ = self.annotations.as_python_constant()
        return func

    def has_closure(self):
        return self.closure is not None

    def has_self(self):
        return False

    def get_globals(self):
        return self.f_globals

    def bind_args(self, parent, args, kwargs):
        closure_items = []
        if self.closure:
            closure_items = [
                self.closure_scope.symbolic_locals[c.name] for c in self.closure.items
            ]

        code = self.get_code()
        func = types.FunctionType(
            code,
            self.f_globals,
            self.fn_name.as_python_constant(),
            self.defaults.items if self.defaults else None,
            tuple(map(make_cell, closure_items)),
        )
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.items

        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())

        for idx, var in enumerate(code.co_freevars):
            assert self.closure.items[idx].name == var
            assert var not in result
            result[var] = closure_items[idx]

        return result

    def export_freevars(self, parent, child):
        code = self.get_code()
        for var in code.co_freevars:
            if var in child.symbolic_locals:
                parent.symbolic_locals[var] = child.symbolic_locals[var]

    def reconstruct(self, codegen):
        flags = 0x00
        if self.defaults:
            flags |= 0x01
            codegen(self.defaults)
        if self.kwdefaults:
            flags |= 0x02
            codegen(self.kwdefaults)
        if self.annotations:
            flags |= 0x04
            codegen(self.annotations)
        if self.closure:
            flags |= 0x08
            codegen(self.closure)
        codegen(self.code)
        codegen(self.fn_name)
        return [create_instruction("MAKE_FUNCTION", flags)]
