import collections
import functools
import inspect
import math
import types
from typing import Callable
from typing import List
from typing import Optional
from typing import Set

import torch.fx

from torchdynamo import config
from torchdynamo.bytecode_transformation import create_instruction
from torchdynamo.utils import make_cell
from torchdynamo.utils import identity
from torchdynamo.variable_source import Source
from torchdynamo.variable_source import GetItemSource

combine_guards = functools.partial(functools.reduce, set.union)


class MutableLocal:
    """
    Marker used to indicate this (list, iter, etc) was constructed
    in local scope and can be mutated safely in analysis without leaking
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
    def propagate(vars: List["VariableTracker"]):
        if len(vars) == 0:
            return {}
        assert all(isinstance(x, VariableTracker) for x in vars)
        return {
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
            return {k: cls.apply(fn, v) for k, v in value.items()}
        else:
            return value

    def add_guard(self, guard):
        return self.clone(guards=set.union(self.guards, {guard}))

    def add_guards(self, guards):
        assert isinstance(guards, set)
        return self.clone(guards=set.union(self.guards, guards))

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
            return ConstantVariable.is_literal(self.get_const_attr(tx, name))
        except NotImplementedError:
            return False

    def get_const_attr(self, tx, name):
        raise NotImplementedError()

    def is_proxy(self):
        try:
            self.as_proxy()
            return True
        except NotImplementedError:
            return False

    def as_proxy(self):
        raise NotImplementedError()

    def reconstruct(self, codegen):
        raise NotImplementedError()

    def unpack_var_sequence(self, tx):
        raise NotImplementedError()

    def has_unpack_var_sequence(self, tx):
        try:
            self.unpack_var_sequence(tx)
            return True
        except Exception:
            return False

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

            def wrapper(*args):
                if len(args) == 1:
                    return result.getitem_const(args[0])
                elif args:
                    return TupleVariable(
                        [result.getitem_const(a) for a in args], **options
                    )
                return result

            return LambdaVariable(wrapper, **options)
        else:
            return result


class NNModuleVariable(VariableTracker):
    def __init__(self, module_type: type, module_key: str, **kwargs):
        super(NNModuleVariable, self).__init__(**kwargs)
        self.module_type = module_type
        self.module_key = module_key
        assert self.source

    def python_type(self):
        return self.module_type

    def unpack_var_sequence(self, tx):
        # implement list/iter/tuple/etc calls
        key = self.module_key
        base = tx.get_submodule(self.module_key)
        options = VariableTracker.propagate([self])
        assert isinstance(
            base, (torch.nn.ModuleList, torch.nn.ParameterList, torch.nn.Sequential)
        ), typestr(base)
        assert self.source
        return [
            tx.add_submodule(
                submod, key, idx, source=GetItemSource(self.source, idx), **options
            )
            for idx, submod in enumerate(base)
        ]


class ConstantVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super(ConstantVariable, self).__init__(**kwargs)
        self.value = value

    def as_proxy(self):
        return self.value

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    def getitem_const(self, arg: VariableTracker):
        return ConstantVariable(
            self.value[arg.as_python_constant()],
            **VariableTracker.propagate([self, arg]),
        )

    @staticmethod
    def is_literal(obj):
        if type(obj) in (int, float, bool, type(None), str):
            return True
        if type(obj) in (list, tuple, set, frozenset):
            return all(ConstantVariable.is_literal(x) for x in obj)
        return False

    def unpack_var_sequence(self, tx):
        try:
            options = VariableTracker.propagate([self])
            return [ConstantVariable(x, **options) for x in self.as_python_constant()]
        except TypeError:
            raise NotImplementedError()


class LambdaVariable(VariableTracker):
    def __init__(self, fn, **kwargs):
        super(LambdaVariable, self).__init__(**kwargs)
        self.fn = fn


class BuiltinVariable(VariableTracker):
    def __init__(self, fn, **kwargs):
        super(BuiltinVariable, self).__init__(**kwargs)
        self.fn = fn

    def __str__(self):
        return f"{self.__class__.__name__}({self.fn.__name__})"

    def python_type(self):
        return type(self.fn)

    def as_python_constant(self):
        return self.fn

    def can_constant_fold_through(self):
        return self.fn in (
            abs,
            all,
            any,
            bool,
            chr,
            callable,
            dict,
            divmod,
            float,
            int,
            len,
            list,
            max,
            min,
            ord,
            pow,
            repr,
            round,
            str,
            sum,
            tuple,
            type,
            math.sqrt,
        )

    def reconstruct(self, codegen):
        name = self.fn.__name__
        assert self.fn.__module__ == "builtins"
        assert name not in codegen.tx.f_globals, "shadowed global"
        return [codegen.create_load_global(name, add=True)]


class ListIteratorVariable(VariableTracker):
    def __init__(self, items, index: int = 0, **kwargs):
        super(ListIteratorVariable, self).__init__(**kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        self.items = items
        self.index = index

    def next_variables(self):
        assert self.mutable_local
        if self.index >= len(self.items):
            raise StopIteration()
        return self.items[self.index], ListIteratorVariable(
            self.items,
            self.index + 1,
            mutable_local=MutableLocal(),
            **VariableTracker.propagate([self]),
        )


class GetAttrVariable(VariableTracker):
    def __init__(self, obj, name, **kwargs):
        super(GetAttrVariable, self).__init__(**kwargs)
        assert isinstance(obj, VariableTracker)
        assert isinstance(name, str)
        self.obj = obj
        self.name = name

    def __str__(self):
        return f"{self.__class__.__name__}({self.obj}, {self.name})"

    def as_proxy(self):
        return getattr(self.obj.as_proxy(), self.name)

    def get_const_attr(self, tx, name):
        if not isinstance(self.obj, NNModuleVariable):
            raise NotImplementedError()
        step1 = tx.get_submodule(self.obj.module_key)
        if self.name not in step1.__dict__:
            raise NotImplementedError()
        step2 = inspect.getattr_static(step1, self.name)
        if name not in step2.__dict__:
            raise NotImplementedError()
        return inspect.getattr_static(step2, name)

    def reconstruct(self, codegen):
        codegen(self.obj)
        return [codegen.create_load_attr(self.name)]


class BaseListVariable(VariableTracker):
    def __init__(self, items, **kwargs):
        super(BaseListVariable, self).__init__(**kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        self.items = items

    def _as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def as_python_constant(self):
        return self.python_type()([x.as_python_constant() for x in self.items])

    def as_proxy(self):
        return self.python_type()(self._as_proxy())

    def getitem_const(self, arg: VariableTracker):
        index = arg.as_python_constant()
        if isinstance(index, slice):
            return self.clone(items=self.items[index]).add_guards(arg.guards)
        else:
            assert isinstance(index, int)
            return self.items[index].add_guards(self.guards).add_guards(arg.guards)

    def unpack_var_sequence(self, tx):
        return list(self.items)


class ListVariable(BaseListVariable):
    def python_type(self):
        return list

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_LIST", len(self.items))]


class TupleVariable(BaseListVariable):
    def python_type(self):
        return tuple

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_TUPLE", len(self.items))]


class SliceVariable(BaseListVariable):
    def as_proxy(self):
        return slice(*self._as_proxy())

    def python_type(self):
        return slice

    def as_python_constant(self):
        return slice(*[x.as_python_constant() for x in self.items])

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_SLICE", len(self.items))]


class ConstDictVariable(VariableTracker):
    def __init__(self, items, **kwargs):
        super(ConstDictVariable, self).__init__(**kwargs)
        if not isinstance(items, collections.OrderedDict):
            assert isinstance(items, dict)
            items = collections.OrderedDict((k, items[k]) for k in sorted(items.keys()))
        self.items = items

    def as_proxy(self):
        return {k: v.as_proxy() for k, v in self.items.items()}

    def python_type(self):
        return dict

    def reconstruct(self, codegen):
        if len(self.items) == 0:
            return [create_instruction("BUILD_MAP", 0)]
        keys = tuple(sorted(self.items.keys()))
        for key in keys:
            codegen(self.items[key])
        return [
            codegen.create_load_const(keys),
            create_instruction("BUILD_CONST_KEY_MAP", len(keys)),
        ]

    def getitem_const(self, arg: VariableTracker):
        index = arg.as_python_constant()
        return self.items[index].add_guards(self.guards).add_guards(arg.guards)


class BaseUserFunctionVariable(VariableTracker):
    def get_filename(self):
        return self.get_code().co_filename

    def get_name(self):
        return self.get_code().co_name


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
        return getattr(self.fn, "__closure__", None) is not None

    def has_self(self):
        return getattr(self.fn, "__self__", None) is not None

    def get_globals(self):
        return self.fn.__globals__

    def bind_args(self, parent, args, kwargs):
        options = VariableTracker.propagate([self])

        def wrap(val):
            if ConstantVariable.is_literal(val):
                return ConstantVariable(val, **options)
            else:
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
        return dict(bound.arguments.items())

    def export_freevars(self, parent, child):
        pass


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
                parent.symbolic_locals.get(c.name, None) for c in self.closure.items
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


class UserDefinedClassVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super(UserDefinedClassVariable, self).__init__(**kwargs)
        self.value = value

    def as_python_constant(self):
        return self.value


class AllowedFunctionOrModuleVariable(VariableTracker):
    """Points to a module or method in torch.*"""

    def __init__(self, value, **kwargs):
        super(AllowedFunctionOrModuleVariable, self).__init__(**kwargs)
        self.value = value

    def as_proxy(self):
        return self.value

    def python_type(self):
        if isinstance(self.value, (torch.Tensor, torch.nn.Module)):
            return type(self.value)
        return super().python_type()

    def as_python_constant(self):
        return self.value

    def can_constant_fold_through(self):
        return getattr(self.value, "__module__", None) == "math"


class PythonModuleVariable(VariableTracker):
    def __init__(self, value: types.ModuleType, **kwargs):
        super(PythonModuleVariable, self).__init__(**kwargs)
        self.value = value

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

    def python_type(self):
        return self.value_type

    def get_const_attr(self, tx, name):
        if name not in getattr(self.value, "__dict__", {}):
            raise NotImplementedError()
        subobj = inspect.getattr_static(self.value, name)
        assert id(subobj) == id(self.value.__dict__[name])
        if not ConstantVariable.is_literal(subobj):
            raise NotImplementedError()
        return subobj


class SuperVariable(VariableTracker):
    def __init__(self, typevar, objvar=None, **kwargs):
        super(SuperVariable, self).__init__(**kwargs)
        self.typevar = typevar
        self.objvar = objvar

    def reconstruct(self, codegen):
        codegen(BuiltinVariable(super))
        codegen(self.typevar)
        if self.objvar is not None:
            codegen(self.objvar)
            return [create_instruction("CALL_FUNCTION", 2)]
        else:
            return [create_instruction("CALL_FUNCTION", 1)]

    def get_const_attr(self, tx, name):
        assert self.objvar, "1-arg super not implemented"
        search_type = self.typevar.as_python_constant()
        # TODO(jansel): there is a small chance this could trigger user code, prevent that
        return getattr(super(search_type, self.objvar.python_type()), name)


class UnknownVariable(VariableTracker):
    """
    It could be anything!
    """


class ClosureVariable(UnknownVariable):
    def __init__(self, name, **kwargs):
        super(ClosureVariable, self).__init__(**kwargs)
        self.name = name

    def reconstruct(self, codegen):
        return [codegen.create_load_closure(self.name)]


def typestr(*objs):
    if len(objs) == 1:
        (obj,) = objs
        if isinstance(obj, VariableTracker):
            return str(obj)
        else:
            return type(obj).__name__
    else:
        return " ".join(map(typestr, objs))
