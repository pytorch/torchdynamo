import inspect
import sys
import types
from typing import Dict
from typing import List

import torch._C

from .. import variables
from ..bytecode_transformation import create_instruction
from ..exc import unimplemented
from ..guards import Guard
from ..guards import GuardBuilder
from ..guards import GuardSource
from ..source import AttrSource
from ..utils import identity
from ..utils import proxy_args_kwargs
from .base import VariableTracker


class SuperVariable(VariableTracker):
    def __init__(self, typevar, objvar=None, **kwargs):
        super(SuperVariable, self).__init__(**kwargs)
        self.typevar = typevar
        self.objvar = objvar

    def reconstruct(self, codegen):
        codegen(variables.BuiltinVariable(super))
        codegen(self.typevar)
        if self.objvar is not None:
            codegen(self.objvar)
            return [create_instruction("CALL_FUNCTION", 2)]
        else:
            return [create_instruction("CALL_FUNCTION", 1)]

    def const_getattr(self, tx, name):
        assert self.objvar, "1-arg super not implemented"
        search_type = self.typevar.as_python_constant()

        # We default to the python type of the object. However,
        # 1. If this is a `type`, then the original object represents the user
        # defined type.
        # 2. If this is `torch._C._TensorMeta`, the original object is the user
        # defined type of a custom tensor subclass.
        # TODO(future PR): figure out how to do this in a less hacky way
        type_to_use = self.objvar.python_type()
        if type_to_use is type or type_to_use is torch._C._TensorMeta:
            type_to_use = self.objvar.value

        # TODO(jansel): there is a small chance this could trigger user code, prevent that
        return getattr(super(search_type, type_to_use), name)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(
            self, args, kwargs.values(), self.objvar, self.typevar
        )
        inner_fn = self.const_getattr(self, name)
        if inner_fn is object.__init__:
            return LambdaVariable(identity, **options)
        elif isinstance(inner_fn, types.FunctionType):
            return variables.UserFunctionVariable(inner_fn, **options).call_function(
                tx, [self.objvar] + args, kwargs
            )
        elif isinstance(inner_fn, types.MethodType):
            return variables.UserMethodVariable(
                inner_fn.__func__, self.objvar, **options
            ).call_function(tx, args, kwargs)
        else:
            unimplemented(f"non-function or method super: {inner_fn}")


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


class NewCellVariable(VariableTracker):
    def __init__(self, **kwargs):
        super(NewCellVariable, self).__init__(**kwargs)


class NewGlobalVariable(VariableTracker):
    def __init__(self, **kwargs):
        super(NewGlobalVariable, self).__init__(**kwargs)


class ContextManagerVariable(VariableTracker):
    pass


class ContextWrappingVariable(ContextManagerVariable):
    """represents torch.{no_grad,enable_grad,set_grad_mode}()"""

    _guards_singleton = {Guard("", GuardSource.GLOBAL, GuardBuilder.GRAD_MODE)}

    def __init__(self, target_value, initial_value=None, **kwargs):
        super(ContextWrappingVariable, self).__init__(**kwargs)
        self.guards = self.guards | self._guards_singleton
        self.target_value = target_value
        if initial_value is None:
            initial_value = self._initial_value()
        self.initial_value = initial_value

    def enter(self, tx):
        self._call_func(tx, self.target_value)
        return variables.ConstantVariable(None, **VariableTracker.propagate(self))

    def exit(self, tx, *args):
        self._call_func(tx, self.initial_value)
        return variables.ConstantVariable(None, **VariableTracker.propagate(self))

    def reconstruct(self, codegen, target_inst=None):
        """
        Generate following Python Bytecode, with a `torch._C._set_grad_enable` call
        Python 3.8
             0 LOAD_GLOBAL              0 (torch)
             2 LOAD_ATTR                1 (_C)
             4 LOAD_METHOD              2 (_set_grad_enable)
             6 LOAD_CONST               1 (False)
             8 CALL_METHOD              1
            10 POP_TOP

            12 SETUP_FINALLY           10 (to 24)

            14 LOAD_GLOBAL              3 (user_inst)
            16 CALL_FUNCTION            0
            18 POP_TOP
            20 POP_BLOCK
            22 BEGIN_FINALLY

            24 LOAD_GLOBAL              0 (torch)
            26 LOAD_ATTR                1 (_C)
            28 LOAD_METHOD              2 (_set_grad_enable)
            30 LOAD_CONST               2 (True)
            32 CALL_METHOD              1
            34 POP_TOP
            36 END_FINALLY
            38 LOAD_CONST               0 (None)
            40 RETURN_VALUE

        Instructions 0-10 and 24-34 call torch._C.set_grad_enable(True/False)

        Python 3.9, 3.10
             0 LOAD_GLOBAL              0 (torch)
             2 LOAD_ATTR                1 (_C)
             4 LOAD_METHOD              2 (_set_grad_enable)
             6 LOAD_CONST               1 (False)
             8 CALL_METHOD              1
            10 POP_TOP

            12 SETUP_FINALLY           22 (to 36)

            14 LOAD_GLOBAL              3 (user_inst)
            16 CALL_FUNCTION            0
            18 POP_TOP
            20 POP_BLOCK

            22 LOAD_GLOBAL              0 (torch)
            24 LOAD_ATTR                1 (_C)
            26 LOAD_METHOD              2 (_set_grad_enable)
            28 LOAD_CONST               2 (True)
            30 CALL_METHOD              1
            32 POP_TOP

            34 JUMP_FORWARD            14 (to 50)

            36 LOAD_GLOBAL              0 (torch)
            38 LOAD_ATTR                1 (_C)
            40 LOAD_METHOD              2 (_set_grad_enable)
            42 LOAD_CONST               2 (True)
            44 CALL_METHOD              1
            46 POP_TOP
            48 RERAISE

            50 LOAD_CONST               0 (None)
            52 RETURN_VALUE

        """
        if self.target_value == self.initial_value:
            return ([], [])

        def set_grad_insts(mode):
            global_torch_source = codegen.tx.import_source("torch")
            attr_source = AttrSource(global_torch_source, self._func_name())
            load_set_grad_enabled_insts = attr_source.reconstruct(codegen)
            return [
                *load_set_grad_enabled_insts,
                codegen.create_load_const(mode),
                create_instruction("CALL_FUNCTION", 1),
                create_instruction("POP_TOP"),
            ]

        init_block = set_grad_insts(self.target_value)
        finally_block = set_grad_insts(self.initial_value)
        setup_final_inst = create_instruction("SETUP_FINALLY", target=finally_block[0])
        prologue = init_block + [setup_final_inst]

        # Generate the epilogue - starts with 20 POP_BLOCK and ends at 34 POP_TOP
        if sys.version_info < (3, 9):
            # Generate the prologue that ends with setup_finally
            epilogue = [
                create_instruction("POP_BLOCK"),
                codegen.create_begin_finally(),
                *finally_block,
                create_instruction("END_FINALLY"),
            ]
        else:
            except_block = set_grad_insts(self.initial_value)
            epilogue = [
                create_instruction("POP_BLOCK"),
                *except_block,
                create_instruction("JUMP_FORWARD", target=target_inst),
                *finally_block,
                create_instruction("RERAISE"),
            ]

        return (prologue, epilogue)

    def _call_func(self, tx, initial_value):
        raise NotImplementedError("_call_func called on base")

    def _func_name(self):
        raise NotImplementedError("_func_name called on base")

    def _initial_value(self):
        raise NotImplementedError("_initial_value called on base")


class GradModeVariable(ContextWrappingVariable):
    def __init__(self, target_value, initial_value=None, **kwargs):
        super(GradModeVariable, self).__init__(
            target_value=target_value, initial_value=initial_value, **kwargs
        )

    def enter(self, tx):
        assert self.initial_value == torch.is_grad_enabled()
        return super(GradModeVariable, self).enter(tx)

    def _call_func(self, tx, value):
        if self.target_value == self.initial_value:
            return
        tx.output.graph.create_node(
            "call_function", torch._C._set_grad_enabled, (value,), {}
        ),
        torch._C._set_grad_enabled(value)

    def _func_name(self):
        return "_C._set_grad_enabled"

    def _initial_value(self):
        return torch.is_grad_enabled()

    def fn_name(self):
        if self.target_value:
            return "enable_grad"
        else:
            return "no_grad"


class ProfileRecordFunctionVariable(ContextWrappingVariable):
    def __init__(self, target_value, initial_value=None, **kwargs):
        kwargs_edited = kwargs
        super(ProfileRecordFunctionVariable, self).__init__(
            target_value=target_value, initial_value=initial_value, **kwargs_edited
        )

    def enter(self, tx):
        self.enter = True
        super(ProfileRecordFunctionVariable, self).enter(tx)

    def exit(self, tx, *args):
        self.enter = False
        super(ProfileRecordFunctionVariable, self).exit(tx)

    def _call_func(self, tx, value):
        if self.enter:
            self.proxy_value = tx.output.create_proxy(
                "call_function", torch.ops.profiler._record_function_enter, (value,), {}
            )
        else:
            tx.output.create_proxy(
                "call_function",
                torch.ops.profiler._record_function_exit,
                (self.proxy_value,),
                {},
            )

    def _func_name(self):
        if self.enter:
            return "torch.ops.profiler._record_function_enter"
        else:
            return "torch.ops.profiler._record_function_exit"

    def _initial_value(self):
        return self.target_value


class WithExitFunctionVariable(VariableTracker):
    def __init__(self, ctx: VariableTracker, target, **kwargs):
        super(WithExitFunctionVariable, self).__init__(**kwargs)
        self.ctx = ctx
        self.target = target

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        assert not kwargs
        return self.ctx.exit(tx, *args)

    def reconstruct(self, codegen):
        # Note here we reconstruct the context manager rather than the
        # exit function.  The handler generated by BlockStackEntry
        # will re-enter the context in the resume function.
        output = AttrSource(
            codegen.tx.import_source("torch"), self.ctx.fn_name()
        ).reconstruct(codegen)

        if codegen.tx.output.partial_convert:
            output.extend(
                [
                    create_instruction("CALL_FUNCTION", 0),
                    create_instruction("SETUP_WITH", target=self.target),
                    create_instruction("POP_TOP"),
                ]
            )
        return output


class InspectSignatureVariable(VariableTracker):
    """represents inspect.signature(...)"""

    @staticmethod
    def create(callable, **kwargs):
        if kwargs:
            unimplemented(f"inspect.signature with {kwargs}")
        return InspectSignatureVariable(callable)

    def __init__(self, inspected, **kwargs):
        super(InspectSignatureVariable, self).__init__(**kwargs)
        self.inspected = inspected


class AutogradFunctionVariable(VariableTracker):
    """represents a torch.autograd.Function subclass"""

    def __init__(self, fn_cls, **kwargs):
        super().__init__(**kwargs)
        self.fn_cls = fn_cls

    def call_apply(self, tx, args, kwargs):
        requires_grad = False

        def visit(node):
            nonlocal requires_grad
            if isinstance(node, variables.TensorVariable):
                if node.requires_grad is not False:
                    requires_grad = True
            if isinstance(node, variables.NNModuleVariable):
                if node.is_training(tx):
                    requires_grad = True
            return node

        VariableTracker.apply(visit, (args, kwargs))

        if requires_grad and torch.is_grad_enabled():
            # TODO(jansel): handle this in training mode
            unimplemented("autograd.Function with requires_grad")

        args = [BlackHoleVariable()] + list(args)
        options = VariableTracker.propagate(self, args, kwargs.values())
        return variables.UserFunctionVariable(
            self.fn_cls.forward, **options
        ).call_function(tx, args, kwargs)


class BlackHoleVariable(VariableTracker):
    """A autograd.function context that just ignores everything (for forward extraction)"""

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        assert name in ("__setattr__", "save_for_backward"), name
        return variables.ConstantVariable(
            None, **VariableTracker.propagate(self, args, kwargs.values())
        )


class LambdaVariable(VariableTracker):
    def __init__(self, fn, **kwargs):
        super(LambdaVariable, self).__init__(**kwargs)
        self.fn = fn

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        return self.fn(*args, **kwargs).add_options(self)


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

    def const_getattr(self, tx, name):
        if not isinstance(self.obj, variables.NNModuleVariable):
            raise NotImplementedError()
        step1 = tx.output.get_submodule(self.obj.module_key)
        if self.name not in step1.__dict__:
            raise NotImplementedError()
        step2 = inspect.getattr_static(step1, self.name)
        if name not in step2.__dict__:
            raise NotImplementedError()
        return inspect.getattr_static(step2, name)

    def reconstruct(self, codegen):
        codegen(self.obj)
        return codegen.create_load_attrs(self.name)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":

        # This variable is True when it corresponds to user code such as
        #
        #   super().__torch_function__(...)
        #
        # and the super().__torch_function__ attribute resolves
        # to torch.Tensor.__torch_function__.
        is_original_tensor_torch_function = (
            self.name == "__torch_function__"
            and isinstance(self.obj, SuperVariable)
            # for now, only support one level of inheritance
            and len(self.obj.objvar.value.__mro__) > 1
            and self.obj.objvar.value.__mro__[1] == torch.Tensor
        )
        if is_original_tensor_torch_function:
            # Instead of tracing inside torch.Tensor.__torch_function__,
            # record the `call_function` or `call_method` call into the graph.
            from . import TensorVariable
            from . import TorchVariable

            original_torch_or_getattr_variable = args[0]
            new_args = args[2].items
            new_kwargs = args[3].items
            options = VariableTracker.propagate(self, new_args, new_kwargs.values())
            # Disable __torch_function__ here to prevent the clone of the
            # example tensor from going into the override.
            with torch._C.DisableTorchFunction():
                if isinstance(args[0], TorchVariable):
                    return TensorVariable.create(
                        tx=tx,
                        proxy=tx.output.create_proxy(
                            "call_function",
                            original_torch_or_getattr_variable.value,
                            *proxy_args_kwargs(new_args, new_kwargs),
                        ),
                        **options,
                    )
                elif isinstance(args[0], GetAttrVariable):
                    return TensorVariable.create(
                        tx=tx,
                        proxy=tx.output.create_proxy(
                            "call_method",
                            original_torch_or_getattr_variable.name,
                            *proxy_args_kwargs(new_args, new_kwargs),
                        ),
                        **options,
                    )
                else:
                    unimplemented(
                        f"GetAttrVariable.call_function original __torch_function__ {args}"
                    )

        if isinstance(self.obj, AutogradFunctionVariable) and self.name == "apply":
            return self.obj.call_apply(tx, args, kwargs).add_options(self)
        return self.obj.call_method(tx, self.name, args, kwargs).add_options(self)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if (
            name == "__len__"
            and isinstance(self.obj, InspectSignatureVariable)
            and self.name == "parameters"
        ):
            return variables.ConstantVariable(
                self.obj.inspected.num_parameters(),
                **VariableTracker.propagate(self, self.obj, self.obj.inspected),
            )
        return super(GetAttrVariable, self).call_method(tx, name, args, kwargs)


class PythonModuleVariable(VariableTracker):
    def __init__(self, value: types.ModuleType, **kwargs):
        super(PythonModuleVariable, self).__init__(**kwargs)
        self.value = value

    def python_type(self):
        return types.ModuleType


class SkipFilesVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        if inspect.getattr_static(self.value, "_torchdynamo_disable", False):
            unimplemented("call torchdynamo.disable() wrapped function")
        else:
            try:
                path = inspect.getfile(self.value)
            except TypeError:
                path = f"Builtin {self.value.__name__}"
            unimplemented("call_function in skip_files " + path)


class TypingVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__getitem__" and len(args) == 1:
            return variables.ConstantVariable(
                self.value[args[0].as_python_constant()],
                **VariableTracker.propagate(self, args),
            )
        unimplemented("typing")


class NumpyVariable(VariableTracker):
    """
    Wrapper around `numpy.*` for better error messages.
    """

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        unimplemented("numpy")

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        unimplemented("numpy")
