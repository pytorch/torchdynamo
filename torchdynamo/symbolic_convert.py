import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import operator
import os
import sys
import types
import typing
from numbers import Real
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from unittest.mock import patch

import torch
from torch import fx

from . import config
from .allowed_functions import is_allowed
from .bytecode_analysis import livevars_analysis
from .bytecode_transformation import Instruction
from .bytecode_transformation import cleaned_instructions
from .bytecode_transformation import create_instruction
from .bytecode_transformation import unique_id
from .guards import Guard
from .guards import GuardBuilder
from .guards import GuardSource
from .resume_execution import ContinueExecutionCache
from .utils import count_calls
from .utils import istensor
from .utils import istype
from .utils import typestr
from .variable_tracker import AllowedFunctionOrModuleVariable, FunctionConstantWrapper
from .variable_tracker import BaseListVariable
from .variable_tracker import BasicTypeVariable
from .variable_tracker import BuiltinVariable
from .variable_tracker import ConstDictVariable
from .variable_tracker import ConstantVariable
from .variable_tracker import GetAttrVariable
from .variable_tracker import ListIteratorVariable
from .variable_tracker import ListVariable
from .variable_tracker import NNModuleVariable
from .variable_tracker import PythonModuleVariable
from .variable_tracker import SliceVariable
from .variable_tracker import TensorVariable
from .variable_tracker import TracingSupported
from .variable_tracker import TupleVariable
from .variable_tracker import UnknownVariable
from .variable_tracker import UnsupportedVariable
from .variable_tracker import UserFunctionVariable
from .variable_tracker import UserMethodVariable
from .variable_tracker import VariableTracker
from .mutation_guard import real_type as type

counters = collections.defaultdict(collections.Counter)


def proxy_args_kwargs(args, kwargs):
    try:
        proxy_args = tuple(arg.as_proxy() for arg in args)
        proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return proxy_args, proxy_kwargs
    except AttributeError:  # "no attribute 'as_proxy'"
        raise unimplemented(
            f"call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}"
        )


def unimplemented(msg: str):
    counters["unimplemented"][msg] += 1
    assert msg != os.environ.get("BREAK", False)
    raise NotImplementedError(msg)


def warning(msg: str):
    counters["warnings"][msg] += 1


def stack_op(fn: typing.Callable):
    nargs = len(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def impl(self: "InstructionTranslatorBase", inst: Instruction):
        inputs = self.popn(nargs)

        options = VariableTracker.propagate(inputs)
        if any(isinstance(i, TensorVariable) for i in inputs):
            val = TensorVariable(
                self.create_proxy(
                    "call_function", fn, tuple(i.as_proxy() for i in inputs), {}
                ),
                **options,
            )
        elif all(isinstance(i, ConstantVariable) for i in inputs):
            # constant fold
            val = ConstantVariable(fn(*[i.value for i in inputs]), **options)
        elif isinstance(inputs[0], ConstantVariable) and fn is operator.getitem:
            base, item = inputs
            val = base.getitem_const(item)
        else:
            unimplemented(f"stack_op {typestr(*inputs)}")

        self.push(val)

    return impl


def generic_jump(truth_fn: typing.Callable, push: bool):
    def inner(self: "InstructionTranslatorBase", inst: Instruction):
        value = self.pop()
        self.guards.update(value.guards)
        if isinstance(value, (AllowedFunctionOrModuleVariable, ConstantVariable)):
            if truth_fn(value.value):
                push and self.push(value)
                self.jump(inst)

        elif isinstance(value, TensorVariable) and self.should_compile_partial_graph():
            # compile a partial subgraph prefix then jump into user code
            self.push(value)
            self.compile_partial_subgraph()
            self.pop()

            if_next = self.create_call_resume_at(self.next_instruction)
            push and self.push(value)
            if_jump = self.create_call_resume_at(inst.target)

            self.output_instructions.extend(
                [(create_instruction(inst.opname, target=if_jump[0]))]
                + if_next
                + if_jump
            )
        else:
            unimplemented(f"generic_jump {typestr(value)}")

    return inner


@dataclasses.dataclass
class Arg:
    name: str
    example: Any

    def get_examples(self):
        return [self.example]

    def __len__(self):
        return 1


@dataclasses.dataclass
class LocalArg(Arg):
    def load(self, tracer):
        return [tracer.create_load(self.name)]


@dataclasses.dataclass
class GlobalArg(Arg):
    def load(self, tracer):
        return [tracer.create_load_global(self.name)]


@dataclasses.dataclass
class LocalListArg(Arg):
    length: int

    def load(self, tracer):
        return [
            tracer.create_load(self.name),
            create_instruction("UNPACK_SEQUENCE", self.length),
        ]

    def get_examples(self):
        return list(reversed(self.example))

    def __len__(self):
        return self.length


class InstructionTranslatorBase(fx.Tracer):
    def create_load(self, name):
        if name in (self.code_options["co_cellvars"] or []):
            return create_instruction(
                "LOAD_DEREF", self.code_options["co_cellvars"].index(name), name
            )
        assert name in self.code_options["co_varnames"]
        return create_instruction(
            "LOAD_FAST", self.code_options["co_varnames"].index(name), name
        )

    def create_store(self, name):
        if name in (self.code_options["co_cellvars"] or []):
            return create_instruction(
                "STORE_DEREF", self.code_options["co_cellvars"].index(name), name
            )
        assert name in self.code_options["co_varnames"]
        return create_instruction(
            "STORE_FAST", self.code_options["co_varnames"].index(name), name
        )

    def create_load_global(self, name, add=False):
        if add and name not in self.code_options["co_names"]:
            self.code_options["co_names"] = tuple(self.code_options["co_names"]) + (
                name,
            )
        assert name in self.code_options["co_names"]
        return create_instruction(
            "LOAD_GLOBAL", self.code_options["co_names"].index(name), name
        )

    def create_load_const(self, value):
        co_consts = self.code_options["co_consts"]
        assert istype(co_consts, tuple)
        if value not in co_consts:
            co_consts = co_consts + (value,)
            self.code_options["co_consts"] = co_consts
        return create_instruction("LOAD_CONST", co_consts.index(value), value)

    def wrap_local(self, name, value):
        """
        Turn an arg/input to the frame into a VariableTracker instance
        """
        if istensor(value):
            self.graphargs.append(LocalArg(name, value))
            return TensorVariable(
                proxy=self.create_graph_input(name, type(value)),
                state=TracingSupported.YES,
                guards={Guard(name, GuardSource.LOCAL, GuardBuilder.TENSOR_MATCH)},
                **TensorVariable.specialize(value),
            )
        elif isinstance(value, torch.nn.Module):
            key = f"{name}_{next(self.cnt)}"
            self.nn_modules[key] = value
            return NNModuleVariable(
                module_key=key,
                state=TracingSupported.YES,
                guards={Guard(name, GuardSource.LOCAL, GuardBuilder.ID_MATCH)},
            )
        elif value is None or istype(value, bool):
            # For these, just specialize on exact value
            return ConstantVariable(
                value=value,
                guards={Guard(name, GuardSource.LOCAL, GuardBuilder.ID_MATCH)},
            )
        elif (
            istype(value, int)
            or (istype(value, float) and value in (-1.0, 0.0, 1.0, 2.0))
            or (istype(value, (list, tuple)) and len(value) == 0)
        ):
            # For these, just specialize on exact value
            return ConstantVariable(
                value=value,
                guards={Guard(name, GuardSource.LOCAL, GuardBuilder.EQUALS_MATCH)},
            )
        elif istype(value, float):
            self.graphargs.append(LocalArg(name, value))
            return BasicTypeVariable(
                proxy=self.create_graph_input(name, type(value)),
                state=TracingSupported.UNKNOWN,
                guards={Guard(name, GuardSource.LOCAL, GuardBuilder.TYPE_MATCH)},
            )
        elif istype(value, (tuple, list)) and all(istensor(x) for x in value):
            guards = {Guard(name, GuardSource.LOCAL, GuardBuilder.FIXED_TENSOR_LIST)}
            self.graphargs.append(LocalListArg(name, value, len(value)))
            # LocalListArg uses UNPACK_SEQUENCE, which reverses the order
            items = [
                TensorVariable(
                    proxy=self.create_graph_input(f"{name}_{idx}", type(v)),
                    state=TracingSupported.YES,
                    guards=guards,
                    **TensorVariable.specialize(v),
                )
                for idx, v in reversed(list(enumerate(value)))
            ]
            cls = {tuple: TupleVariable, list: ListVariable}[type(value)]
            return cls(list(reversed(items)), guards=guards)
        else:
            return UnsupportedVariable(
                value,
                state=TracingSupported.NO,
                guards={Guard(name, GuardSource.LOCAL, GuardBuilder.TYPE_MATCH)},
            )

    def prune_dead_locals(self):
        reads = livevars_analysis(self.instructions, self.current_instruction)
        self.symbolic_locals = collections.OrderedDict(
            [
                (k, v)
                for k, v in self.symbolic_locals.items()
                if k in reads
                or v in self.stack
                or not isinstance(v, (TensorVariable, ConstantVariable))
            ]
        )

    def create_graph_input(self, name, type_expr=None):
        placeholders = [n for n in self.graph.nodes if n.op == "placeholder"]

        # unique
        used_names = {n.name for n in placeholders}
        if name in used_names:
            for i in itertools.count():
                if f"{name}_{i}" not in used_names:
                    name = f"{name}_{i}"
                    break

        if placeholders:
            ctx = self.graph.inserting_after(placeholders[-1])
        else:
            ctx = self.graph.inserting_before(None)
        with ctx:
            return self.create_proxy("placeholder", name, (), {}, type_expr=type_expr)

    def call_function(self, fn, args, kwargs):
        assert isinstance(fn, VariableTracker)
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        all_const_inputs = all(
            isinstance(x, ConstantVariable)
            for x in itertools.chain(args, kwargs.values())
        )
        options = VariableTracker.propagate(
            [
                fn,
            ]
            + list(args)
            + list(kwargs.values())
        )

        if (
            isinstance(fn, AllowedFunctionOrModuleVariable)
            and fn.value in config.constant_functions
        ):
            assert not args and not kwargs
            self.push(ConstantVariable(config.constant_functions[fn.value], **options))
        elif (
            isinstance(fn, AllowedFunctionOrModuleVariable)
            and fn.is_basic_math()
            and all_const_inputs
        ):
            # constant fold
            self.push(
                ConstantVariable(
                    fn.value(
                        *[x.value for x in args],
                        **{k: v.value for k, v in kwargs.items()},
                    ),
                    **options,
                )
            )
        elif isinstance(fn, AllowedFunctionOrModuleVariable):
            self_should_be_none = getattr(fn.value, "__self__", None)
            if self_should_be_none is not None:
                # weird ones like torch.nn.functional.avg_pool2d have __self__
                assert isinstance(
                    self_should_be_none, types.ModuleType
                ) and self_should_be_none.__name__ == getattr(
                    fn.value, "__module__", None
                )
            self.push(
                TensorVariable(
                    proxy=self.create_proxy(
                        "call_function", fn.value, *proxy_args_kwargs(args, kwargs)
                    ),
                    **options,
                )
            )
        elif isinstance(fn, GetAttrVariable):
            name = fn.name
            obj = fn.obj
            args = [obj] + list(args)
            self.push(
                TensorVariable(
                    proxy=self.create_proxy(
                        "call_method", name, *proxy_args_kwargs(args, kwargs)
                    ),
                    **options,
                )
            )
        elif isinstance(fn, NNModuleVariable):
            mod = self.get_submodule(fn.module_key)
            if isinstance(mod, torch.nn.Sequential):
                if (
                    len(args) != 1
                    or len(kwargs) != 0
                    or mod.__class__.forward is not torch.nn.Sequential.forward
                ):
                    unimplemented("custom Sequential")
                # unroll Sequential()
                options = VariableTracker.propagate([fn])
                (arg,) = args
                for idx, submod in enumerate(mod):
                    # Just this would work most of the time, but not when using names
                    # key = f"{fn.module_key}.{idx}"
                    key = unique_id(f"{fn.module_key.replace('.', '_')}_{idx}")
                    self.nn_modules[key] = submod
                    self.call_function(NNModuleVariable(key, **options), [arg], {})
                    arg = self.pop()
                self.push(arg)
            elif is_allowed(mod.__class__):
                self.push(
                    TensorVariable(
                        proxy=self.create_proxy(
                            "call_module",
                            fn.module_key,
                            *proxy_args_kwargs(args, kwargs),
                        ),
                        **options,
                    )
                )
            else:
                forward = mod.__class__.forward
                assert forward is not torch.nn.Module.forward
                self.inline_user_function(fn.guards, forward, [fn] + args, kwargs)
        elif isinstance(fn, UserFunctionVariable):
            self.guards.update(fn.guards)
            try:
                self.inline_user_function(
                    fn.guards, fn.fn, fn.self_args() + args, kwargs
                )
            except NotImplementedError:
                if not self.should_compile_partial_graph():
                    raise
                self.partial_subgraph_and_call(fn, fn.self_args() + args, kwargs)
        elif isinstance(fn, BuiltinVariable):
            allargs = args + list(kwargs.values())
            constant_args = all(isinstance(x, ConstantVariable) for x in allargs)
            if fn.fn is range and constant_args:
                items = [
                    ConstantVariable(x, **options)
                    for x in range(
                        *[x.value for x in args],
                        **{k: v.value for k, v in kwargs.items()},
                    )
                ]
                self.push(ListVariable(items, **options))
            elif fn.fn is iter and args and isinstance(args[0], BaseListVariable):
                assert not kwargs and len(args) == 1
                self.push(ListIteratorVariable(args[0].items, **options))
            elif fn.fn is len:
                assert not kwargs and len(args) == 1
                arg = args[0]
                if isinstance(arg, TensorVariable):
                    self.push(
                        BasicTypeVariable(
                            self.create_proxy("call_function", len, (arg.proxy,), {}),
                            **options,
                        )
                    )
                elif isinstance(
                    arg, (ConstantVariable, BaseListVariable, ConstDictVariable)
                ):
                    item = (
                        arg.value
                        if isinstance(arg, ConstantVariable)
                        else tuple(arg.as_proxy())
                    )
                    self.push(ConstantVariable(len(item), **options))
                else:
                    unimplemented(f"`len` with arg type {arg}")
            elif fn.fn is isinstance:
                assert not kwargs and len(args) == 2
                arg, isinstance_type = args
                arg_type = arg.python_type()
                isinstance_type = isinstance_type.python_value()
                try:
                    val = issubclass(arg_type, isinstance_type)
                except TypeError:
                    val = arg_type is isinstance_type
                self.push(ConstantVariable(val, **options))
            elif fn.fn is float:
                assert not kwargs and len(args) == 1
                try:
                    self.push(ConstantVariable(float(args[0].value), **options))
                except (TypeError, AttributeError):
                    unimplemented("float constructor with non-const argument")
            elif self.should_compile_partial_graph():
                warning(f"breaking graph on call({fn.fn})")
                self.partial_subgraph_and_call(fn, args, kwargs)
            else:
                unimplemented(f"builtin call {fn.fn}")
        elif isinstance(fn, FunctionConstantWrapper):
            self.push(fn.call_const(args, kwargs))
        elif self.should_compile_partial_graph():
            warning(f"breaking graph on call({fn})")
            self.partial_subgraph_and_call(fn, args, kwargs)
        else:
            unimplemented(f"call({fn})")

    def partial_subgraph_and_call(self, fn, args, kwargs):
        keys = list(kwargs.keys())
        keys.sort(key=lambda k: (self.is_const_var(kwargs[k]), k))
        args_and_kwargs = list(args) + [kwargs[k] for k in keys]
        self.push(fn)
        self.push_many(args_and_kwargs)
        self.compile_partial_subgraph()
        self.popn(len(args_and_kwargs) + 1)
        if not kwargs:
            self.grow_stack_to(len(self.stack) + len(args_and_kwargs) + 2)
            self.output_instructions.append(
                create_instruction("CALL_FUNCTION", len(args_and_kwargs))
            )
        else:
            self.grow_stack_to(len(self.stack) + len(args_and_kwargs) + 3)
            self.output_instructions.extend(
                [
                    self.create_load_const(tuple(keys)),
                    create_instruction("CALL_FUNCTION_KW", len(args_and_kwargs)),
                ]
            )
        self.push(UnknownVariable())
        self.output_instructions.extend(
            self.create_call_resume_at(self.next_instruction)
        )

    def inline_user_function(self, guards, fn, args, kwargs):
        """
        A call to some user defined function by inlining it.
        """
        self.guards.update(guards)
        self.push(InliningInstructionTranslator.inline_call(self, fn, args, kwargs))

    def step(self):
        """Process exactly one instruction, return False we should exit"""
        inst = self.instructions[self.instruction_pointer]
        self.current_instruction = inst
        self.instruction_pointer += 1
        if self.instruction_pointer < len(self.instructions):
            self.next_instruction = self.instructions[self.instruction_pointer]
        else:
            self.instruction_pointer = None
            self.next_instruction = None

        if len(self.stack) == 0 and self.should_compile_partial_graph():
            self.checkpoint = inst, self.copy_graphstate()

        try:
            if not hasattr(self, inst.opname):
                unimplemented(f"missing: {inst.opname}")
            getattr(self, inst.opname)(inst)
            # print(len(self.stack), inst.opname)
            return (
                inst.opname != "RETURN_VALUE" and self.instruction_pointer is not None
            )
        except NotImplementedError:
            if self.checkpoint:
                continue_inst, state = self.checkpoint
                self.restore_graphstate(state)
                self.compile_partial_subgraph()
                self.output_instructions.append(
                    create_instruction("JUMP_ABSOLUTE", target=continue_inst)
                )
            else:
                raise

    def run(self):
        try:
            while self.step():
                pass
        except NotImplementedError:
            raise
        except Exception as e:
            sys.stderr.write(
                f"ERROR FROM offset={self.current_instruction.offset} "
                f"filename {self.code_options.get('co_filename')} "
                f"{self.code_options.get('co_firstlineno')} {typestr(e)}\n"
            )
            raise

    def push(self, val):
        assert val is None or isinstance(
            val, VariableTracker
        ), f"push expects VariableTracker, got {typestr(val)}"
        self.stack.append(val)
        self.grow_stack_to(len(self.stack))

    def push_many(self, vals):
        for val in vals:
            self.push(val)

    def pop(self):
        return self.stack.pop()

    def popn(self, n):
        return list(reversed([self.pop() for _ in range(n)]))

    def LOAD_FAST(self, inst):
        assert inst.argval not in (self.code_options["co_cellvars"] or [])
        if inst.argval not in self.symbolic_locals:
            unimplemented("undefined local")
        self.push(self.symbolic_locals[inst.argval])
        if inst.argval.startswith("___stack"):
            self.symbolic_locals.pop(inst.argval)

    def LOAD_DEREF(self, inst):
        if inst.argval not in self.code_options["co_cellvars"]:
            unimplemented("LOAD_DEREF freevar")
        if inst.argval not in self.symbolic_locals:
            unimplemented("undefined local deref")
        self.push(self.symbolic_locals[inst.argval])

    def STORE_FAST(self, inst):
        self.symbolic_locals[inst.argval] = self.pop()

    def LOAD_CONST(self, inst):
        self.push(ConstantVariable(value=inst.argval, state=TracingSupported.UNKNOWN))

    def LOAD_GLOBAL(self, inst):
        try:
            value = self.f_globals[inst.argval]
        except KeyError:
            return self.load_builtin(inst)
        if is_allowed(value):
            self.push(
                AllowedFunctionOrModuleVariable(
                    value=value,
                    state=TracingSupported.YES,
                    guards={
                        Guard(
                            inst.argval, GuardSource.GLOBAL, GuardBuilder.FUNCTION_MATCH
                        )
                    },
                    global_name=inst.argval,
                )
            )
        elif istensor(value):
            # turn a load of a global tensor into an arg for the graph
            self.graphargs.append(GlobalArg(inst.argval, value))
            self.push(
                TensorVariable(
                    proxy=self.create_graph_input(inst.argval),
                    state=TracingSupported.YES,
                    guards={
                        Guard(
                            inst.argval, GuardSource.GLOBAL, GuardBuilder.TENSOR_MATCH
                        )
                    },
                    global_name=inst.argval,
                    **TensorVariable.specialize(value),
                )
            )
        elif istype(value, types.FunctionType):
            self.push(
                UserFunctionVariable(
                    value,
                    guards={
                        Guard(
                            inst.argval, GuardSource.GLOBAL, GuardBuilder.FUNCTION_MATCH
                        )
                    },
                    global_name=inst.argval,
                )
            )
        elif isinstance(value, bool):
            self.push(
                ConstantVariable(
                    value=self.f_globals[inst.argval],
                    state=TracingSupported.UNKNOWN,
                    guards={
                        Guard(inst.argval, GuardSource.GLOBAL, GuardBuilder.ID_MATCH)
                    },
                    global_name=inst.argval,
                )
            )
        elif istype(value, types.ModuleType):
            self.push(
                PythonModuleVariable(
                    value,
                    guards={
                        Guard(
                            inst.argval, GuardSource.GLOBAL, GuardBuilder.FUNCTION_MATCH
                        )
                    },
                    global_name=inst.argval,
                )
            )
        elif isinstance(value, torch.nn.Module):
            key = unique_id(inst.argval)
            self.nn_modules[key] = value
            self.push(
                NNModuleVariable(
                    key,
                    guards={
                        Guard(inst.argval, GuardSource.GLOBAL, GuardBuilder.ID_MATCH)
                    },
                    global_name=inst.argval,
                )
            )
        else:
            self.push(
                UnsupportedVariable(
                    value,
                    guards={
                        Guard(inst.argval, GuardSource.GLOBAL, GuardBuilder.TYPE_MATCH)
                    },
                    global_name=inst.argval,
                )
            )

    def load_builtin(self, inst):
        assert inst.argval in self.f_builtins
        self.push(
            BuiltinVariable(self.f_builtins[inst.argval], global_name=inst.argval)
        )

    def jump(self, inst):
        self.instruction_pointer = self.indexof[id(inst.target)]

    JUMP_FORWARD = jump
    JUMP_ABSOLUTE = jump

    POP_JUMP_IF_FALSE = generic_jump(operator.not_, False)
    POP_JUMP_IF_TRUE = generic_jump(operator.truth, False)
    JUMP_IF_FALSE_OR_POP = generic_jump(operator.not_, True)
    JUMP_IF_TRUE_OR_POP = generic_jump(operator.truth, True)

    def FOR_ITER(self, inst):
        it = self.pop()
        if isinstance(it, ListIteratorVariable):
            self.guards.update(it.guards)
            try:
                val, next_iter = it.next_variables()
                self.push(next_iter)
                self.push(val)
            except StopIteration:
                self.jump(inst)
        else:
            unimplemented(f"FOR_ITER {typestr(it)}")

    def SETUP_LOOP(self, inst):
        pass  # TODO(jansel): support blocks

    def POP_BLOCK(self, inst):
        pass  # TODO(jansel): support blocks

    def COMPARE_OP(self, inst):
        left, right = self.popn(2)
        options = VariableTracker.propagate([left, right])
        op = inst.argval
        supported_is_const = {
            "is": operator.is_,
            "is not": operator.is_not,
            "==": operator.eq,
            "!=": operator.ne,
        }
        supported_tensors = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
        }
        supported_any = dict(
            itertools.chain(supported_tensors.items(), supported_is_const.items())
        )
        if (
            isinstance(left, (TensorVariable, NNModuleVariable))
            and isinstance(right, ConstantVariable)
            and right.value is None
            and op in supported_is_const
        ):
            self.push(
                ConstantVariable(
                    supported_is_const[op](object(), right.value), **options
                )
            )
        elif (
            isinstance(left, TensorVariable) or isinstance(right, TensorVariable)
        ) and op in supported_tensors:
            self.push(
                TensorVariable(
                    supported_tensors[op](left.as_proxy(), right.as_proxy()),
                    **options,
                )
            )
        elif (
            isinstance(left, ConstantVariable)
            and isinstance(right, ConstantVariable)
            and op in supported_any
        ):
            # constant fold
            self.push(
                ConstantVariable(supported_any[op](left.value, right.value), **options)
            )
        elif (
            isinstance(left, (AllowedFunctionOrModuleVariable, ConstantVariable))
            and isinstance(right, (AllowedFunctionOrModuleVariable, ConstantVariable))
            and op in supported_is_const
        ):
            self.push(
                ConstantVariable(
                    supported_is_const[op](left.value, right.value), **options
                )
            )
        else:
            unimplemented(f"COMPARE_OP {typestr(left)} {op} {typestr(right)}")

    def CALL_FUNCTION(self, inst):
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})

    def GET_ITER(self, inst):
        self.call_function(BuiltinVariable(iter), [self.pop()], {})

    def CALL_FUNCTION_EX(self, inst):
        if inst.argval == 0:
            kwargsvars = ConstDictVariable({})
            argsvars = self.pop()
        elif inst.argval == 1:
            kwargsvars = self.pop()
            argsvars = self.pop()
        else:
            unimplemented("CALL_FUNCTION_EX")
        fn = self.pop()

        if (
            isinstance(fn, GetAttrVariable)
            and isinstance(fn.obj, TensorVariable)
            and fn.name == "view"
            and isinstance(argsvars, (ConstantVariable, TensorVariable))
        ):
            # Hack to handle special case in some bert models.  Converts
            # x.view(*shape) into x.view(shape), which is correct for view()
            # but not generally.  See test_transpose_for_scores().
            argsvars = TupleVariable([argsvars])

        if not isinstance(argsvars, BaseListVariable) or not isinstance(
            kwargsvars, ConstDictVariable
        ):
            unimplemented(f"non-static call {typestr(argsvars)} {typestr(kwargsvars)}")
        self.call_function(fn, argsvars.items, kwargsvars.items)

    def CALL_FUNCTION_KW(self, inst):
        argnames = self.pop()
        args = self.popn(inst.argval)
        fn = self.pop()
        assert isinstance(argnames, ConstantVariable)
        argnames = argnames.value
        args, kwargs = args[: -len(argnames)], args[-len(argnames) :]
        kwargs = dict(zip(argnames, kwargs))
        assert len(kwargs) == len(argnames)
        self.call_function(fn, args, kwargs)

    def get_submodule(self, keys):
        assert keys
        obj = self.nn_modules
        for k in keys.split("."):
            if isinstance(obj, dict):
                obj = obj[k]
            else:
                obj = getattr(obj, k)
        return obj

    def LOAD_METHOD(self, inst):
        self.LOAD_ATTR(inst)
        self.push(None)

    def CALL_METHOD(self, inst):
        args = self.popn(inst.argval)
        dummy = self.pop()
        assert dummy is None
        fn = self.pop()
        self.call_function(fn, args, {})

    def LOAD_ATTR(self, inst):
        obj = self.pop()
        name = inst.argval
        options = VariableTracker.propagate([obj])
        if isinstance(obj, NNModuleVariable):
            base = self.get_submodule(obj.module_key)
            key = f"{obj.module_key}.{name}"
            try:
                subobj = inspect.getattr_static(base, name)
            except AttributeError:
                # TODO(jansel): figure out how to remove this
                subobj = self.get_submodule(key)
            if istensor(subobj):
                options.update(TensorVariable.specialize(subobj))
                self.push(
                    TensorVariable(
                        proxy=self.create_proxy("get_attr", key, tuple(), {}), **options
                    )
                )
            elif isinstance(subobj, torch.nn.Module):
                self.push(NNModuleVariable(key, **options))
            elif istype(subobj, (int, float, bool, type(None))):
                # Assumes module attributes are constant
                # TODO(jansel): add guards?
                self.push(
                    ConstantVariable(
                        subobj,
                        **options,
                    )
                )
            elif is_allowed(subobj):
                self.push(AllowedFunctionOrModuleVariable(subobj, **options))
            elif istype(subobj, property):
                unimplemented("property")
            elif istype(subobj, classmethod):
                unimplemented("classmethod")
            elif istype(subobj, staticmethod):
                self.push(UserFunctionVariable(subobj.__get__(base), **options))
            elif istype(subobj, types.FunctionType) and name not in base.__dict__:
                self.push(UserMethodVariable(subobj, obj, **options))
            elif istype(subobj, types.FunctionType) and name in base.__dict__:
                self.push(UserFunctionVariable(subobj, **options))
            elif callable(subobj):
                base = self.get_submodule(obj.module_key)
                method = getattr(base.__class__, name, None)
                if isinstance(method, types.FunctionType):
                    self.push(UserMethodVariable(method, obj, **options))
                else:
                    unimplemented("nn.Module callable")
            elif isinstance(subobj, (list, tuple)):
                subobj_proxy = self.create_proxy("get_attr", key, tuple(), {})
                # If the container has exclusively nn.Modules, transform
                # each module into a torchdynamo.NNModuleVariable;
                # similarly for Tensors/torchdynamo.TensorVariable
                output = []
                for i, item in enumerate(subobj):
                    if isinstance(item, torch.nn.Module):
                        key = f"{obj.module_key}_{name}_{i}"
                        self.nn_modules[key] = item
                        output.append(
                            NNModuleVariable(
                                module_key=key,
                                **options,
                            )
                        )
                    elif istensor(item):
                        output.append(
                            TensorVariable(
                                proxy=self.create_proxy(
                                    "call_function",
                                    operator.getitem,
                                    (subobj_proxy, i),
                                    {},
                                ),
                                **options,
                            )
                        )
                if len(output) != len(subobj):
                    unimplemented("non-nn.Module items in list")
                tracker_type = (
                    ListVariable if isinstance(subobj, list) else TupleVariable
                )
                self.push(tracker_type(output, **options))
            else:
                self.push(GetAttrVariable(obj, name, **options))
        elif isinstance(obj, TensorVariable):
            const_result = obj.const_attr(name)
            if const_result is not None:
                self.push(const_result)
            else:
                self.push(GetAttrVariable(obj, name, **options))
        elif isinstance(obj, AllowedFunctionOrModuleVariable):
            self.push(
                AllowedFunctionOrModuleVariable(
                    value=getattr(obj.value, name), **options
                )
            )
        elif isinstance(obj, PythonModuleVariable):
            member = obj.value.__dict__[name]
            if is_allowed(member):
                self.push(AllowedFunctionOrModuleVariable(member, **options))
            elif callable(member):
                self.push(UserFunctionVariable(member, **options))
            else:
                unimplemented("PythonModuleVariable attribute")
        elif obj.has_const_attr(self, name) and obj.can_create_guard():
            try:
                options["guards"] = {
                    g for g in options["guards"] if g.name != obj.initial_name
                }
                if obj.initial_name:
                    options["guards"].add(obj.create_guard(GuardBuilder.ID_MATCH))
                options["guards"].add(obj.create_guard(GuardBuilder.OBJECT_MUTATION))
                self.push(ConstantVariable(obj.get_const_attr(self, name), **options))
            except AttributeError:
                unimplemented("dynamic attr UnsupportedVariable")
        else:
            unimplemented(f"LOAD_ATTR {obj}")

    def BUILD_TUPLE(self, inst):
        items = self.popn(inst.argval)
        options = VariableTracker.propagate(items)
        self.push(TupleVariable(items, **options))

    def BUILD_SLICE(self, inst):
        items = self.popn(inst.argval)
        options = VariableTracker.propagate(items)
        self.push(SliceVariable(items, **options))

    def BUILD_LIST(self, inst):
        items = self.popn(inst.argval)
        options = VariableTracker.propagate(items)
        self.push(ListVariable(items, **options))

    def BUILD_MAP(self, inst):
        items = self.popn(inst.argval * 2)
        options = VariableTracker.propagate(items)
        result = dict()
        for k, v in zip(items[::2], items[1::2]):
            assert isinstance(k, ConstantVariable)
            result[k.value] = v
        assert len(result) == len(items) / 2
        self.push(ConstDictVariable(result, **options))

    def BUILD_CONST_KEY_MAP(self, inst):
        keys = self.pop()
        values = self.popn(inst.argval)
        options = VariableTracker.propagate([keys] + values)
        assert isinstance(keys, ConstantVariable)
        keys = keys.value
        assert istype(keys, tuple)
        assert len(keys) == len(values)
        self.push(ConstDictVariable(dict(zip(keys, values)), **options))

    def MAKE_FUNCTION(self, inst):
        fn_name_var = self.stack.pop()
        code_obj_var = self.stack.pop()
        fn_name = fn_name_var.as_proxy()
        code_obj = code_obj_var.as_proxy()
        assert isinstance(fn_name, str)
        assert isinstance(code_obj, types.CodeType)
        options = VariableTracker.propagate([fn_name_var, code_obj_var])
        self.push(
            UserFunctionVariable(
                types.FunctionType(code_obj, self.f_globals), **options
            )
        )

    def UNPACK_SEQUENCE(self, inst):
        seq = self.pop()
        options = VariableTracker.propagate([seq])
        if isinstance(seq, BaseListVariable):
            assert len(seq.items) == inst.argval
            self.guards.update(seq.guards)
            for i in reversed(seq.items):
                self.push(i)
        elif isinstance(seq, ConstantVariable):
            assert len(seq.value) == inst.argval
            for i in reversed(seq.value):
                self.push(ConstantVariable(i, **options))
        elif isinstance(seq, TensorVariable):
            proxy = seq.as_proxy()
            for i in reversed(range(inst.argval)):
                self.push(TensorVariable(proxy[i], **options))
        elif isinstance(seq, GetAttrVariable) and isinstance(seq.obj, TensorVariable):
            # x, y = a.shape
            proxy = getattr(seq.obj.as_proxy(), seq.name)
            for i in reversed(range(inst.argval)):
                self.push(TensorVariable(proxy[i], **options))
        else:
            unimplemented(f"UNPACK_SEQUENCE {seq}")

    def NOP(self, inst):
        pass

    def POP_TOP(self, inst):
        self.pop()

    def ROT_TWO(self, inst):
        a = self.pop()
        b = self.pop()
        self.push(a)
        self.push(b)

    def ROT_THREE(self, inst):
        a = self.pop()
        b = self.pop()
        c = self.pop()
        self.push(a)
        self.push(c)
        self.push(b)

    def ROT_FOUR(self, inst):
        a = self.pop()
        b = self.pop()
        c = self.pop()
        d = self.pop()
        self.push(a)
        self.push(d)
        self.push(c)
        self.push(b)

    def DUP_TOP(self, inst):
        a = self.pop()
        self.push(a)
        self.push(a)

    def DUP_TOP_TWO(self, inst):
        a = self.pop()
        b = self.pop()
        self.push(b)
        self.push(a)
        self.push(b)
        self.push(a)

    UNARY_POSITIVE = stack_op(operator.pos)
    UNARY_NEGATIVE = stack_op(operator.neg)
    UNARY_NOT = stack_op(operator.not_)
    UNARY_INVERT = stack_op(operator.invert)

    BINARY_POWER = stack_op(operator.pow)
    BINARY_MULTIPLY = stack_op(operator.mul)
    BINARY_MATRIX_MULTIPLY = stack_op(operator.matmul)
    BINARY_FLOOR_DIVIDE = stack_op(operator.floordiv)
    BINARY_TRUE_DIVIDE = stack_op(operator.truediv)
    BINARY_MODULO = stack_op(operator.mod)
    BINARY_ADD = stack_op(operator.add)
    BINARY_SUBTRACT = stack_op(operator.sub)
    BINARY_SUBSCR = stack_op(operator.getitem)
    BINARY_LSHIFT = stack_op(operator.lshift)
    BINARY_RSHIFT = stack_op(operator.rshift)
    BINARY_AND = stack_op(operator.and_)
    BINARY_OR = stack_op(operator.or_)
    BINARY_XOR = stack_op(operator.xor)

    INPLACE_POWER = stack_op(operator.ipow)
    INPLACE_MULTIPLY = stack_op(operator.imul)
    INPLACE_MATRIX_MULTIPLY = stack_op(operator.imatmul)
    INPLACE_FLOOR_DIVIDE = stack_op(operator.ifloordiv)
    INPLACE_TRUE_DIVIDE = stack_op(operator.itruediv)
    INPLACE_MODULO = stack_op(operator.imod)
    INPLACE_ADD = stack_op(operator.iadd)
    INPLACE_SUBTRACT = stack_op(operator.isub)
    INPLACE_LSHIFT = stack_op(operator.ilshift)
    INPLACE_RSHIFT = stack_op(operator.irshift)
    INPLACE_AND = stack_op(operator.iand)
    INPLACE_XOR = stack_op(operator.ixor)
    INPLACE_OR = stack_op(operator.ior)

    def compile_subgraph(self, rv):
        """
        Generate code from self.graph and return the Instruction()s to
        call that generated code.
        """
        if isinstance(rv, TensorVariable) or isinstance(rv, BaseListVariable):
            try:
                self.create_node(
                    "output", "output", (self.create_arg(rv.as_proxy()),), {}
                )
            except AttributeError:
                unimplemented("unsupported value in output")
        elif isinstance(rv, list):
            outputs = []
            for x in rv:
                if isinstance(x, TensorVariable):
                    outputs.append(self.create_arg(x.as_proxy()))
                elif isinstance(x, NNModuleVariable):
                    outputs.append(
                        self.create_arg(
                            self.create_proxy("get_attr", x.module_key, tuple(), {})
                        )
                    )
                else:
                    unimplemented(f"restore state for {type(x).__name__}")
            outputs = tuple(outputs)
            rv = VariableTracker(**VariableTracker.propagate(rv))
            self.create_node("output", "output", (outputs,), {})
        else:
            unimplemented(f"RETURN_VALUE {type(rv).__name__}")
        self.remove_unused_graphargs()
        ncalls = count_calls(self.graph)
        counters["stats"]["calls_captured"] += ncalls
        counters["stats"]["fusions_possible"] += ncalls - 1
        self.guards.update(rv.guards)
        gm = fx.GraphModule(FakeRootModule(self.nn_modules), self.graph)
        gm.recompile()
        name = unique_id("__compiled_fn")
        self.f_globals[name] = self.compiler_fn(gm, self.example_inputs())
        assert callable(self.f_globals[name]), "compiler_fn did not return callable"
        nargs = sum(map(len, self.graphargs))
        if config.debug:
            print(f"\n{name}")
            self.graph.print_tabular()
        return self.create_call_generated_code(name, nargs)

    def example_inputs(self):
        result = []
        for arg in self.graphargs:
            result.extend(arg.get_examples())
        return result

    def grow_stack_to(self, size):
        """Ensure co_stacksize is at least size"""
        self.code_options["co_stacksize"] = max(self.code_options["co_stacksize"], size)

    def create_call_generated_code(self, fn_name: str, nargs: int) -> List[Instruction]:
        """Call the generated code function stored in fn_name"""
        self.grow_stack_to(
            # +1 for function name, +1 for scratch space (restore vars)
            2
            + nargs
        )
        output = self.load_function_name(fn_name)

        for arg in self.graphargs:
            output.extend(arg.load(self))

        output.append(create_instruction("CALL_FUNCTION", nargs))
        return output

    def load_function_name(self, fn_name, num_on_stack=0):
        """Load the global fn_name on the stack num_on_stack down"""
        output = [self.create_load_global(fn_name, add=True)]
        if num_on_stack == 0:
            pass
        elif num_on_stack == 1:
            output.append(create_instruction("ROT_TWO"))
        elif num_on_stack == 2:
            output.append(create_instruction("ROT_THREE"))
        elif num_on_stack == 3:
            output.append(create_instruction("ROT_FOUR"))
        else:
            unimplemented("3+ stack args")
            # not tested, but should be something like:
            #   BUILD_TUPLE num_on_stack
            #   LOAD_GLOBAL reversed  (should assert this is not a local/global, etc)
            #   CALL_FUNCTION 1
            #   LOAD_GLOBAL fn_name
            #   ROT_TWO
            #   UNPACK_SEQUENCE num_on_stack
        return output

    def remove_unused_graphargs(self):
        expanded_graphargs = []
        for arg in self.graphargs:
            expanded_graphargs.extend([arg] * len(arg))
            arg.uses = 0

        for node, arg in zip(self.graph.nodes, expanded_graphargs):
            assert node.op == "placeholder"
            arg.uses += len(node.users)

        for node, arg in list(zip(self.graph.nodes, expanded_graphargs)):
            if arg.uses == 0:
                self.graph.erase_node(node)

        self.graphargs = [arg for arg in self.graphargs if arg.uses > 0]

    def compile_partial_subgraph(self):
        """
        Generate a subgraph to continue execution on user code.
        Automatically restore live variables.
        """
        stack_values = list(self.stack)
        self.grow_stack_to(len(stack_values) + 1)
        var_names = []
        constant_locals = []
        clobbered = set()
        self.prune_dead_locals()
        for k, v in self.symbolic_locals.items():
            if v.initial_name == k:
                continue  # no need to restore initial state
            elif self.is_const_var(v) and v.initial_name not in clobbered:
                constant_locals.append(self.load_const_var(v))
                constant_locals.append(self.create_store(k))
                clobbered.add(k)
            else:
                # must get the value from compiled graph
                var_names.append(k)

        clobbered.update(var_names)

        const_stack_prefix = []
        while stack_values and self.is_const_var(stack_values[0]):
            const_stack_prefix.append(self.load_const_var(stack_values.pop(0)))

        const_stack_suffix = []
        while stack_values and self.is_const_var(stack_values[-1]):
            if stack_values[-1].initial_name in clobbered:
                break
            const_stack_suffix.append(self.load_const_var(stack_values.pop()))
        const_stack_suffix = list(reversed(const_stack_suffix))

        self.grow_stack_to(len(const_stack_prefix) + len(var_names) + 1)

        if len(var_names) == 0 and len(stack_values) == 0:
            self.add_output_instructions(
                const_stack_prefix + constant_locals + const_stack_suffix
            )
        elif len(var_names) == 0 and len(stack_values) == 1:
            self.add_output_instructions(
                const_stack_prefix
                + self.compile_subgraph(stack_values[0])
                + constant_locals
                + const_stack_suffix
            )
        elif len(var_names) == 1 and len(stack_values) == 0:
            self.add_output_instructions(
                const_stack_prefix
                + self.compile_subgraph(self.symbolic_locals[var_names[0]])
                + constant_locals
                + self.restore_locals(var_names, 0, unpack=False)
                + const_stack_suffix
            )
        else:
            self.add_output_instructions(
                const_stack_prefix
                + self.compile_subgraph(
                    [self.symbolic_locals[k] for k in var_names]
                    + list(reversed(stack_values))
                )
                + constant_locals
                + self.restore_locals(var_names, len(stack_values))
                + const_stack_suffix
            )

    def is_const_var(self, value: VariableTracker):
        return (
            value.initial_name is not None
            or value.global_name is not None
            or isinstance(value, ConstantVariable)
        )

    def load_const_var(self, value: VariableTracker):
        if value.initial_name is not None:
            # guards to should not be needed for a copy?
            return self.create_load(value.initial_name)
        elif value.global_name is not None:
            return self.create_load_global(value.global_name)
        elif isinstance(value, ConstantVariable):
            # no need to get a constant from the compiled graph
            self.guards.update(value.guards)
            return self.create_load_const(value.value)
        else:
            assert False

    def add_output_instructions(self, prefix: List[Instruction]):
        """
        We call this on the creation of a new compiled subgraph that is inserted
        before user code.

        Currently, this stops the analysis (we only support a prefix of user code).

        Later we should extend this to continue the analysis
        """
        self.output_instructions.extend(prefix)
        self.fully_converted = False

        # TODO(jansel): resume the analysis instead of exiting
        self.instruction_pointer = None  # exit

    def restore_locals(self, var_names, extra, unpack=True):
        """
        Used by compile_partial_subgraph() to set local variables from
        the result of a self.compile_subgraph() call.
        """
        code = []
        if unpack:
            code.append(create_instruction("UNPACK_SEQUENCE", len(var_names) + extra))
        for name in var_names:
            code.append(self.create_store(name))
        return code

    def copy_graphstate(self):
        """Create a checkpoint of the current state by copying everything"""
        graph_nodes = set(self.graph.nodes)
        guards = copy.deepcopy(self.guards)
        graphargs = list(self.graphargs)
        symbolic_locals = collections.OrderedDict(
            VariableTracker.copy(list(self.symbolic_locals.items()))
        )
        stack = VariableTracker.copy(self.stack)
        nn_modules = dict(self.nn_modules)
        return (
            graph_nodes,
            graphargs,
            guards,
            symbolic_locals,
            stack,
            nn_modules,
            self.instruction_pointer,
            self.current_instruction,
            self.next_instruction,
        )

    def restore_graphstate(self, state):
        """Restore a checkpoint created by self.copy_graphstate()"""
        (
            graph_nodes,
            self.graphargs,
            self.guards,
            self.symbolic_locals,
            self.stack,
            self.nn_modules,
            self.instruction_pointer,
            self.current_instruction,
            self.next_instruction,
        ) = state
        # FX deepcopy doesn't work for a partially created graph, so just remove new nodes
        for node in reversed(list(self.graph.nodes)):
            if node not in graph_nodes:
                self.graph.erase_node(node)

    def __init__(
        self,
        cnt: typing.Iterable,
        graph: fx.Graph,
        graphargs: List,
        nn_modules: Dict,
        guards: Set[Guard],
        instructions: List[Instruction],
        f_globals: Dict[str, Any],
        f_builtins: Dict[str, Any],
        code_options: Dict[str, Any],
        compiler_fn=None,
        symbolic_locals=None,
        f_code=None,
    ):
        super(InstructionTranslatorBase, self).__init__()
        # Mutable state checkpointed by copy_graphstate()
        self.graph = graph
        self.graphargs = graphargs
        self.stack = []
        self.symbolic_locals = symbolic_locals
        self.guards = guards
        self.nn_modules = nn_modules
        self.instruction_pointer = 0
        self.next_instruction = None
        self.current_instruction = create_instruction("NOP")

        # Properties of the input/output code
        self.instructions = instructions
        self.indexof = {id(i): n for n, i in enumerate(instructions)}
        self.f_globals = f_globals
        self.f_builtins = f_builtins
        self.code_options = code_options
        self.output_instructions = []
        self.fully_converted = None
        self.compiler_fn = compiler_fn
        self.f_code = f_code

        # Dynamic state not checkpointed
        self.checkpoint = None
        self.cnt = cnt


class InstructionTranslator(InstructionTranslatorBase):
    def __init__(
        self,
        instructions: List[Instruction],
        f_code,
        f_locals,
        f_globals,
        f_builtins,
        code_options,
        compiler_fn,
    ):
        super(InstructionTranslator, self).__init__(
            cnt=itertools.count(),
            graph=fx.Graph(),
            graphargs=[],
            nn_modules={},
            guards=set(),
            instructions=instructions,
            f_globals=f_globals,
            f_builtins=f_builtins,
            code_options=code_options,
            compiler_fn=compiler_fn,
            f_code=f_code,
        )
        self.symbolic_locals = collections.OrderedDict(
            (k, self.wrap_local(k, f_locals[k]).with_initial_name(k))
            for k in code_options["co_varnames"]
            if k in f_locals
        )

    def should_compile_partial_graph(self):
        return True
        # if count_calls(self.graph) >= config.minimum_call_count:
        #     self.should_compile_partial_graph = lambda: True
        #     return True
        # return False

    def create_call_resume_at(self, inst):
        reads = livevars_analysis(self.instructions, inst)
        argnames = tuple(k for k in self.symbolic_locals.keys() if k in reads)
        nargs = len(self.stack) + len(argnames)

        if self.code_options["co_cellvars"]:
            raise unimplemented("resume_at with cellvars")
        if self.code_options["co_freevars"]:
            raise unimplemented("resume_at with freevars")

        name = unique_id(f"__resume_at_{self.next_instruction.offset}")
        self.grow_stack_to(1 + nargs)
        self.f_globals[name] = ContinueExecutionCache.lookup(
            self.f_code,
            self.f_globals,
            inst.offset,
            len(self.stack),
            argnames,
        )
        return (
            self.load_function_name(name, len(self.stack))
            + [self.create_load(k) for k in argnames]
            + [
                create_instruction("CALL_FUNCTION", nargs),
                create_instruction("RETURN_VALUE"),
            ]
        )

    def RETURN_VALUE(self, inst):
        rv = self.pop()
        self.instruction_pointer = None
        if count_calls(self.graph) == 0:
            unimplemented("no graph found")
        elif rv.state == TracingSupported.YES:
            self.output_instructions.extend(
                self.compile_subgraph(rv) + [create_instruction("RETURN_VALUE")]
            )
            if self.fully_converted is None:
                self.fully_converted = True
        else:
            unimplemented("not traceable")


class InliningInstructionTranslator(InstructionTranslatorBase):
    """Trace and inline a called method"""

    @classmethod
    def inline_call(cls, parent, func, args, kwargs):
        with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
            return cls.inline_call_(parent, func, args, kwargs)

    @staticmethod
    def inline_call_(parent, func, args, kwargs):
        assert callable(func)
        if getattr(func, "__closure__", None) is not None:
            unimplemented("inline with  __closure__")
        if getattr(func, "__self__", None) is not None:
            unimplemented("inline with  __self__")
        try:
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
        except Exception as e:
            raise unimplemented(f"signature issues {e}")
        sub_locals = dict()
        sub_globals = func.__globals__
        for k, v in bound.arguments.items():
            if isinstance(v, VariableTracker):
                sub_locals[k] = v
            elif isinstance(v, (bool, int, float, type(None))):
                sub_locals[k] = ConstantVariable(v)
            else:
                unimplemented(f"inline_call unsupported default: {typestr(v)}")
        tracer = InliningInstructionTranslator(
            parent, func.__code__, sub_locals, sub_globals
        )
        tracer.run()
        assert tracer.symbolic_result is not None
        return tracer.symbolic_result

    def __init__(
        self,
        parent: InstructionTranslatorBase,
        code: types.CodeType,
        symbolic_locals,
        f_globals,
    ):
        super(InliningInstructionTranslator, self).__init__(
            cnt=parent.cnt,
            graph=parent.graph,
            graphargs=parent.graphargs,
            nn_modules=parent.nn_modules,
            guards=parent.guards,
            f_globals=f_globals,
            f_builtins=f_globals["__builtins__"],
            symbolic_locals=symbolic_locals,
            instructions=cleaned_instructions(code),
            code_options={k: getattr(code, k) for k in dir(code)},
            compiler_fn=parent.compiler_fn,
        )
        self.symbolic_result = None

    def should_compile_partial_graph(self):
        return False  # inlining functions is all-or-nothing

    def create_call_resume_at(self, offset):
        unimplemented("cant resume while inlining")

    def RETURN_VALUE(self, inst):
        self.symbolic_result = self.pop()
        self.instruction_pointer = None


class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""

    def __init__(self, nn_modules: dict):
        super(FakeRootModule, self).__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)
