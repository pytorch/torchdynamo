import collections
import copy
import dataclasses
import functools
import importlib
import inspect
import itertools
import math
import operator
import os
import re
import sys
import types
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from unittest.mock import patch

import torch
from torch import fx

from . import config, skipfiles
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
from .variable_tracker import AllowedFunctionOrModuleVariable
from .variable_tracker import BaseListVariable
from .variable_tracker import BaseUserFunctionVariable
from .variable_tracker import BasicTypeVariable
from .variable_tracker import BuiltinVariable
from .variable_tracker import ClosureVariable
from .variable_tracker import ConstantVariable
from .variable_tracker import ConstDictVariable
from .variable_tracker import FunctionConstantWrapper
from .variable_tracker import GetAttrVariable
from .variable_tracker import ListIteratorVariable
from .variable_tracker import ListVariable
from .variable_tracker import NestedUserFunctionVariable
from .variable_tracker import NNModuleVariable
from .variable_tracker import PythonModuleVariable
from .variable_tracker import SliceVariable
from .variable_tracker import TensorVariable
from .variable_tracker import TracingSupported
from .variable_tracker import TupleVariable
from .variable_tracker import typestr
from .variable_tracker import UnknownVariable
from .variable_tracker import UnsupportedVariable
from .variable_tracker import UserFunctionVariable
from .variable_tracker import UserMethodVariable
from .variable_tracker import VariableTracker

counters = collections.defaultdict(collections.Counter)


def proxy_args_kwargs(args, kwargs):
    try:
        proxy_args = tuple(arg.as_proxy() for arg in args)
        proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return proxy_args, proxy_kwargs
    except NotImplementedError:
        raise unimplemented(
            f"call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}"
        )


class Unsupported(RuntimeError):
    pass


def unimplemented(msg: str):
    counters["unimplemented"][msg] += 1
    assert msg != os.environ.get("BREAK", False)
    raise Unsupported(msg)


def warning(msg: str):
    counters["warnings"][msg] += 1


def stack_op(fn: typing.Callable):
    nargs = len(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def impl(self: "InstructionTranslatorBase", inst: Instruction):
        inputs: List[VariableTracker] = self.popn(nargs)
        options = VariableTracker.propagate(inputs)

        if any(isinstance(i, TensorVariable) for i in inputs):
            val = TensorVariable(
                self.create_proxy(
                    "call_function", fn, tuple(i.as_proxy() for i in inputs), {}
                ),
                **options,
            )
        elif all(i.is_python_constant() for i in inputs):
            # constant fold
            val = ConstantVariable(
                fn(*[i.as_python_constant() for i in inputs]), **options
            )
        elif (
            isinstance(inputs[0], BaseListVariable)
            and fn is operator.getitem
            and inputs[1].is_python_constant()
        ):
            base, item = inputs
            val = base.getitem_const(item)
        elif (
            isinstance(inputs[0], NNModuleVariable)
            and fn is operator.getitem
            and inputs[1].is_python_constant()
        ):
            assert len(inputs) == 2
            key = inputs[0].module_key
            mod = self.get_submodule(key)
            assert type(mod).__getitem__ is torch.nn.ModuleList.__getitem__, typestr(
                mod
            )
            submod = mod[inputs[1].as_python_constant()]
            val = NNModuleVariable(
                self.add_submodule(submod, key, inputs[1].as_python_constant()),
                **options,
            )
        else:
            unimplemented(f"stack_op {typestr(*inputs)}")

        self.push(val)

    return impl


def generic_jump(truth_fn: typing.Callable, push: bool):
    def inner(self: "InstructionTranslatorBase", inst: Instruction):
        value: VariableTracker = self.pop()
        self.guards.update(value.guards)
        if value.is_python_constant():
            if truth_fn(value.as_python_constant()):
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
    def cell_and_freevars(self):
        if not hasattr(self, "_cell_and_freevars"):
            self._cell_and_freevars = tuple(
                self.code_options["co_cellvars"] or []
            ) + tuple(self.code_options["co_freevars"] or [])
        return self._cell_and_freevars

    def create_load(self, name):
        if name in self.cell_and_freevars():
            return create_instruction(
                "LOAD_DEREF", self.cell_and_freevars().index(name), name
            )
        assert name in self.code_options["co_varnames"]
        return create_instruction(
            "LOAD_FAST", self.code_options["co_varnames"].index(name), name
        )

    def create_store(self, name):
        if name in self.cell_and_freevars():
            return create_instruction(
                "STORE_DEREF", self.cell_and_freevars().index(name), name
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
        if value in self.code_options["co_consts"]:
            return [self._create_load_const(value)]

        def is_safe(v):
            if istype(v, (tuple, frozenset)):
                return all(map(is_safe, v))
            if istype(v, (list, dict, set)):
                # These could mutate
                return False
            assert istype(
                v, (types.CodeType, int, float, bool, str, bytes, type(None))
            ), f"unsupported constant type {typestr(v)}"
            return True

        output = []

        def visit(v):
            if is_safe(v):
                output.append(self._create_load_const(v))
            elif isinstance(v, (list, tuple, set)):
                self.grow_stack_to(len(v) + len(self.stack))
                for item in v:
                    visit(item)
                output.append(
                    create_instruction(f"BUILD_{type(v).__name__.upper()}", len(v))
                )
            elif isinstance(v, dict):
                self.grow_stack_to(len(v) + len(self.stack) + 1)
                keys = tuple(sorted(v.keys()))
                for k in keys:
                    visit(v[k])
                output.append(self._create_load_const(keys))
                output.append(create_instruction("BUILD_CONST_KEY_MAP", len(keys)))

        visit(value)
        return output

    def _create_load_const(self, value):
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
            return NNModuleVariable(
                module_key=self.add_submodule(value, name),
                state=TracingSupported.YES,
                guards={
                    Guard(name, GuardSource.LOCAL, GuardBuilder.ID_MATCH),
                    Guard(name, GuardSource.LOCAL, GuardBuilder.OBJECT_MUTATION),
                },
            )
        elif value is None or istype(value, bool):
            # For these, just specialize on exact value
            return ConstantVariable(
                value=value,
                guards={Guard(name, GuardSource.LOCAL, GuardBuilder.ID_MATCH)},
            )
        elif (
            istype(value, int)
            or (istype(value, float) and value in (-1.0, 0.0, 0.25, 0.5, 1.0, 2.0))
            or (
                istype(value, (tuple, list, torch.Size))
                and all(istype(x, int) for x in value)
            )
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

    def add_submodule(self, mod: torch.nn.Module, *names):
        assert isinstance(mod, (torch.nn.Module, torch.Tensor))

        for k, v in self.nn_modules.items():
            if v is mod:
                return k

        name = re.sub(r"[^a-zA-Z0-9]", "_", "_".join(map(str, names)))
        if not name or not name[0].isalpha():
            name = "sub" + name

        base = name
        for i in itertools.count():
            if name not in self.nn_modules:
                self.nn_modules[name] = mod
                return name
            name = f"{base}_{i}"

        assert False

    def prune_dead_locals(self):
        reads = livevars_analysis(self.instructions, self.current_instruction)
        # implicit use by super()
        reads = reads | {"__class__"}
        # output variables?
        reads = reads | set(self.code_options["co_freevars"] or [])
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

    def call_function(
        self,
        fn: VariableTracker,
        args: List[VariableTracker],
        kwargs: Dict[str, VariableTracker],
    ):
        assert isinstance(fn, VariableTracker)
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        constant_args = all(
            x.is_python_constant() for x in itertools.chain(args, kwargs.values())
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
            and constant_args
        ):
            # constant fold
            self.push(
                ConstantVariable(
                    fn.value(
                        *[x.as_python_constant() for x in args],
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
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
                    self.call_function(
                        NNModuleVariable(
                            self.add_submodule(submod, fn.module_key, idx),
                            **options,
                        ),
                        [arg],
                        {},
                    )
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
                self.inline_user_function(
                    UserFunctionVariable(fn=forward, **VariableTracker.propagate([fn])),
                    [fn] + args,
                    kwargs,
                )
        elif isinstance(fn, BaseUserFunctionVariable):
            self.guards.update(fn.guards)
            try:
                self.inline_user_function(fn, fn.self_args() + args, kwargs)
            except Unsupported:
                if not self.should_compile_partial_graph():
                    raise
                self.partial_subgraph_and_call(fn, fn.self_args() + args, kwargs)
        elif isinstance(fn, BuiltinVariable):
            if constant_args and fn.fn in (
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
            ):
                # constant folding
                self.push(
                    ConstantVariable(
                        fn.fn(
                            *[x.as_python_constant() for x in args],
                            **{k: v.as_python_constant() for k, v in kwargs.items()},
                        ),
                        **options,
                    )
                )
            elif fn.fn is range and constant_args:
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
            elif fn.fn is iter and args and isinstance(args[0], NNModuleVariable):
                assert not kwargs and len(args) == 1
                self.push(
                    ListIteratorVariable(args[0].expand_module_list(self), **options)
                )
            elif fn.fn is len:
                assert not kwargs and len(args) == 1
                arg = args[0]
                if isinstance(arg, TensorVariable):
                    if arg.size:
                        assert not config.dynamic_shapes
                        self.push(ConstantVariable(arg.size[0], **options))
                    else:
                        self.push(
                            BasicTypeVariable(
                                self.create_proxy(
                                    "call_function", len, (arg.as_proxy(),), {}
                                ),
                                **options,
                            )
                        )
                elif isinstance(arg, (BaseListVariable, ConstDictVariable)):
                    self.push(ConstantVariable(len(arg.items), **options))
                elif isinstance(arg, NNModuleVariable):
                    # assuming constant length of nn.ModuleList, etc
                    self.push(
                        ConstantVariable(
                            len(self.get_submodule(arg.module_key)), **options
                        )
                    )
                else:
                    unimplemented(f"`len` with arg type {arg}")
            elif fn.fn is isinstance:
                assert not kwargs and len(args) == 2
                arg, isinstance_type = args
                arg_type = arg.python_type()
                isinstance_type = isinstance_type.as_python_constant()
                try:
                    val = issubclass(arg_type, isinstance_type)
                except TypeError:
                    val = arg_type is isinstance_type
                self.push(ConstantVariable(val, **options))
            elif fn.fn is super:
                unimplemented("super")
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
        keys.sort(key=lambda k: (self.is_constant_or_input(kwargs[k]), k))
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
            self.output_instructions.extend(self.create_load_const(tuple(keys)))
            self.output_instructions.append(
                create_instruction("CALL_FUNCTION_KW", len(args_and_kwargs))
            )
        self.push(UnknownVariable())
        self.output_instructions.extend(
            self.create_call_resume_at(self.next_instruction)
        )

    def inline_user_function(self, fn, args, kwargs):
        """
        A call to some user defined function by inlining it.
        """
        self.guards.update(fn.guards)
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
        if inst.starts_line:
            self.lineno = inst.starts_line

        if len(self.stack) == 0 and self.should_compile_partial_graph():
            self.checkpoint = inst, self.copy_graphstate()

        if config.trace:
            print("TRACE", inst.opname, inst.argval, self.stack)

        try:
            if not hasattr(self, inst.opname):
                unimplemented(f"missing: {inst.opname}")
            getattr(self, inst.opname)(inst)
            return (
                inst.opname != "RETURN_VALUE" and self.instruction_pointer is not None
            )
        except Unsupported:
            if self.checkpoint:
                assert not self.output_instructions
                continue_inst, state = self.checkpoint
                self.restore_graphstate(state)
                if count_calls(self.graph) < config.minimum_call_count:
                    raise
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
        except Unsupported:
            raise
        except Exception as e:
            sys.stderr.write(
                f"ERROR FROM offset={self.current_instruction.offset} "
                f"filename {self.code_options.get('co_filename')} "
                f"{self.lineno} {typestr(e)}\n"
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

    def pop(self) -> TensorVariable:
        return self.stack.pop()

    def popn(self, n) -> List[TensorVariable]:
        return list(reversed([self.pop() for _ in range(n)]))

    def LOAD_FAST(self, inst):
        assert inst.argval not in self.cell_and_freevars()
        if inst.argval not in self.symbolic_locals:
            unimplemented("undefined LOAD_FAST")
        self.push(self.symbolic_locals[inst.argval])
        if inst.argval.startswith("___stack"):
            self.symbolic_locals.pop(inst.argval)

    def LOAD_DEREF(self, inst):
        assert inst.argval in self.cell_and_freevars()
        if inst.argval not in self.symbolic_locals:
            unimplemented(f"undefined LOAD_DEREF {inst.argval}")
        self.push(self.symbolic_locals[inst.argval])

    def STORE_FAST(self, inst):
        self.symbolic_locals[inst.argval] = self.pop()

    STORE_DEREF = STORE_FAST

    def LOAD_CLOSURE(self, inst):
        self.push(ClosureVariable(name=inst.argval))

    def LOAD_CONST(self, inst):
        self.push(ConstantVariable(value=inst.argval))

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
        elif isinstance(value, bool) or value is None:
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
            self.push(
                NNModuleVariable(
                    self.add_submodule(value, inst.argval),
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

    def IMPORT_NAME(self, inst):
        value = importlib.import_module(inst.argval)
        if is_allowed(value):
            self.push(
                AllowedFunctionOrModuleVariable(value=value, state=TracingSupported.YES)
            )
        elif istype(value, types.ModuleType):
            self.push(
                PythonModuleVariable(
                    value,
                )
            )
        else:
            unimplemented(f"IMPORT_NAME {typestr(value)}")

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
            left.is_python_constant()
            and right.is_python_constant()
            and op in supported_any
        ):
            # constant fold
            self.push(
                ConstantVariable(
                    supported_any[op](
                        left.as_python_constant(), right.as_python_constant()
                    ),
                    **options,
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
            elif ConstantVariable.is_literal(subobj):
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
                        output.append(
                            NNModuleVariable(
                                module_key=self.add_submodule(
                                    item, obj.module_key, name, i
                                ),
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
            elif callable(member) and not isinstance(
                member, (types.BuiltinFunctionType, types.BuiltinMethodType)
            ):
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

    IMPORT_FROM = LOAD_ATTR

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
        flags = inst.arg
        old_stack = list(self.stack)
        fn_name = self.pop()
        code = self.pop()
        defaults = None
        closure = None
        annotations = None
        kwdefaults = None

        if flags & 0x08:
            closure = self.pop()
        if flags & 0x04:
            annotations = self.pop()
        if flags & 0x02:
            kwdefaults = self.pop()
        if flags & 0x01:
            defaults = self.pop()

        options = VariableTracker.propagate(old_stack[len(self.stack) :])
        self.push(
            NestedUserFunctionVariable(
                fn_name,
                code,
                self.f_globals,
                defaults,
                kwdefaults,
                annotations,
                closure,
                **options,
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
        elif seq.is_python_constant() and isinstance(seq, ConstantVariable):
            val = seq.as_python_constant()
            assert len(val) == inst.argval
            for i in reversed(val):
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

    def output_proxy(self, rv):
        if isinstance(rv, VariableTracker):
            self.guards.update(rv.guards)

        if isinstance(rv, VariableTracker) and rv.is_proxy():
            return rv.as_proxy()
        elif isinstance(rv, NNModuleVariable):
            return self.create_proxy("get_attr", rv.module_key, tuple(), {})
        elif isinstance(rv, (list, tuple)):
            return tuple(map(self.output_proxy, rv))

        raise unimplemented(f"RETURN_VALUE {type(rv).__name__}")

    def compile_subgraph(self, rv):
        """
        Generate code from self.graph and return the Instruction()s to
        call that generated code.
        """
        self.create_node(
            "output", "output", (self.create_arg(self.output_proxy(rv)),), {}
        )
        self.remove_unused_graphargs()
        ncalls = count_calls(self.graph)
        counters["stats"]["calls_captured"] += ncalls
        counters["stats"]["fusions_possible"] += ncalls - 1
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
        return [self.create_load_global(fn_name, add=True)] + self.rot_n(
            num_on_stack + 1
        )

    def make_function_with_closure(
        self, fn_name: str, code: types.CodeType, num_on_stack=0
    ):
        freevars = code.co_freevars
        assert freevars
        self.grow_stack_to(num_on_stack + 3)
        self.grow_stack_to(num_on_stack + len(freevars))
        output = []
        for var in freevars:
            assert var in self.cell_and_freevars()
            output.append(
                create_instruction(
                    "LOAD_CLOSURE", self.cell_and_freevars().index(var), var
                )
            )
        output.append(create_instruction("BUILD_TUPLE", len(freevars)))
        output.extend(self.create_load_const(code))
        output.extend(self.create_load_const(fn_name))
        output.append(create_instruction("MAKE_FUNCTION", 0x08))
        output.extend(self.rot_n(num_on_stack + 1))
        return output

    def rot_n(self, n):
        if n == 0 or n == 1:
            return []
        elif n == 2:
            return [create_instruction("ROT_TWO")]
        elif n == 3:
            return [create_instruction("ROT_THREE")]
        elif n == 4:
            return [create_instruction("ROT_FOUR")]
        else:
            raise unimplemented("4+ stack args")
            # not tested, but should be something like:
            #   BUILD_TUPLE num_on_stack
            #   LOAD_GLOBAL reversed  (should assert this is not a local/global, etc)
            #   CALL_FUNCTION 1
            #   LOAD_GLOBAL fn_name
            #   ROT_TWO
            #   UNPACK_SEQUENCE num_on_stack

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
            elif self.is_constant_or_input(v) and v.initial_name not in clobbered:
                constant_locals.extend(self.load_const_var(v))
                constant_locals.append(self.create_store(k))
                clobbered.add(k)
            else:
                # must get the value from compiled graph
                var_names.append(k)

        clobbered.update(var_names)

        const_stack_prefix = []
        while stack_values and self.is_constant_or_input(stack_values[0]):
            const_stack_prefix.extend(self.load_const_var(stack_values.pop(0)))

        const_stack_suffix = []
        while stack_values and self.is_constant_or_input(stack_values[-1]):
            if stack_values[-1].initial_name in clobbered:
                break
            const_stack_suffix.extend(self.load_const_var(stack_values.pop()))
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

    def is_constant_or_input(self, value: VariableTracker):
        result = value.initial_name is not None or value.global_name is not None
        if not result and value.is_python_constant():
            try:
                self.create_load_const(value.as_python_constant())
                return True
            except AssertionError:
                return False
        return result

    def load_const_var(self, value: VariableTracker):
        if value.initial_name is not None:
            # guards to should not be needed for a copy?
            return [self.create_load(value.initial_name)]
        elif value.global_name is not None:
            return [self.create_load_global(value.global_name)]
        elif value.is_python_constant():
            # no need to get a constant from the compiled graph
            self.guards.update(value.guards)
            return self.create_load_const(value.as_python_constant())
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
        self.lineno = code_options.get("co_firstlineno")


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
        vars = list(code_options["co_varnames"])
        vars.extend(x for x in self.cell_and_freevars() if x not in vars)
        self.symbolic_locals = collections.OrderedDict(
            (k, self.wrap_local(k, f_locals[k]).with_initial_name(k))
            for k in vars
            if k in f_locals
        )

    def should_compile_partial_graph(self):
        return True
        # if count_calls(self.graph) >= config.minimum_call_count:
        #     self.should_compile_partial_graph = lambda: True
        #     return True
        # return False

    def create_call_resume_at(self, inst):
        self.instruction_pointer = None
        self.fully_converted = False

        reads = livevars_analysis(self.instructions, inst)
        argnames = tuple(
            k
            for k in self.symbolic_locals.keys()
            if k in reads and k not in self.cell_and_freevars()
        )
        nargs = len(self.stack) + len(argnames)

        name = unique_id(f"__resume_at_{self.next_instruction.offset}")
        self.grow_stack_to(1 + nargs)

        new_code: types.CodeType = ContinueExecutionCache.lookup(
            self.f_code, inst.offset, len(self.stack), argnames
        )

        if new_code.co_freevars:
            load_fn = self.make_function_with_closure(name, new_code, len(self.stack))
        else:
            self.f_globals[name] = types.FunctionType(new_code, self.f_globals, name)
            load_fn = self.load_function_name(name, len(self.stack))

        return (
            load_fn
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
        assert isinstance(func, (UserFunctionVariable, NestedUserFunctionVariable))
        if func.has_closure() and isinstance(func, UserFunctionVariable):
            unimplemented("inline with  __closure__")
        if func.has_self():
            unimplemented("inline with  __self__")
        if skipfiles.check(func.get_filename()):
            unimplemented("inline in skipfiles")

        sub_locals = func.bind_args(parent, args, kwargs)
        for v in sub_locals.values():
            if not isinstance(v, VariableTracker):
                unimplemented(f"unconverted arg {v}")

        tracer = InliningInstructionTranslator(
            parent, func.get_code(), sub_locals, func.get_globals()
        )
        tracer.run()
        assert tracer.symbolic_result is not None
        assert tracer.fully_converted

        func.export_freevars(parent, tracer)
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
        self.fully_converted = True


class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""

    def __init__(self, nn_modules: dict):
        super(FakeRootModule, self).__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)
