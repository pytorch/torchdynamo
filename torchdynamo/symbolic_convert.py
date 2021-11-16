import collections
import copy
import dis
import functools
import importlib
import inspect
import itertools
import operator
import re
import sys
import types
import typing
from functools import lru_cache
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from unittest.mock import patch

import torch
from torch import fx

from . import config
from . import skipfiles
from .allowed_functions import is_allowed, is_builtin
from .bytecode_analysis import livevars_analysis
from .bytecode_transformation import cleaned_instructions
from .bytecode_transformation import create_instruction
from .bytecode_transformation import Instruction
from .bytecode_transformation import is_generator
from .bytecode_transformation import unique_id
from .guards import Guard
from .guards import GuardBuilder
from .resume_execution import ContinueExecutionCache
from .utils import count_calls
from .utils import counters
from .utils import istype
from .utils import unimplemented
from .utils import Unsupported
from .utils import warning
from .variable_builder import VariableBuilder
from .variable_source import AttrSource
from .variable_source import GetItemSource
from .variable_source import GlobalSource
from .variable_source import LocalSource
from .variable_source import NNModuleSource
from .variable_source import Source
from .variable_tracker import AllowedFunctionOrModuleVariable
from .variable_tracker import BaseListVariable
from .variable_tracker import BuiltinVariable
from .variable_tracker import ClosureVariable
from .variable_tracker import ConstantVariable
from .variable_tracker import ConstDictVariable
from .variable_tracker import GetAttrVariable
from .variable_tracker import ListIteratorVariable
from .variable_tracker import ListVariable
from .variable_tracker import MutableLocal
from .variable_tracker import NestedUserFunctionVariable
from .variable_tracker import NNModuleVariable
from .variable_tracker import PythonModuleVariable
from .variable_tracker import SliceVariable
from .variable_tracker import TensorVariable
from .variable_tracker import TupleVariable
from .variable_tracker import typestr
from .variable_tracker import UnknownVariable
from .variable_tracker import UnsupportedVariable
from .variable_tracker import UserFunctionVariable
from .variable_tracker import UserMethodVariable
from .variable_tracker import VariableTracker


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
            isinstance(inputs[0], (BaseListVariable, ConstDictVariable))
            and fn is operator.getitem
            and inputs[1].is_python_constant()
        ):
            base, item = inputs
            val = base.getitem_const(item)

        elif (
            isinstance(inputs[0], BaseListVariable)
            and isinstance(inputs[1], BaseListVariable)
            and fn is operator.add
        ):
            a, b = inputs
            assert type(a) is type(b)
            val = type(a)(a.items + b.items, **options)
        elif (
            isinstance(inputs[0], NNModuleVariable)
            and fn is operator.getitem
            and inputs[1].is_python_constant()
        ):
            assert len(inputs) == 2
            key = inputs[0].module_key
            mod = self.get_submodule(key)
            assert type(mod).__getitem__ in (
                torch.nn.ModuleList.__getitem__,
                torch.nn.ParameterList.__getitem__,
            ), typestr(mod)
            assert inputs[0].source
            key = inputs[1].as_python_constant()
            submod = mod[key]
            val = self.add_submodule(
                submod,
                key,
                inputs[1].as_python_constant(),
                source=NNModuleSource(GetItemSource(inputs[0].source, key)),
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


def break_graph_if_unsupported(inner_fn):
    @functools.wraps(inner_fn)
    def wrapper(self: "InstructionTranslatorBase", inst: Instruction):
        state = self.copy_graphstate()
        try:
            inner_fn(self, inst)
        except Unsupported:
            if not self.should_compile_partial_graph():
                raise
            self.restore_graphstate(state)
            self.compile_partial_subgraph()
            # note, assuming inst pushes 1
            vars = self.popn(1 - dis.stack_effect(inst.opcode, inst.arg))
            warning(f"breaking graph: {vars[0]}")
            self.add_output_instructions([inst])
            self.push(UnknownVariable())
            self.add_output_instructions(
                self.create_call_resume_at(self.next_instruction)
            )

    return wrapper


def is_safe_constant(v):
    if istype(v, (tuple, frozenset)):
        return all(map(is_safe_constant, v))
    return istype(v, (types.CodeType, int, float, bool, str, bytes, type(None)))


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

    def create_load_closure(self, name):
        assert name in self.cell_and_freevars()
        return create_instruction(
            "LOAD_CLOSURE", self.cell_and_freevars().index(name), name
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

    def new_var(self, name="tmp"):
        existing = set(self.code_options["co_varnames"])
        for i in itertools.count():
            var = f"___{name}_{i}"
            if var not in existing:
                self.code_options["co_varnames"] = self.code_options["co_varnames"] + (
                    var,
                )
                return var

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
        assert is_safe_constant(value), f"unsafe constant {value}"
        return self._create_load_const(value)

    def _create_load_const(self, value):
        co_consts = self.code_options["co_consts"]
        assert istype(co_consts, tuple)
        index = None
        for i, v in enumerate(co_consts):
            if type(v) is type(value) and v == value:
                index = i
                break
        if index is None:
            index = len(co_consts)
            co_consts = co_consts + (value,)
            self.code_options["co_consts"] = co_consts
        return create_instruction("LOAD_CONST", index, value)

    create_load_output = _create_load_const

    def create_load_attr(self, name):
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] = self.code_options["co_names"] + (name,)
        return create_instruction("LOAD_ATTR", self.code_options["co_names"], name)

    def add_submodule(self, mod: torch.nn.Module, *names, **options):
        options = dict(options)
        options["guards"] = set(options.get("guards", []))
        source: Source = options["source"]
        if isinstance(mod, torch.Tensor):
            options.update(TensorVariable.specialize(mod))
            options["guards"].add(source.create_guard(GuardBuilder.TENSOR_MATCH))

            def wrap_name(module_key):
                return TensorVariable(
                    self.create_proxy("get_attr", module_key, tuple(), {}), **options
                )

        else:
            assert isinstance(mod, torch.nn.Module)
            options["guards"].add(source.create_guard(GuardBuilder.NN_MODULE))

            def wrap_name(module_key):
                return NNModuleVariable(type(mod), module_key, **options)

        for k, v in self.nn_modules.items():
            if v is mod:
                # it already exists
                return wrap_name(k)

        # create a new unique name
        name = re.sub(r"[^a-zA-Z0-9]", "_", "_".join(map(str, names)))
        if not name or not name[0].isalpha():
            name = "sub" + name
        base = name
        for i in itertools.count():
            if name not in self.nn_modules:
                self.nn_modules[name] = mod
                return wrap_name(name)
            name = f"{base}_{i}"

        assert False

    def prune_dead_locals(self):
        reads = livevars_analysis(self.instructions, self.current_instruction)
        # implicit use by super()
        # reads = reads | {"__class__"}
        # output variables?
        reads = reads | set(self.cell_and_freevars())
        self.symbolic_locals = collections.OrderedDict(
            [(k, v) for k, v in self.symbolic_locals.items() if k in reads]
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
        assert all(
            isinstance(x, VariableTracker)
            for x in itertools.chain(args, kwargs.values())
        )
        self.push(fn.call_function(self, args, kwargs))

    def replace_all(self, oldvar, newvar):
        assert oldvar.mutable_local
        assert newvar.mutable_local

        def repl(v: VariableTracker):
            if v.mutable_local is oldvar.mutable_local:
                return newvar
            return v

        self.stack = [VariableTracker.apply(repl, x) for x in self.stack]
        for k, x in self.symbolic_locals.items():
            self.symbolic_locals[k] = VariableTracker.apply(repl, x)

    def inline_user_function_return(self, fn, args, kwargs):
        """
        A call to some user defined function by inlining it.
        """
        state = self.copy_graphstate()
        try:
            result = InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
            self.guards.update(fn.guards)
            return result
        except Exception:
            self.restore_graphstate(state)
            raise

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
                self.output_instructions.extend(self.instructions)
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

    def push_many(self, vals: List[TensorVariable]):
        for val in vals:
            self.push(val)

    def pop(self) -> TensorVariable:
        return self.stack.pop()

    def popn(self, n: int) -> List[TensorVariable]:
        assert n >= 0
        return list(reversed([self.pop() for _ in range(n)]))

    def LOAD_FAST(self, inst):
        name = inst.argval
        if name.startswith(".") and name not in self.symbolic_locals:
            # This happens in dict/list comprehensions
            name = name.replace(".", "implicit")
        assert name not in self.cell_and_freevars()
        if name not in self.symbolic_locals:
            unimplemented("undefined LOAD_FAST")
        self.push(self.symbolic_locals[name])
        if name.startswith("___stack"):
            self.symbolic_locals.pop(name)

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
        self.push(VariableBuilder(self, GlobalSource(inst.argval))(value))

    def IMPORT_NAME(self, inst):
        value = importlib.import_module(inst.argval)
        if is_allowed(value):
            self.push(AllowedFunctionOrModuleVariable(value))
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
        val = self.f_builtins[inst.argval]
        assert is_builtin(val)
        self.push(VariableBuilder(self, GlobalSource(inst.argval))(val))

    def jump(self, inst):
        self.instruction_pointer = self.indexof[id(inst.target)]

    JUMP_FORWARD = jump
    JUMP_ABSOLUTE = jump

    POP_JUMP_IF_FALSE = generic_jump(operator.not_, False)
    POP_JUMP_IF_TRUE = generic_jump(operator.truth, False)
    JUMP_IF_FALSE_OR_POP = generic_jump(operator.not_, True)
    JUMP_IF_TRUE_OR_POP = generic_jump(operator.truth, True)

    def SETUP_LOOP(self, inst):
        self.block_depth += 1

    def POP_BLOCK(self, inst):
        assert self.block_depth > 0
        self.block_depth -= 1

    # def SETUP_WITH(self, inst):
    #     # with is handled in resume_execution.py
    #     self.compile_partial_subgraph()
    #     self.add_output_instructions(self.create_call_resume_at(inst))

    def FOR_ITER(self, inst):
        it = self.pop()
        if isinstance(it, ListIteratorVariable):
            self.guards.update(it.guards)
            try:
                val, next_iter = it.next_variables()
                self.replace_all(it, next_iter)
                self.push(next_iter)
                self.push(val)
            except StopIteration:
                self.jump(inst)
        else:
            unimplemented(f"FOR_ITER {typestr(it)}")

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

    def GET_ITER(self, inst):
        self.call_function(BuiltinVariable(iter), [self.pop()], {})

    @break_graph_if_unsupported
    def CALL_FUNCTION(self, inst):
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})

    @break_graph_if_unsupported
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

    @break_graph_if_unsupported
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
        self.push(self.pop())
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
        guards = options.get("guards", set())
        if obj.source:
            source = AttrSource(obj.source, name)
            options["source"] = source
        else:
            source = None

        if isinstance(obj, NNModuleVariable):
            base = self.get_submodule(obj.module_key)
            base_dict = object.__getattribute__(base, "__dict__")
            class_member = True

            if not obj.source:
                unimplemented("GETATTR with no source")

            if name in base_dict:
                subobj = base_dict[name]
            elif name in base_dict["_modules"]:
                subobj = base_dict["_modules"][name]
            elif name in base_dict["_parameters"]:
                subobj = base_dict["_parameters"][name]
            elif name in base_dict["_buffers"]:
                subobj = base_dict["_buffers"][name]
            else:
                subobj = inspect.getattr_static(base, name)
                class_member = False

            if class_member:
                self.push(
                    VariableBuilder(self, NNModuleSource(source))(subobj).add_guards(
                        guards
                    )
                )
            else:
                if istype(subobj, property):
                    self.call_function(
                        UserFunctionVariable(subobj.fget, guards=guards), [obj], {}
                    )
                elif istype(subobj, classmethod):
                    self.push(
                        UserMethodVariable(
                            subobj.__func__,
                            UnsupportedVariable(type(base), guards=guards),
                            **options,
                        )
                    )
                elif istype(subobj, staticmethod):
                    self.push(UserFunctionVariable(subobj.__get__(base), **options))
                elif istype(subobj, types.FunctionType):
                    self.push(UserMethodVariable(subobj, obj, **options))
                else:
                    unimplemented(f"class property {typestr(base)} {typestr(subobj)}")

        elif isinstance(obj, TensorVariable):
            try:
                self.push(obj.get_var_attr(self, name))
            except NotImplementedError:
                self.push(GetAttrVariable(obj, name, **options))
        elif isinstance(obj, AllowedFunctionOrModuleVariable):
            self.push(
                AllowedFunctionOrModuleVariable(
                    value=getattr(obj.value, name), **options
                )
            )
        elif isinstance(obj, PythonModuleVariable):
            member = obj.value.__dict__[name]
            self.push(VariableBuilder(self, source)(member).add_guards(guards))
        elif obj.has_const_attr(self, name) and obj.can_create_guard():
            try:
                options["guards"] = obj.replace_guards(
                    options.get("guards"),
                    GuardBuilder.ID_MATCH,
                    GuardBuilder.OBJECT_MUTATION,
                )
                self.push(ConstantVariable(obj.get_const_attr(self, name), **options))
            except AttributeError:
                unimplemented("dynamic attr UnsupportedVariable")
        else:
            self.push(GetAttrVariable(obj, name, **options))

    def STORE_ATTR(self, inst):
        obj = self.pop()
        val = self.pop()
        unimplemented(f"STORE_ATTR {obj} {val}")

    IMPORT_FROM = LOAD_ATTR

    def STORE_SUBSCR(self, inst):
        val, obj, key = self.popn(3)
        if isinstance(obj, TensorVariable) and val.is_proxy() and key.is_proxy():
            self.create_proxy(
                "call_function",
                operator.setitem,
                (obj.as_proxy(), key.as_proxy(), val.as_proxy()),
                {},
            ),
            # no result is pushed, so need to lift the guards to global
            self.guards.update(
                VariableTracker.propagate([val, obj, key]).get("guards", set())
            )
        elif isinstance(obj, ConstDictVariable):
            obj.call_method(self, "__setattr__", [key, val], {})
        else:
            unimplemented(f"STORE_SUBSCR {obj}[{key}] = {val}")

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
        self.push(ListVariable(items, mutable_local=MutableLocal(), **options))

    def BUILD_LIST_UNPACK(self, inst, cls=ListVariable):
        seqs = self.popn(inst.argval)
        options = VariableTracker.propagate(seqs)
        items = list()
        for seq in seqs:
            try:
                items.extend(seq.unpack_var_sequence(self))
            except NotImplementedError:
                unimplemented(f"BUILD_LIST_UNPACK {seq}")
        self.push(cls(items, mutable_local=MutableLocal(), **options))

    def BUILD_TUPLE_UNPACK(self, inst):
        self.BUILD_LIST_UNPACK(inst, cls=TupleVariable)

    def BUILD_MAP(self, inst):
        items = self.popn(inst.argval * 2)
        options = VariableTracker.propagate(items)
        result = collections.OrderedDict()
        for k, v in zip(items[::2], items[1::2]):
            assert isinstance(k, ConstantVariable)
            result[k.value] = v
        assert len(result) == len(items) / 2
        self.push(ConstDictVariable(result, mutable_local=MutableLocal(), **options))

    def BUILD_CONST_KEY_MAP(self, inst):
        keys = self.pop()
        values = self.popn(inst.argval)
        options = VariableTracker.propagate([keys] + values)
        assert isinstance(keys, ConstantVariable)
        keys = keys.value
        assert istype(keys, tuple)
        assert len(keys) == len(values)
        self.push(
            ConstDictVariable(
                collections.OrderedDict(zip(keys, values)),
                mutable_local=MutableLocal(),
                **options,
            )
        )

    def MAP_ADD(self, inst):
        if sys.version_info < (3, 8):
            v, k = self.popn(2)
        else:
            k, v = self.popn(2)

        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, ConstDictVariable)
        assert obj.mutable_local
        items = collections.OrderedDict(obj.items)
        items[k.as_python_constant()] = v
        self.replace_all(
            obj,
            ConstDictVariable(
                items,
                mutable_local=MutableLocal(),
                **VariableTracker.propagate([obj, k, v]),
            ),
        )

    def LIST_APPEND(self, inst):
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, ListVariable)
        assert obj.mutable_local
        self.replace_all(
            obj,
            ListVariable(
                obj.items + [v],
                mutable_local=MutableLocal(),
                **VariableTracker.propagate([obj, v]),
            ),
        )

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

    def compile_partial_subgraph(self):
        """
        Generate a subgraph to continue execution on user code.
        Automatically restore live variables.
        """
        if self.block_depth != 0:
            unimplemented("compile_partial_subgraph with block_depth != 0")

        self.prune_dead_locals()
        stack_values = list(self.stack)
        root = FakeRootModule(self.nn_modules)

        # Add all the local vars to the "stack" so restore at the end
        restore_vars = []
        val_to_names = collections.OrderedDict()
        if stack_values:
            val_to_names[stack_values[-1]] = list()
        for k, v in self.symbolic_locals.items():
            if isinstance(v.source, LocalSource) and v.source.name() == k:
                continue  # no need to restore initial state
            if v not in val_to_names:
                val_to_names[v] = list()
            val_to_names[v].append(k)
        for v in val_to_names.keys():
            restore_vars.extend(val_to_names[v])
            stack_values.extend([v] * len(val_to_names[v]))

        if (
            stack_values
            and all(isinstance(x, TensorVariable) for x in stack_values)
            and len(set(stack_values)) == len(stack_values)
        ):
            # optimization to generate better code in a common case
            self.add_output_instructions(
                self.compile_subgraph(list(reversed(stack_values)), root)
                + [create_instruction("UNPACK_SEQUENCE", len(stack_values))]
            )
        else:
            graph_output_var = self.new_var("graph_out")
            pass1 = PyCodegen(self, root, graph_output_var)
            pass1.foreach(stack_values)
            # one more time now that we have established tempvars
            pass2 = PyCodegen(
                self,
                root,
                graph_output_var,
                tempvars={val: None for val, count in pass1.uses.items() if count > 1},
            )
            pass2.foreach(stack_values)
            output = []
            if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
                output.extend(
                    self.compile_subgraph(list(pass2.graph_outputs.keys()), root)
                )
                if len(pass2.graph_outputs) != 0:
                    output.append(self.create_store(graph_output_var))
                else:
                    output.append(create_instruction("POP_TOP"))
            self.add_output_instructions(output + pass2.output)

        # restore all the live local vars
        self.add_output_instructions(
            [self.create_store(var) for var in reversed(restore_vars)]
        )
        self.instruction_pointer = None  # exit

    def compile_subgraph(self, rv, root):
        """
        Generate code from self.graph and return the Instruction()s to
        call that generated code.
        """
        assert isinstance(rv, list)
        assert isinstance(root, FakeRootModule)
        for output in rv:
            self.guards.update(output.guards)

        self.create_node(
            "output", "output", (self.create_arg(tuple(x.as_proxy() for x in rv)),), {}
        )
        self.remove_unused_graphargs()
        ncalls = count_calls(self.graph)
        counters["stats"]["calls_captured"] += ncalls
        counters["stats"]["fusions_possible"] += ncalls - 1
        gm = fx.GraphModule(root, self.graph)
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

    def create_call_generated_code(self, fn_name: str, nargs: int) -> List[Instruction]:
        """Call the generated code function stored in fn_name"""
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
        output = []
        for var in freevars:
            assert var in self.cell_and_freevars()
            output.append(
                create_instruction(
                    "LOAD_CLOSURE", self.cell_and_freevars().index(var), var
                )
            )
        output.append(create_instruction("BUILD_TUPLE", len(freevars)))
        output.append(self.create_load_const(code))
        output.append(self.create_load_const(fn_name))
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
        elif n == 4 and sys.version_info >= (3, 8):
            return [create_instruction("ROT_FOUR")]
        else:
            return [
                create_instruction("BUILD_TUPLE", n),
                self._create_load_const(rot_n_helper(n)),
                create_instruction("ROT_TWO"),
                create_instruction("CALL_FUNCTION_EX", 0),
                create_instruction("UNPACK_SEQUENCE", n),
            ]

    def remove_unused_graphargs(self):
        for node in reversed(list(self.graph.nodes)):
            if len(list(node.users)) == 0:
                if node.op == "get_attr":
                    self.graph.erase_node(node)
                elif node.op == "call_function" and node.target is operator.getitem:
                    self.graph.erase_node(node)

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

    def is_constant_or_input(self, value: VariableTracker):
        if value.source is not None:
            return True
        if value.is_python_constant() and is_safe_constant(value.as_python_constant()):
            return True
        return False

    def add_output_instructions(self, prefix: List[Instruction]):
        """
        We call this on the creation of a new compiled subgraph that is inserted
        before user code.
        """
        self.output_instructions.extend(prefix)
        self.instruction_pointer = None  # exit

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
            self.block_depth,
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
            self.block_depth,
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
        self.block_depth = 0

        # Properties of the input/output code
        self.instructions = instructions
        self.indexof = {id(i): n for n, i in enumerate(instructions)}
        self.f_globals = f_globals
        self.f_builtins = f_builtins
        self.code_options = code_options
        self.output_instructions = []
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
            (k, VariableBuilder(self, LocalSource(k))(f_locals[k]))
            for k in vars
            if k in f_locals
        )

    def should_compile_partial_graph(self):
        return True

    def create_call_resume_at(self, inst):
        self.instruction_pointer = None

        if inst.opname == "RETURN_VALUE":
            return [create_instruction("RETURN_VALUE")]

        reads = livevars_analysis(self.instructions, inst)
        argnames = tuple(
            k
            for k in self.symbolic_locals.keys()
            if k in reads and k not in self.cell_and_freevars()
        )
        nargs = len(self.stack) + len(argnames)

        name = unique_id(f"__resume_at_{self.next_instruction.offset}")

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
        if count_calls(self.graph) == 0:
            unimplemented("no graph found")
        self.instruction_pointer = None
        self.compile_partial_subgraph()
        self.output_instructions.append(create_instruction("RETURN_VALUE"))


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
            unimplemented(
                f"inline in skipfiles: {func.get_name()} {func.get_filename()}"
            )

        sub_locals = func.bind_args(parent, args, kwargs)
        for v in sub_locals.values():
            if not isinstance(v, VariableTracker):
                unimplemented(f"unconverted arg {v}")

        code = func.get_code()

        if config.trace:
            print("INLINING ", code)
            print(dis.dis(code))
            print()

        if is_generator(code):
            tracer = InliningGeneratorInstructionTranslator(
                parent, code, sub_locals, func.get_globals()
            )
        else:
            tracer = InliningInstructionTranslator(
                parent, code, sub_locals, func.get_globals()
            )

        tracer.run()
        assert tracer.symbolic_result is not None
        func.export_freevars(parent, tracer)
        parent.guards.update(tracer.guards)

        if config.trace:
            print("DONE INLINING", code)

        if is_generator(code):
            assert tracer.symbolic_result.as_python_constant() is None
            return ListIteratorVariable(
                tracer.generated_items,
                **VariableTracker.propagate(tracer.symbolic_result),
            )
        else:
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


class InliningGeneratorInstructionTranslator(InliningInstructionTranslator):
    def __init__(self, *args, **kwargs):
        super(InliningGeneratorInstructionTranslator, self).__init__(*args, **kwargs)
        self.generated_items = []

    def YIELD_VALUE(self, inst: Instruction):
        self.generated_items.append(self.pop())
        # TODO(jansel): figure out why this is needed, it isn't in the docs for YIELD_VALUE
        self.push(ConstantVariable(None))


class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""

    def __init__(self, nn_modules: dict):
        super(FakeRootModule, self).__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)

    def __repr__(self):
        return "FakeRootModule(...)"


class PyCodegen(object):
    def __init__(
        self,
        tx: InstructionTranslatorBase,
        root: FakeRootModule,
        graph_output_var: str,
        tempvars=None,
    ):
        self.top_of_stack = None
        self.uses = collections.Counter()
        self.graph_outputs = collections.OrderedDict()
        self.output: List[Instruction] = []
        self.tempvars = tempvars or {}
        self.tx = tx
        self.root = root
        self.graph_output_var = graph_output_var

    def __call__(self, value):
        assert isinstance(value, VariableTracker)
        output = self.output
        graph_outputs = self.graph_outputs

        if self.top_of_stack is value:
            output.append(create_instruction("DUP_TOP"))
            return

        if self.tempvars.get(value) is not None:
            output.append(self.create_load(self.tempvars[value]))
            return

        self.guards.update(value.guards)
        if value.source is not None:
            output.extend(value.source.reconstruct(self))
        elif value.is_python_constant() and is_safe_constant(
            value.as_python_constant()
        ):
            output.append(self.create_load_const(value.as_python_constant()))
        elif isinstance(value, TensorVariable):
            if value not in graph_outputs:
                graph_outputs[value] = len(graph_outputs)
            output.append(self.create_load(self.graph_output_var))
            output.append(self._create_load_const(graph_outputs[value]))
            output.append(create_instruction("BINARY_SUBSCR"))
        elif isinstance(value, NNModuleVariable):
            parts = value.module_key.split(".")
            if parts[0] in self.code_options["co_varnames"]:
                output.append(self.create_load(parts[0]))
                parts = parts[1:]
            else:
                output.append(self.create_load_output(self.root))
            for part in parts:
                output.append(self.create_load_attr(part))
        else:
            self.uses[value] += 1
            try:
                output.extend(value.reconstruct(self))
            except NotImplementedError:
                unimplemented(f"reconstruct: {value}")
            if value in self.tempvars:
                var = self.new_var()
                self.tempvars[value] = var
                output.extend([create_instruction("DUP_TOP"), self.create_store(var)])

        self.top_of_stack = value

    def foreach(self, items):
        for i in items:
            self(i)

    def __getattr__(self, item):
        return getattr(self.tx, item)


@lru_cache(32)
def rot_n_helper(n):
    assert n > 1
    vars = [f"v{i}" for i in range(n)]
    rotated = reversed(vars[-1:] + vars[:-1])
    fn = eval(f"lambda {','.join(vars)}: ({','.join(rotated)})")
    fn.__name__ = f"rot_{n}_helper"
    return fn
