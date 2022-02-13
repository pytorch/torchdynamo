import collections
import dis
import functools
import importlib
import inspect
import itertools
import logging
import operator
import sys
import types
import typing
from typing import Any
from typing import Dict
from typing import List
from unittest.mock import patch

import torchdynamo.side_effects

from . import config
from . import skipfiles
from .allowed_functions import is_allowed
from .allowed_functions import is_builtin
from .allowed_functions import is_disallowed
from .bytecode_analysis import livevars_analysis
from .bytecode_transformation import Instruction
from .bytecode_transformation import cleaned_instructions
from .bytecode_transformation import create_instruction
from .bytecode_transformation import is_generator
from .bytecode_transformation import unique_id
from .codegen import PyCodegen
from .output_graph import OutputGraph
from .resume_execution import ContinueExecutionCache
from .utils import Unsupported
from .utils import counters
from .utils import istype
from .utils import unimplemented
from .utils import warning
from .variable_builder import VariableBuilder
from .variable_source import AttrSource
from .variable_source import GlobalSource
from .variable_source import LocalSource
from .variable_source import NNModuleSource
from .variable_tracker import AllowedFunctionOrModuleVariable
from .variable_tracker import BaseListVariable
from .variable_tracker import BlackHoleVariable
from .variable_tracker import BuiltinVariable
from .variable_tracker import ClosureVariable
from .variable_tracker import ConstantVariable
from .variable_tracker import ConstDictVariable
from .variable_tracker import ContextManagerVariable
from .variable_tracker import GetAttrVariable
from .variable_tracker import LambdaVariable
from .variable_tracker import ListIteratorVariable
from .variable_tracker import ListVariable
from .variable_tracker import MutableLocal
from .variable_tracker import NamedTupleVariable
from .variable_tracker import NestedUserFunctionVariable
from .variable_tracker import NNModuleVariable
from .variable_tracker import PythonModuleVariable
from .variable_tracker import SliceVariable
from .variable_tracker import TensorVariable
from .variable_tracker import TupleVariable
from .variable_tracker import UnknownVariable
from .variable_tracker import UnsupportedVariable
from .variable_tracker import UserFunctionVariable
from .variable_tracker import UserMethodVariable
from .variable_tracker import VariableTracker
from .variable_tracker import typestr

log = logging.getLogger(__name__)


def stack_op(fn: typing.Callable):
    nargs = len(inspect.signature(fn).parameters)
    fn_var = BuiltinVariable(fn)

    @functools.wraps(fn)
    def impl(self: "InstructionTranslatorBase", inst: Instruction):
        self.push(fn_var.call_function(self, self.popn(nargs), {}))

    return impl


def generic_jump(truth_fn: typing.Callable, push: bool):
    def inner(self: "InstructionTranslatorBase", inst: Instruction):
        value: VariableTracker = self.pop()
        self.output.guards.update(value.guards)
        if value.is_python_constant():
            if truth_fn(value.as_python_constant()):
                push and self.push(value)
                self.jump(inst)

        elif isinstance(value, TensorVariable) and self.should_compile_partial_graph():
            # compile a partial subgraph prefix then jump into user code
            self.push(value)
            self.output.compile_subgraph(self)
            self.pop()

            if_next = self.create_call_resume_at(self.next_instruction)
            push and self.push(value)
            if_jump = self.create_call_resume_at(inst.target)

            self.output.add_output_instructions(
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
            return inner_fn(self, inst)
        except Unsupported:
            if not self.should_compile_partial_graph():
                raise
        self.restore_graphstate(state)
        self.output.compile_subgraph(self)
        # note, assuming inst pushes 1
        vars = self.popn(1 - dis.stack_effect(inst.opcode, inst.arg))
        warning(f"breaking graph: {vars[0]}")
        self.output.add_output_instructions([inst])
        self.push(UnknownVariable())
        self.output.add_output_instructions(
            self.create_call_resume_at(self.next_instruction)
        )

    return wrapper


class InstructionTranslatorBase(object):
    def cell_and_freevars(self):
        if not hasattr(self, "_cell_and_freevars"):
            self._cell_and_freevars = tuple(
                self.code_options["co_cellvars"] or []
            ) + tuple(self.code_options["co_freevars"] or [])
        return self._cell_and_freevars

    def prune_dead_locals(self):
        reads = livevars_analysis(self.instructions, self.current_instruction)
        # implicit use by super()
        # reads = reads | {"__class__"}
        # output variables?
        reads = reads | set(self.cell_and_freevars())
        self.symbolic_locals = collections.OrderedDict(
            [(k, v) for k, v in self.symbolic_locals.items() if k in reads]
        )

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

    def replace_all(self, oldvar: VariableTracker, newvar: VariableTracker):
        if isinstance(oldvar.mutable_local, torchdynamo.side_effects.Mutable):
            newvar = self.output.side_effects.mutation(oldvar, newvar)
        else:
            assert isinstance(
                oldvar.mutable_local, torchdynamo.variable_tracker.MutableLocal
            )
            newvar = newvar.clone(
                mutable_local=torchdynamo.variable_tracker.MutableLocal()
            )

        def repl(v: VariableTracker):
            if v.mutable_local is oldvar.mutable_local:
                return newvar
            return v

        self.output.side_effects.apply(repl)
        self.stack = [VariableTracker.apply(repl, x) for x in self.stack]
        for k, x in self.symbolic_locals.items():
            self.symbolic_locals[k] = VariableTracker.apply(repl, x)
        return newvar

    def inline_user_function_return(self, fn, args, kwargs):
        """
        A call to some user defined function by inlining it.
        """
        state = self.copy_graphstate()
        try:
            result = InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
            self.output.guards.update(fn.guards)
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
            return inst.opname != "RETURN_VALUE"
        except Unsupported:
            if not self.checkpoint:
                raise

        # generate code from checkpoint
        assert not self.output.output_instructions
        continue_inst, state = self.checkpoint
        self.restore_graphstate(state)
        self.output.compile_subgraph(self)
        self.output.add_output_instructions(
            [create_instruction("JUMP_ABSOLUTE", target=continue_inst)]
            + self.instructions
        )

    def run(self):
        try:
            while (
                self.instruction_pointer is not None
                and not self.output.should_exit
                and self.step()
            ):
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

    def DELETE_FAST(self, inst):
        del self.symbolic_locals[inst.argval]

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
        if self.output.root_globals is self.f_globals:
            source = GlobalSource(inst.argval)
        else:
            source = AttrSource(
                self.import_source(self.f_globals["__name__"]), inst.argval
            )
        self.push(VariableBuilder(self, source)(value))

    def import_source(self, module_name):
        """Create an alias to a module for use in guards"""
        value = importlib.import_module(module_name)
        alias = f"__import_{module_name.replace('.', '_dot_')}"
        f_globals = self.output.root_globals
        assert alias not in f_globals or f_globals[alias] is value
        f_globals[alias] = value
        self.output.update_co_names(alias)
        return GlobalSource(alias)

    def IMPORT_NAME(self, inst):
        level, fromlist = self.popn(2)
        if level.as_python_constant() != 0:
            unimplemented("IMPORT_NAME with level")

        value = importlib.import_module(inst.argval)
        source = self.import_source(inst.argval)

        if is_allowed(value):
            self.push(AllowedFunctionOrModuleVariable(value, source=source))
        elif istype(value, types.ModuleType):
            self.push(PythonModuleVariable(value, source=source))
        else:
            unimplemented(f"IMPORT_NAME {typestr(value)}")

    def IMPORT_FROM(self, inst):
        self.DUP_TOP(inst)
        self.LOAD_ATTR(inst)

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
        # only exists in python<=3.7
        self.block_stack.append(inst.target)

    def POP_BLOCK(self, inst):
        self.block_stack.pop()

    def SETUP_WITH(self, inst):
        ctx = self.pop()
        self.output.guards.update(ctx.guards)
        if not isinstance(ctx, ContextManagerVariable):
            unimplemented(f"SETUP_WITH {ctx}")

        def exit(*args):
            return ctx.exit(self, *args)

        self.push(LambdaVariable(exit, **VariableTracker.propagate(ctx)))
        self.block_stack.append(inst.target)
        self.push(ctx.enter(self))

    def SETUP_FINALLY(self, inst):
        self.block_stack.append(inst.target)

    def BEGIN_FINALLY(self, inst):
        self.push(None)

    def WITH_CLEANUP_START(self, inst):
        exit, exc = self.popn(2)
        assert exc is None
        self.push(exc)
        self.push(exit.call_function(self, [ConstantVariable(None)] * 3, {}))

    def WITH_CLEANUP_FINISH(self, inst):
        self.popn(2)
        self.push(None)

    def END_FINALLY(self, inst):
        assert self.pop() is None

    def FOR_ITER(self, inst):
        it = self.pop()
        if isinstance(it, ListIteratorVariable):
            self.output.guards.update(it.guards)
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
            isinstance(left, (TensorVariable, NNModuleVariable, BaseListVariable))
            and isinstance(right, ConstantVariable)
            and right.value is None
            and op in supported_is_const
        ):
            # <non-None> is None
            self.push(
                ConstantVariable(
                    supported_is_const[op](object(), right.value), **options
                )
            )
        elif (
            isinstance(left, TensorVariable) or isinstance(right, TensorVariable)
        ) and op in supported_tensors:
            self.push(
                TensorVariable.create(
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
        elif op in ("in", "not in"):
            self.push(right.call_method(self, "__contains__", [left], {}))
            if op == "not in":
                self.UNARY_NOT(inst)
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
            base = self.output.get_submodule(obj.module_key)
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

        elif isinstance(obj, (TensorVariable, NamedTupleVariable, ConstantVariable)):
            try:
                self.push(
                    obj.get_var_attr(self, name).clone(source=source).add_guards(guards)
                )
            except NotImplementedError:
                self.push(GetAttrVariable(obj, name, **options))
        elif isinstance(obj, AllowedFunctionOrModuleVariable):
            member = getattr(obj.value, name)
            if not is_disallowed(member):
                self.push(AllowedFunctionOrModuleVariable(member, **options))
            elif ConstantVariable.is_literal(member):
                self.push(ConstantVariable(member, **options))
            else:
                self.push(VariableBuilder(self, source)(member).add_guards(guards))
        elif isinstance(obj, PythonModuleVariable):
            member = obj.value.__dict__[name]
            self.push(VariableBuilder(self, source)(member).add_guards(guards))
        elif isinstance(obj, UnsupportedVariable) and name in getattr(
            obj.value, "__dict__", {}
        ):
            subobj = inspect.getattr_static(obj.value, name)
            assert id(subobj) == id(obj.value.__dict__[name])
            self.push(VariableBuilder(self, source)(subobj).add_guards(guards))
        elif istype(obj, UserFunctionVariable) and name in ("__name__", "__module__"):
            self.push(
                ConstantVariable(
                    getattr(obj.fn, name), **VariableTracker.propagate(obj)
                )
            )
        else:
            self.push(GetAttrVariable(obj, name, **options))

    def STORE_ATTR(self, inst):
        if isinstance(self.stack[-1], BlackHoleVariable):
            val, obj = self.popn(2)
            self.output.guards.update(
                obj.call_method(
                    self, "__setattr__", [ConstantVariable(inst.argval), val], {}
                ).guards
            )
            return

        if not self.should_compile_partial_graph():
            unimplemented("inline STORE_ATTR")
        warning("breaking graph: STORE_ATTR")
        self.output.compile_subgraph(self)
        self.output.add_output_instructions([inst])
        self.popn(2)
        self.output.add_output_instructions(
            self.create_call_resume_at(self.next_instruction)
        )

    def STORE_SUBSCR(self, inst):
        val, obj, key = self.popn(3)
        result = obj.call_method(self, "__setitem__", [key, val], {})
        # no result is pushed, so need to lift the guards to global
        self.output.guards.update(result.guards)

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

    BUILD_TUPLE_UNPACK_WITH_CALL = BUILD_TUPLE_UNPACK

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
                closure_scope=self,
                **options,
            )
        )

    def UNPACK_SEQUENCE(self, inst):
        seq = self.pop()
        options = VariableTracker.propagate([seq])
        if isinstance(seq, BaseListVariable):
            assert len(seq.items) == inst.argval
            self.output.guards.update(seq.guards)
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
                self.push(TensorVariable.create(proxy[i], **options))
        elif isinstance(seq, GetAttrVariable) and isinstance(seq.obj, TensorVariable):
            # x, y = a.shape
            proxy = getattr(seq.obj.as_proxy(), seq.name)
            for i in reversed(range(inst.argval)):
                self.push(TensorVariable.create(proxy[i], **options))
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

    def copy_graphstate(self):
        """Create a checkpoint of the current state by copying everything"""
        return (
            self.output.copy_graphstate(),
            collections.OrderedDict(self.symbolic_locals),
            list(self.stack),
            list(self.block_stack),
            self.instruction_pointer,
            self.current_instruction,
            self.next_instruction,
            self.lineno,
        )

    def restore_graphstate(self, state):
        """Restore a checkpoint created by self.copy_graphstate()"""
        (
            output_state,
            self.symbolic_locals,
            self.stack,
            self.block_stack,
            self.instruction_pointer,
            self.current_instruction,
            self.next_instruction,
            self.lineno,
        ) = state
        self.output.restore_graphstate(output_state)

    def __init__(
        self,
        output: OutputGraph,
        instructions: List[Instruction],
        f_globals: Dict[str, Any],
        f_builtins: Dict[str, Any],
        code_options: Dict[str, Any],
        symbolic_locals: Dict[str, VariableTracker],
        f_code: types.CodeType,
    ):
        super(InstructionTranslatorBase, self).__init__()

        # Mutable state checkpointed by copy_graphstate()
        self.output = output
        self.symbolic_locals = symbolic_locals
        self.stack = []
        self.instruction_pointer = 0
        self.current_instruction = create_instruction("NOP")
        self.next_instruction = None
        self.block_stack = []
        self.lineno = code_options.get("co_firstlineno")

        # Properties of the input/output code
        self.instructions = instructions
        self.indexof = {id(i): n for n, i in enumerate(instructions)}
        self.f_globals = f_globals
        self.f_builtins = f_builtins
        self.code_options = code_options
        self.f_code = f_code

        self.checkpoint = None


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
            output=OutputGraph(f_globals, code_options, compiler_fn),
            instructions=instructions,
            f_globals=f_globals,
            f_builtins=f_builtins,
            code_options=code_options,
            symbolic_locals=collections.OrderedDict(),  # set below
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
        return len(self.block_stack) == 0

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

        name = unique_id(f"__resume_at_{inst.offset}")

        new_code: types.CodeType = ContinueExecutionCache.lookup(
            self.f_code, inst.offset, len(self.stack), argnames
        )

        cg = PyCodegen(self)

        if new_code.co_freevars:
            cg.make_function_with_closure(name, new_code, len(self.stack))
        else:
            self.output.install_global(
                name, types.FunctionType(new_code, self.f_globals, name)
            )
            cg.extend_output(cg.load_function_name(name, len(self.stack)))

        cg.extend_output([cg.create_load(k) for k in argnames])
        cg.extend_output(
            [
                create_instruction("CALL_FUNCTION", nargs),
                create_instruction("RETURN_VALUE"),
            ]
        )
        return cg.get_instructions()

    def RETURN_VALUE(self, inst):
        if self.output.count_calls() == 0:
            unimplemented("no graph found")
        self.instruction_pointer = None
        self.output.compile_subgraph(self)
        self.output.add_output_instructions([create_instruction("RETURN_VALUE")])


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
            unimplemented(f"inline with __closure__ {func}")
        if func.has_self():
            unimplemented("inline with __self__")
        if skipfiles.check(func.get_filename()) and not skipfiles.is_torch_nn(
            func.get_filename()
        ):
            unimplemented(
                f"inline in skipfiles: {func.get_name()} {func.get_filename()}"
            )

        sub_locals = func.bind_args(parent, args, kwargs)
        sub_locals.update(func.closure_vars(parent))
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
            output=parent.output,
            f_globals=f_globals,
            f_builtins=f_globals["__builtins__"],
            symbolic_locals=symbolic_locals,
            instructions=cleaned_instructions(code),
            code_options={k: getattr(code, k) for k in dir(code)},
            f_code=code,
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
