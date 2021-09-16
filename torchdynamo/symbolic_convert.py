import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import operator
import types
import typing
from typing import List

import torch
from torch import fx

from .allowed_functions import is_allowed
from .bytecode_transformation import Instruction
from .bytecode_transformation import cleaned_instructions
from .bytecode_transformation import create_instruction
from .bytecode_transformation import unique_id
from .guards import Guard
from .guards import GuardBuilder
from .guards import GuardSource
from .variable_tracker import AllowedFunctionOrModuleVariable, PythonModuleVariable
from .variable_tracker import BaseListVariable
from .variable_tracker import BuiltinVariable
from .variable_tracker import ConstDictVariable
from .variable_tracker import ConstantVariable
from .variable_tracker import GetAttrVariable
from .variable_tracker import ListIteratorVariable
from .variable_tracker import ListVariable
from .variable_tracker import NNModuleVariable
from .variable_tracker import SliceVariable
from .variable_tracker import TensorVariable
from .variable_tracker import TracingSupported
from .variable_tracker import TupleVariable
from .variable_tracker import UserFunctionVariable
from .variable_tracker import UserMethodVariable
from .variable_tracker import VariableTracker

counters = collections.defaultdict(collections.Counter)


def typestr(*objs):
    return " ".join(type(obj).__name__ for obj in objs)


def unimplemented(msg: str):
    counters["unimplemented"][msg] += 1
    raise NotImplementedError(msg)


def stack_op(fn):
    nargs = len(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def impl(self, inst):
        inputs = self.popn(nargs)

        options = VariableTracker.propagate(inputs)
        if any(isinstance(i, TensorVariable) for i in inputs):
            val = TensorVariable(fn(*[i.as_proxy() for i in inputs]), **options)
        elif all(isinstance(i, ConstantVariable) for i in inputs):
            # constant fold
            val = ConstantVariable(fn(*[i.value for i in inputs]), **options)
        else:
            unimplemented(f"stack_op {typestr(*inputs)}")

        self.push(val)

    return impl


@dataclasses.dataclass
class LocalArg:
    name: str

    def __len__(self):
        return 1

    def load(self, tracer):
        return [tracer.create_load_fast(self.name)]


@dataclasses.dataclass
class GlobalArg:
    name: str

    def __len__(self):
        return 1

    def load(self, tracer):
        return [tracer.create_load_global(self.name)]


@dataclasses.dataclass
class TensorListArgs:
    name: str
    count: int

    def __len__(self):
        return self.count

    def load(self, tracer):
        return [
            tracer.create_load_fast(self.name),
            create_instruction("UNPACK_SEQUENCE", self.count),
        ]


class InstructionTranslatorBase(fx.Tracer):
    def create_load_fast(self, name):
        assert name in self.code_options["co_varnames"]
        return create_instruction(
            "LOAD_FAST", self.code_options["co_varnames"].index(name), name
        )

    def create_store_fast(self, name):
        assert name in self.code_options["co_varnames"]
        return create_instruction(
            "STORE_FAST", self.code_options["co_varnames"].index(name), name
        )

    def create_load_global(self, name):
        assert name in self.code_options["co_names"]
        return create_instruction(
            "LOAD_GLOBAL", self.code_options["co_names"].index(name), name
        )

    def create_load_const(self, value):
        co_consts = self.code_options["co_consts"]
        assert isinstance(co_consts, tuple)
        if value not in co_consts:
            co_consts = co_consts + (value,)
            self.code_options["co_consts"] = co_consts
        return create_instruction("LOAD_CONST", co_consts.index(value), value)

    def wrap_local(self, name, value):
        """
        Turn an arg/input to the frame into a VariableTracker instance
        """
        if isinstance(value, torch.Tensor):
            self.graphargs.append(LocalArg(name))
            return TensorVariable(
                proxy=self.create_graph_input(name),
                state=TracingSupported.YES,
                guards={Guard(name, GuardSource.LOCAL, GuardBuilder.TYPE_MATCH)},
            )
        elif isinstance(value, torch.nn.Module):
            key = f"{name}_{next(self.cnt)}"
            self.nn_modules[key] = value
            return NNModuleVariable(
                key=key,
                state=TracingSupported.YES,
                guards={Guard(name, GuardSource.LOCAL, GuardBuilder.VALUE_MATCH)},
            )
        elif value is True or value is False or value is None:
            # For these, just specialize on exact value
            return ConstantVariable(
                value=value,
                guards={Guard(name, GuardSource.LOCAL, GuardBuilder.VALUE_MATCH)},
            )
        elif type(value) in (tuple, list) and all(
            isinstance(x, torch.Tensor) for x in value
        ):
            guards = {Guard(name, GuardSource.LOCAL, GuardBuilder.FIXED_TENSOR_LIST)}
            self.graphargs.append(LocalArg(name))
            proxy = self.create_graph_input(name)
            items = []
            for i in range(len(value)):
                items.append(
                    TensorVariable(
                        proxy=proxy[i],
                        state=TracingSupported.YES,
                        guards=guards,
                    )
                )
            cls = {tuple: TupleVariable, list: ListVariable}[type(value)]
            return cls(items, guards=guards)
        else:
            unimplemented(f"wrap_local: {typestr(value)}")

    def mark_initial_state(self):
        for k, v in self.symbolic_locals.items():
            self.symbolic_locals[k] = v.with_initial_name(k)

    def create_graph_input(self, name):
        placeholders = [n for n in self.graph.nodes if n.op == "placeholder"]
        if placeholders:
            ctx = self.graph.inserting_after(placeholders[-1])
        else:
            ctx = self.graph.inserting_before(None)
        with ctx:
            return self.create_proxy("placeholder", f"{name}_{next(self.cnt)}", (), {})

    def call_function(self, fn, args, kwargs):
        if isinstance(fn, AllowedFunctionOrModuleVariable):
            options = VariableTracker.propagate(
                [
                    fn,
                ]
                + list(args)
                + list(kwargs.values())
            )
            self_should_be_none = getattr(fn.value, "__self__", None)
            if self_should_be_none is not None:
                # weird ones like torch.nn.functional.avg_pool2d have __self__
                assert isinstance(
                    self_should_be_none, types.ModuleType
                ) and self_should_be_none.__name__ == getattr(
                    fn.value, "__module__", None
                )
            proxy_args = tuple(arg.as_proxy() for arg in args)
            proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
            self.push(
                TensorVariable(
                    proxy=self.create_proxy(
                        "call_function", fn.value, proxy_args, proxy_kwargs
                    ),
                    **options,
                )
            )
        elif isinstance(fn, GetAttrVariable):
            name = fn.name
            obj = fn.obj
            args = [obj] + list(args)
            options = VariableTracker.propagate(
                [
                    fn,
                ]
                + list(args)
                + list(kwargs.values())
            )
            proxy_args = tuple(arg.as_proxy() for arg in args)
            proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
            self.push(
                TensorVariable(
                    proxy=self.create_proxy(
                        "call_method", name, proxy_args, proxy_kwargs
                    ),
                    **options,
                )
            )
        elif isinstance(fn, NNModuleVariable):
            mod = self.get_submodule(fn.key)
            if is_allowed(mod.__class__):
                options = VariableTracker.propagate([fn] + args)
                proxy_args = tuple(x.as_proxy() for x in args)
                self.push(
                    TensorVariable(
                        proxy=self.create_proxy("call_module", fn.key, proxy_args, {}),
                        **options,
                    )
                )
            else:
                forward = mod.__class__.forward
                assert forward is not torch.nn.Module.forward
                self.inline_user_function(fn.guards, forward, [fn] + args, kwargs)
        elif isinstance(fn, UserFunctionVariable):
            self.inline_user_function(fn.guards, fn.fn, fn.self_args() + args, kwargs)
        elif isinstance(fn, BuiltinVariable):
            allargs = args + list(kwargs.values())
            options = VariableTracker.propagate(allargs)
            constant_args = all(isinstance(x, ConstantVariable) for x in allargs)
            if fn.fn is range and constant_args:
                items = list(
                    range(
                        *[x.value for x in args],
                        **{k: v.value for k, v in kwargs.items()},
                    )
                )
                self.push(ListVariable(items, **options))
            elif fn.fn is iter and args and isinstance(args[0], BaseListVariable):
                assert not kwargs and len(args) == 1
                self.push(ListIteratorVariable(args[0].items, **options))
            else:
                unimplemented(f"builtin call {fn.fn}")
        else:
            unimplemented(f"call_function {type(fn).__name__}")

    def inline_user_function(self, guards, fn, args, kwargs):
        """
        A call to some user defined function by inlining it.
        """
        self.guards.update(guards)
        self.push(RecursiveInstructionTranslator.inline_call(self, fn, args, kwargs))

    def step(self):
        inst = self.instructions[self.instruction_pointer]
        self.instruction_pointer += 1

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
                self.compile_partial_subgraph(
                    self.stack,
                    [create_instruction("JUMP_ABSOLUTE", target=continue_inst)],
                )
            else:
                raise

    def run(self):
        while self.step():
            pass

    def push(self, val):
        self.stack.append(val)

    def pop(self):
        return self.stack.pop()

    def popn(self, n):
        return list(reversed([self.pop() for _ in range(n)]))

    def LOAD_FAST(self, inst):
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
                )
            )
        elif isinstance(value, torch.Tensor):
            # turn a load of a global tensor into an arg for the graph
            self.graphargs.append(GlobalArg(inst.argval))
            self.push(
                TensorVariable(
                    proxy=self.create_graph_input(inst.argval),
                    state=TracingSupported.YES,
                    guards={
                        Guard(inst.argval, GuardSource.GLOBAL, GuardBuilder.TYPE_MATCH)
                    },
                )
            )
        elif isinstance(value, types.FunctionType):
            self.push(
                UserFunctionVariable(
                    value,
                    guards={
                        Guard(
                            inst.argval, GuardSource.GLOBAL, GuardBuilder.FUNCTION_MATCH
                        )
                    },
                )
            )
        elif isinstance(value, types.ModuleType):
            self.push(
                PythonModuleVariable(
                    value,
                    guards={
                        Guard(
                            inst.argval, GuardSource.GLOBAL, GuardBuilder.FUNCTION_MATCH
                        )
                    },
                )
            )
        elif isinstance(value, torch.nn.Module):
            key = unique_id(inst.argval)
            self.nn_modules[key] = value
            self.push(
                NNModuleVariable(
                    key,
                    guards={
                        Guard(inst.argval, GuardSource.GLOBAL, GuardBuilder.VALUE_MATCH)
                    },
                )
            )
        elif (
            isinstance(value, types.BuiltinFunctionType)
            and self.should_compile_partial_graph()
        ):
            self.compile_partial_subgraph(
                self.stack,
                [
                    create_instruction("JUMP_ABSOLUTE", target=inst),
                ],
            )
        else:
            unimplemented(f"LOAD_GLOBAL {typestr(value)}")

    def load_builtin(self, inst):
        assert inst.argval in self.f_builtins
        self.push(BuiltinVariable(self.f_builtins[inst.argval]))

    def jump(self, inst):
        self.instruction_pointer = self.indexof[id(inst.target)]

    JUMP_FORWARD = jump
    JUMP_ABSOLUTE = jump

    def POP_JUMP_IF_FALSE(self, inst):
        value = self.pop()
        self.guards.update(value.guards)
        if isinstance(value, (AllowedFunctionOrModuleVariable, ConstantVariable)):
            if not value.value:
                self.jump(inst)
        elif isinstance(value, TensorVariable) and self.should_compile_partial_graph():
            # compile a partial subgraph prefix then jump into user code
            assert len(self.stack) == 0
            jump_to_user_code = [
                create_instruction("POP_JUMP_IF_FALSE", target=inst.target),
                create_instruction(
                    "JUMP_ABSOLUTE", target=self.instructions[self.instruction_pointer]
                ),
            ]
            self.compile_partial_subgraph([value], jump_to_user_code)
        else:
            unimplemented(f"POP_JUMP_IF_FALSE {typestr(value)}")

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
            and isinstance(right, ConstantVariable)
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
        assert isinstance(argsvars, BaseListVariable)
        assert isinstance(kwargsvars, ConstDictVariable)
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
            key = f"{obj.key}.{name}"
            subobj = self.get_submodule(key)
            if isinstance(subobj, torch.Tensor):
                self.push(
                    TensorVariable(
                        proxy=self.create_proxy("get_attr", key, tuple(), {}), **options
                    )
                )
            elif isinstance(subobj, torch.nn.Module):
                self.push(NNModuleVariable(key, **options))
            elif isinstance(subobj, (int, float, bool, type(None))):
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
            elif callable(subobj):
                base = self.get_submodule(obj.key)
                method = getattr(base.__class__, name, None)
                if isinstance(method, types.FunctionType):
                    self.push(UserMethodVariable(method, obj, **options))
                else:
                    unimplemented("nn.Module callable")
            else:
                unimplemented(f"nn.Module attr {type(subobj).__name__}")
        elif isinstance(obj, TensorVariable):
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
        else:
            unimplemented("LOAD_ATTR")

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
        assert isinstance(keys, tuple)
        assert len(keys) == len(values)
        self.push(ConstDictVariable(dict(zip(keys, values)), **options))

    def UNPACK_SEQUENCE(self, inst):
        seq = self.pop()
        options = VariableTracker.propagate([seq])
        if isinstance(seq, BaseListVariable):
            assert len(seq.items) == inst.argval
            self.guards.update(seq.guards)
            for i in reversed(seq.items):
                self.push(i)
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
            unimplemented(f"UNPACK_SEQUENCE {type(seq).__name__}")

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

    UNARY_POSITIVE = stack_op(lambda tos: +tos)
    UNARY_NEGATIVE = stack_op(lambda tos: -tos)
    UNARY_NOT = stack_op(lambda tos: not tos)
    UNARY_INVERT = stack_op(lambda tos: ~tos)

    BINARY_POWER = stack_op(lambda tos1, tos: tos1 ** tos)
    BINARY_MULTIPLY = stack_op(lambda tos1, tos: tos1 * tos)
    BINARY_FLOOR_DIVIDE = stack_op(lambda tos1, tos: tos1 // tos)
    BINARY_TRUE_DIVIDE = stack_op(lambda tos1, tos: tos1 / tos)
    BINARY_MODULO = stack_op(lambda tos1, tos: tos1 % tos)
    BINARY_ADD = stack_op(lambda tos1, tos: tos1 + tos)
    BINARY_SUBTRACT = stack_op(lambda tos1, tos: tos1 - tos)
    BINARY_SUBSCR = stack_op(lambda tos1, tos: tos1[tos])
    BINARY_LSHIFT = stack_op(lambda tos1, tos: tos1 << tos)
    BINARY_RSHIFT = stack_op(lambda tos1, tos: tos1 >> tos)
    BINARY_XOR = stack_op(lambda tos1, tos: tos1 ^ tos)
    BINARY_OR = stack_op(lambda tos1, tos: tos1 | tos)

    # TODO(jansel): looks like FX lacks support for this one, submit upstream fix
    BINARY_MATRIX_MULTIPLY = stack_op(lambda tos1, tos: tos1.__matmul__(tos))

    # TODO(jansel): FX is buggy here too, submit upstream fix
    # BINARY_AND = stack_op(lambda tos1, tos: tos1 & tos)
    # workaround version:
    def BINARY_AND(self, inst):
        inputs = self.popn(2)

        options = VariableTracker.propagate(inputs)
        if any(isinstance(i, TensorVariable) for i in inputs):
            val = TensorVariable(
                self.create_proxy("call_function", operator.and_, inputs, {}), **options
            )
        elif all(isinstance(i, ConstantVariable) for i in inputs):
            # constant fold
            val = ConstantVariable(operator.and_(*[i.value for i in inputs]), **options)
        else:
            unimplemented(f"stack_op {typestr(*inputs)}")

        self.push(val)

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
        if isinstance(rv, TensorVariable):
            self.create_node("output", "output", (self.create_arg(rv.as_proxy()),), {})
        elif isinstance(rv, list):
            outputs = []
            for x in rv:
                if isinstance(x, TensorVariable):
                    outputs.append(self.create_arg(x.as_proxy()))
                elif isinstance(x, NNModuleVariable):
                    outputs.append(
                        self.create_arg(
                            self.create_proxy("get_attr", x.key, tuple(), {})
                        )
                    )
                else:
                    unimplemented(f"restore state for {type(x).__name__}")
            outputs = tuple(outputs)
            rv = VariableTracker(**VariableTracker.propagate(rv))
            self.create_node("output", "output", (outputs,), {})
        else:
            unimplemented(f"RETURN_VALUE {type(rv).__name__}")
        ncalls = count_calls(self.graph)
        counters["stats"]["calls_captured"] += ncalls
        counters["stats"]["fusions_possible"] += ncalls - 1
        self.guards.update(rv.guards)
        gm = fx.GraphModule(FakeRootModule(self.nn_modules), self.graph)
        gm.recompile()
        name = unique_id("__translated_fn")
        self.f_globals[name] = self.compiler_fn(gm)
        self.code_options["co_names"] = tuple(self.code_options["co_names"]) + (name,)
        nargs = sum(map(len, self.graphargs))
        self.code_options["co_stacksize"] = max(
            self.code_options["co_stacksize"], 1 + nargs
        )
        return (
            [self.create_load_global(name)]
            + list(
                itertools.chain.from_iterable(arg.load(self) for arg in self.graphargs)
            )
            + [
                create_instruction("CALL_FUNCTION", nargs),
            ]
        )

    def compile_partial_subgraph(
        self, stack_values: List[VariableTracker], jump_to_user_code: List[Instruction]
    ):
        """
        Generate a subgraph to continue execution on user code.
        Automatically restore live variables.
        """
        # TODO(jansel): some dead code elimination code help this
        # TODO(jansel): add some liveness analysis to see if we need all these
        stack_values = list(stack_values)
        prefix = []
        var_names = []
        for k, v in self.symbolic_locals.items():
            if v.initial_name == k:
                continue  # no need to restore initial state
            elif isinstance(v, ConstantVariable):
                unimplemented("TODO: debug issue in pyhpc_equation_of...")
                # no need to get a constant from the compiled graph
                self.guards.update(v.guards)
                prefix.extend(
                    [self.create_load_const(v.value), self.create_store_fast(k)]
                )
            else:
                # must get the value from compiled graph
                var_names.append(k)

        # this should work, but need a testcase to make sure
        # while stack_values and isinstance(stack_values[0], ConstantVariable):
        #     # no need to get a constant from the compiled graph
        #     self.guards.update(stack_values[0].guards)
        #     prefix.append(self.create_load_const(stack_values.pop(0).value))

        if len(var_names) == 0 and len(stack_values) == 1:
            self.insert_instruction_prefix(
                prefix + self.compile_subgraph(stack_values[0]) + jump_to_user_code
            )
        elif len(var_names) == 1 and len(stack_values) == 0:
            self.insert_instruction_prefix(
                prefix
                + self.compile_subgraph(self.symbolic_locals[var_names[0]])
                + self.restore_locals(var_names, 0, unpack=False)
                + jump_to_user_code
            )
        else:
            self.insert_instruction_prefix(
                prefix
                + self.compile_subgraph(
                    [self.symbolic_locals[k] for k in var_names]
                    + list(reversed(stack_values))
                )
                + self.restore_locals(var_names, len(stack_values))
                + jump_to_user_code
            )

    def should_compile_partial_graph(self):
        return isinstance(self, InstructionTranslator) and count_calls(self.graph) > 0

    def insert_instruction_prefix(self, prefix: List[Instruction]):
        """
        We call this on the creation of a new compiled subgraph that is inserted
        before user code.

        Currently, this stops the analysis (we only support a prefix of user code).

        Later we should extend this to continue the analysis
        """
        self.instructions[:] = prefix + self.instructions
        # TODO(jansel): resume the analysis instead of exiting
        self.instruction_pointer = None  # exit analysi

    def restore_locals(self, var_names, extra, unpack=True):
        """
        Used by compile_partial_subgraph() to set local variables from
        the result of a self.compile_subgraph() call.
        """
        code = []
        if unpack:
            code.append(create_instruction("UNPACK_SEQUENCE", len(var_names) + extra))
        for name in var_names:
            code.append(self.create_store_fast(name))
        return code

    def copy_graphstate(self):
        """Create a checkpoint of the current state by copying everything"""
        graph_nodes = set(self.graph.nodes)
        guards = copy.deepcopy(self.guards)
        graphargs = copy.deepcopy(self.graphargs)
        symbolic_locals = dict(self.symbolic_locals)
        stack = list(self.stack)
        nn_modules = dict(self.nn_modules)
        return graph_nodes, graphargs, guards, symbolic_locals, stack, nn_modules

    def restore_graphstate(self, state):
        """Restore a checkpoint created by self.copy_graphstate()"""
        (
            graph_nodes,
            self.graphargs,
            self.guards,
            self.symbolic_locals,
            self.stack,
            self.nn_modules,
        ) = state
        for node in reversed(list(self.graph.nodes)):
            if node not in graph_nodes:
                self.graph.erase_node(node)

    def __init__(
        self,
        cnt: typing.Iterable,
        graph: fx.Graph,
        graphargs,
        nn_modules,
        guards,
        instructions: List[Instruction],
        f_globals,
        f_builtins,
        code_options,
        symbolic_locals=None,
    ):
        super(InstructionTranslatorBase, self).__init__()
        self.graph = graph
        self.graphargs = graphargs
        self.stack = []
        self.symbolic_locals = symbolic_locals
        self.guards = guards
        self.nn_modules = nn_modules

        self.instructions = instructions
        self.code_options = code_options
        self.instruction_pointer = 0
        self.indexof = {id(i): n for n, i in enumerate(instructions)}
        self.f_globals = f_globals
        self.f_builtins = f_builtins
        self.cnt = cnt
        self.checkpoint = None


class InstructionTranslator(InstructionTranslatorBase):
    def __init__(
        self,
        instructions: List[Instruction],
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
        )
        self.compiler_fn = compiler_fn
        self.symbolic_locals = {
            k: self.wrap_local(k, f_locals[k])
            for k in code_options["co_varnames"]
            if k in f_locals
        }
        self.mark_initial_state()

    def RETURN_VALUE(self, inst):
        rv = self.pop()
        self.instruction_pointer = None
        if count_calls(self.graph) == 0:
            unimplemented("no graph found")
        elif rv.state == TracingSupported.YES:
            self.instructions[:] = self.compile_subgraph(rv) + [
                create_instruction("RETURN_VALUE")
            ]
        else:
            unimplemented("not traceable")


class RecursiveInstructionTranslator(InstructionTranslatorBase):
    """Trace and inline a called method"""

    @staticmethod
    def inline_call(parent, func, args, kwargs):
        assert callable(func)
        if getattr(func, "__closure__", None) is not None:
            unimplemented("inline with  __closure__")
        if getattr(func, "__self__", None) is not None:
            unimplemented("inline with  __self__")
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        sub_locals = dict()
        sub_globals = func.__globals__
        for k, v in bound.arguments.items():
            if isinstance(v, VariableTracker):
                sub_locals[k] = v
            else:
                unimplemented(f"call user defined {v}")
        tracer = RecursiveInstructionTranslator(
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
        super(RecursiveInstructionTranslator, self).__init__(
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
        )
        self.symbolic_result = None

    def RETURN_VALUE(self, inst):
        self.symbolic_result = self.pop()
        self.instruction_pointer = None


class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""

    def __init__(self, nn_modules: dict):
        super(FakeRootModule, self).__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)


def count_calls(g: fx.Graph):
    c = 0
    for n in g.nodes:
        if "call" in n.op:
            c += 1
    return c


def identity(x):
    return x
