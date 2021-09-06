import collections
import dataclasses
import dis
import functools
import inspect
import itertools
import types
from typing import List

import torch
from torch import fx
from torch.fx import GraphModule

from .allowed_functions import is_allowed
from .bytecode_transformation import Instruction
from .bytecode_transformation import create_instruction
from .bytecode_transformation import debug_checks
from .bytecode_transformation import transform_code_object
from .bytecode_transformation import unique_id
from .guards import Guard, GuardedCode
from .guards import GuardRequirement
from .guards import GuardSource
from .variable_tracker import AllowedFunctionOrModuleVariable, GetAttrVariable, TupleVariable, ListVariable, \
    ConstDictVariable, SliceVariable
from .variable_tracker import ConstantVariable
from .variable_tracker import MethodNameVariable
from .variable_tracker import NNModuleVariable
from .variable_tracker import TensorVariable
from .variable_tracker import TracingSupported
from .variable_tracker import VariableTracker

DEBUG = False
counters = collections.Counter()


def unimplemented(name):
    counters[f"E:{name}"] += 1
    raise NotImplementedError(name)


def stack_op(fn):
    nargs = len(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def impl(self, inst):
        inputs = self.popn(nargs)

        cls, kwargs = VariableTracker.combine(inputs)
        if issubclass(cls, TensorVariable):
            val = cls(proxy=fn(*[i.as_proxy() for i in inputs]),
                      **kwargs)
        else:
            unimplemented("stack_op")

        self.push(val)

    return impl


@dataclasses.dataclass
class LocalArg:
    name: str

    def load(self, tracer):
        return tracer.create_load_fast(self.name)


@dataclasses.dataclass
class GlobalArg:
    name: str

    def load(self, tracer):
        return tracer.create_load_global(self.name)


class InstructionTracer(fx.Tracer):
    def __init__(self, instructions: List[Instruction], f_locals, f_globals, f_builtins, code_options):
        super(InstructionTracer, self).__init__()
        self.graph = fx.Graph()
        self.instructions = instructions
        self.stack = []
        self.f_globals = f_globals
        self.f_builtins = f_builtins
        self.indexof = {id(i): n for n, i in enumerate(instructions)}
        self.instruction_pointer = 0
        self.cnt = itertools.count()
        self.graphargs = []
        self.code_options = code_options
        self.nn_modules = {}
        self.guards = None

        self.symbolic_locals = {k: self.wrap_local(k, f_locals[k])
                                for k in code_options["co_varnames"]
                                if k in f_locals}
        if DEBUG:
            print("names     ", code_options["co_names"])
            print("varnames  ", code_options["co_varnames"])
            print("cellvars  ", code_options["co_cellvars"])
            print("freevars  ", code_options["co_freevars"])
            print("consts    ", code_options["co_consts"])
            print("stacksize ", code_options["co_stacksize"])
            print("argnames  ", self.graphargs)

    def create_load_fast(self, name):
        assert name in self.code_options["co_varnames"]
        return create_instruction("LOAD_FAST",
                                  self.code_options["co_varnames"].index(name),
                                  name)

    def create_load_global(self, name):
        assert name in self.code_options["co_names"]
        return create_instruction("LOAD_GLOBAL",
                                  self.code_options["co_names"].index(name),
                                  name)

    def wrap_local(self, name, value):
        if isinstance(value, torch.Tensor):
            self.graphargs.append(LocalArg(name))
            return TensorVariable(
                proxy=self.create_graph_input(name),
                state=TracingSupported.YES,
                guards={Guard(name, GuardSource.LOCAL, GuardRequirement.TYPE_MATCH)},
            )
        elif isinstance(value, torch.nn.Module):
            key = f"{name}_{next(self.cnt)}"
            self.nn_modules[key] = value
            return NNModuleVariable(
                key=key,
                state=TracingSupported.YES,
                guards={Guard(name, GuardSource.LOCAL, GuardRequirement.VALUE_MATCH)},
            )
        else:
            unimplemented("wrap_local")

    def create_graph_input(self, name):
        placeholders = [n for n in self.graph.nodes if n.op == "placeholder"]
        if placeholders:
            ctx = self.graph.inserting_after(placeholders[-1])
        else:
            ctx = self.graph.inserting_before(None)
        with ctx:
            return self.create_proxy('placeholder', f'{name}_{next(self.cnt)}', (), {})

    def call_function(self, fn, args, kwargs):
        if isinstance(fn, AllowedFunctionOrModuleVariable):
            _, options = VariableTracker.combine([fn, ] + list(args) + list(kwargs.values()))
            assert getattr(fn.value, "__self__", None) is None
            proxy_args = tuple(arg.as_proxy() for arg in args)
            proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
            self.push(TensorVariable(
                proxy=self.create_proxy('call_function', fn.value, proxy_args, proxy_kwargs),
                **options
            ))
        elif isinstance(fn, GetAttrVariable):
            name = fn.name
            obj = fn.obj
            args = [obj] + list(args)
            cls, options = VariableTracker.combine([fn, ] + list(args) + list(kwargs.values()))
            proxy_args = tuple(arg.as_proxy() for arg in args)
            proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
            self.push(cls(
                proxy=self.create_proxy('call_method', name, proxy_args, proxy_kwargs),
                **options
            ))
        else:
            unimplemented("call_function")

    def step(self):
        inst = self.instructions[self.instruction_pointer]
        self.instruction_pointer += 1
        if not hasattr(self, inst.opname):
            unimplemented(f"missing: {inst.opname}")
        getattr(self, inst.opname)(inst)
        return inst.opname != "RETURN_VALUE"

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
        self.push(ConstantVariable(value=inst.argval,
                                   state=TracingSupported.UNKNOWN))

    def LOAD_GLOBAL(self, inst):
        value = self.f_globals[inst.argval]
        if is_allowed(value):
            # TODO(jansel): if we wanted to be paranoid, we could generate a guard here
            self.push(AllowedFunctionOrModuleVariable(
                value=value,
                state=TracingSupported.YES,
                guards={Guard(inst.argval, GuardSource.GLOBAL, GuardRequirement.FUNCTION_MATCH)},
            ))
        elif isinstance(value, torch.Tensor):
            # turn a load of a global tensor into an arg for the graph
            self.graphargs.append(GlobalArg(inst.argval))
            self.push(TensorVariable(
                proxy=self.create_graph_input(inst.argval),
                state=TracingSupported.YES,
                guards={Guard(inst.argval, GuardSource.GLOBAL, GuardRequirement.TYPE_MATCH)},
            ))
        else:
            unimplemented("LOAD_GLOBAL")

    def CALL_FUNCTION(self, inst):
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})

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
        assert isinstance(argsvars, ListVariable)
        assert isinstance(kwargsvars, ConstDictVariable)
        self.call_function(fn, argsvars.items, kwargsvars.items)

    def CALL_FUNCTION_KW(self, inst):
        argnames = self.pop()
        args = self.popn(inst.argval)
        fn = self.pop()
        assert isinstance(argnames, ConstantVariable)
        argnames = argnames.value
        args, kwargs = args[:-len(argnames)], args[-len(argnames):]
        kwargs = dict(zip(argnames, kwargs))
        assert len(kwargs) == len(argnames)
        self.call_function(fn, args, kwargs)

    def CALL_METHOD(self, inst):
        args = self.popn(inst.argval)
        self_ptr = self.pop()
        fn = self.pop()
        if isinstance(fn, AllowedFunctionOrModuleVariable):
            assert self_ptr is None
            self.call_function(fn, args, {})
        elif isinstance(fn, MethodNameVariable):
            assert isinstance(self_ptr, TensorVariable)
            cls, options = VariableTracker.combine([fn, self_ptr] + args)
            proxy_args = tuple(x.as_proxy() for x in [self_ptr] + args)
            self.push(cls(
                proxy=self.create_proxy('call_method', fn.name, proxy_args, {}),
                **options
            ))
        elif isinstance(fn, NNModuleVariable):
            mod = self.get_submodule(fn.key)
            if is_allowed(mod.__class__):
                _, options = VariableTracker.combine([fn] + args)
                proxy_args = tuple(x.as_proxy() for x in args)
                self.push(TensorVariable(
                    proxy=self.create_proxy('call_module', fn.key, proxy_args, {}),
                    **options
                ))
            else:
                unimplemented("user defined module")
        else:
            unimplemented("CALL_METHOD")

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
        obj = self.pop()
        name = inst.argval
        _, options = VariableTracker.combine([obj])
        if isinstance(obj, AllowedFunctionOrModuleVariable):
            self.push(AllowedFunctionOrModuleVariable(
                value=getattr(obj.value, name),
                **options
            ))
            self.push(None)  # Null self ptr
        elif isinstance(obj, TensorVariable):
            self.push(MethodNameVariable(name=name,
                                         state=TracingSupported.UNKNOWN))
            self.push(obj)
        elif isinstance(obj, NNModuleVariable):
            key = f"{obj.key}.{name}"
            subobj = self.get_submodule(key)
            if isinstance(subobj, torch.nn.Module):
                self.push(NNModuleVariable(
                    key,
                    **options
                ))
                self.push(None)  # Null self ptr
            else:
                unimplemented("nn.Module method")
        else:
            unimplemented("LOAD_METHOD")

    def LOAD_ATTR(self, inst):
        obj = self.pop()
        name = inst.argval
        _, options = VariableTracker.combine([obj])
        if isinstance(obj, NNModuleVariable):
            key = f"{obj.key}.{name}"
            subobj = self.get_submodule(key)
            if isinstance(subobj, torch.Tensor):
                self.push(TensorVariable(
                    proxy=self.create_proxy("get_attr", key, tuple(), {}),
                    **options
                ))
            else:
                unimplemented("nn.Module attr")
        elif isinstance(obj, TensorVariable):
            self.push(GetAttrVariable(obj, name, **options))
        elif isinstance(obj, AllowedFunctionOrModuleVariable):
            self.push(AllowedFunctionOrModuleVariable(
                value=getattr(obj.value, name),
                **options
            ))
        else:
            unimplemented("LOAD_ATTR")

    def BUILD_TUPLE(self, inst):
        items = self.popn(inst.argval)
        _, options = VariableTracker.combine(items)
        self.push(TupleVariable(items, **options))

    def BUILD_SLICE(self, inst):
        items = self.popn(inst.argval)
        _, options = VariableTracker.combine(items)
        self.push(SliceVariable(items, **options))

    def BUILD_LIST(self, inst):
        items = self.popn(inst.argval)
        _, options = VariableTracker.combine(items)
        self.push(ListVariable(items, **options))

    def BUILD_MAP(self, inst):
        assert inst.argval == 0
        self.push(ConstDictVariable({}))

        # items = self.popn(inst.argval)
        # _, options = VariableTracker.combine(items)
        # self.push(ConstDictVariable(items, **options))

    def RETURN_VALUE(self, inst):
        rv = self.pop()
        if rv.state == TracingSupported.YES:
            self.create_node('output', 'output', (self.create_arg(rv.proxy),), {})
            ncalls = count_calls(self.graph)
            counters["calls_captured"] += ncalls
            counters["fusions_possible"] += ncalls - 1
            DEBUG and self.graph.print_tabular()
            self.guards = rv.guards
            gm = GraphModule(FakeRootModule(self.nn_modules), self.graph)
            gm.recompile()
            name = unique_id("__translated_fn")
            self.f_globals[name] = gm.forward
            self.code_options["co_names"] = tuple(self.code_options["co_names"]) + (name,)
            self.code_options["co_stacksize"] = len(self.graphargs) + 1
            self.instructions[:] = (
                    [self.create_load_global(name)] +
                    [arg.load(self) for arg in self.graphargs] +
                    [create_instruction("CALL_FUNCTION", len(self.graphargs)),
                     create_instruction("RETURN_VALUE")]
            )
        else:
            unimplemented("not traceable")

    BINARY_POWER = stack_op(lambda tos1, tos: tos1 ** tos)
    BINARY_MULTIPLY = stack_op(lambda tos1, tos: tos1 * tos)
    BINARY_MATRIX_MULTIPLY = stack_op(lambda tos1, tos: tos1 @ tos)
    BINARY_FLOOR_DIVIDE = stack_op(lambda tos1, tos: tos1 // tos)
    BINARY_TRUE_DIVIDE = stack_op(lambda tos1, tos: tos1 / tos)
    BINARY_MODULO = stack_op(lambda tos1, tos: tos1 % tos)
    BINARY_ADD = stack_op(lambda tos1, tos: tos1 + tos)
    BINARY_SUBTRACT = stack_op(lambda tos1, tos: tos1 - tos)
    BINARY_SUBSCR = stack_op(lambda tos1, tos: tos1[tos])
    BINARY_LSHIFT = stack_op(lambda tos1, tos: tos1 << tos)
    BINARY_RSHIFT = stack_op(lambda tos1, tos: tos1 >> tos)
    BINARY_AND = stack_op(lambda tos1, tos: tos1 & tos)
    BINARY_XOR = stack_op(lambda tos1, tos: tos1 ^ tos)
    BINARY_OR = stack_op(lambda tos1, tos: tos1 | tos)


class FakeRootModule(torch.nn.Module):
    """ Trick the constructor of fx.GraphModule """

    def __init__(self, nn_modules: dict):
        super(FakeRootModule, self).__init__()
        training = None
        for k, v in nn_modules.items():
            setattr(self, k, v)
            training2 = getattr(v, "training", None)
            assert None in (training, training2) or training == training2
            if training2 is not None:
                training = training2


def count_calls(g: fx.Graph):
    c = 0
    for n in g.nodes:
        if "call" in n.op:
            c += 1
    return c


def convert_frame_assert(frame: types.FrameType):
    code = frame.f_code
    if code.co_filename.startswith("<eval_with_key>"):
        return GuardedCode(code)  # skip FX output
    # TODO(jansel): detect and skip other types of generated code
    debug_checks(code)
    guards = None

    def transform(instructions, code_options):
        nonlocal guards
        tracer = InstructionTracer(instructions,
                                   frame.f_locals,
                                   frame.f_globals,
                                   frame.f_builtins,
                                   code_options)
        tracer.run()
        guards = tracer.guards

    code = transform_code_object(frame.f_code, transform)
    if DEBUG:
        print("ORIGINAL")
        print(dis.Bytecode(code).info())
        print(dis.Bytecode(code).dis())
        print("NEW CODE")
        print(dis.Bytecode(code).info())
        print(dis.Bytecode(code).dis())
    assert guards is not None
    return GuardedCode(code, guards)


def convert_frame(frame: types.FrameType):
    counters["F:total"] += 1
    try:
        result = convert_frame_assert(frame)
        counters["F:ok"] += 1
        return result
    except NotImplementedError:
        pass
    except Exception as e:
        counters[f"E:{e.__class__.__name__}:{e}"] += 1
    return GuardedCode(frame.f_code)
