import dataclasses
import dis
import enum
import functools
import inspect
import itertools
import types
from typing import List

import torch
from torch import fx
from torch.fx import GraphModule

from .bytecode_transformation import debug_checks, transform_code_object, Instruction, create_instruction, unique_id

TORCH_OBJECT_IDS = set()


def _find_torch_objects(module):
    TORCH_OBJECT_IDS.add(id(module))
    for name, obj in list(module.__dict__.items()):
        if id(obj) not in TORCH_OBJECT_IDS:
            if isinstance(obj, types.ModuleType):
                if obj.__name__.startswith("torch."):
                    TORCH_OBJECT_IDS.add(id(obj))
                    _find_torch_objects(obj)
            else:
                TORCH_OBJECT_IDS.add(id(obj))


_find_torch_objects(torch)


class TracingSupported(enum.Enum):
    UNKNOWN = 0
    YES = 1
    NO = 2


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1


class GuardRequirement(enum.Enum):
    TYPE_MATCH = 0
    VALUE_MATCH = 1
    FUNCTION_MATCH = 2  # e.q. "from torch import add"


@dataclasses.dataclass
class Guard:
    name: str
    source: GuardSource
    requirement: GuardRequirement

    def __hash__(self):
        return hash((self.name, self.source, self.requirement))


def combine_state(a, b):
    return TracingSupported(max(a.value, b.value))


combine_states = functools.partial(functools.reduce, combine_state)
combine_guards = functools.partial(functools.reduce, set.union)


class VariableTracker:
    """ Base class for tracked locals and stack values """

    @staticmethod
    def combine(vars):
        vars = list(vars)
        priority = [TensorVariable, TorchVariable, NNModuleVariable, ConstantVariable, MethodNameVariable]
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


class TorchVariable(VariableTracker):
    """ Points to a module or method in torch.* """

    def __init__(self, value, **kwargs):
        super(TorchVariable, self).__init__(**kwargs)
        self.value = value


class MethodNameVariable(VariableTracker):
    def __init__(self, name, **kwargs):
        super(MethodNameVariable, self).__init__(**kwargs)
        self.name = name


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
            assert False

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

        self.symbolic_locals = {k: self.wrap_local(k, f_locals[k])
                                for k in code_options["co_varnames"]
                                if k in f_locals}

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
            assert False

    def create_graph_input(self, name):
        placeholders = [n for n in self.graph.nodes if n.op == "placeholder"]
        if placeholders:
            ctx = self.graph.inserting_after(placeholders[-1])
        else:
            ctx = self.graph.inserting_before(None)
        with ctx:
            return self.create_proxy('placeholder', f'{name}_{next(self.cnt)}', (), {})

    def call_function(self, fn, args, kwargs):
        if isinstance(fn, TorchVariable):
            assert getattr(fn.value, "__self__", None) is None
            cls, options = VariableTracker.combine([fn, ] + list(args) + list(kwargs.values()))
            proxy_args = tuple(arg.as_proxy() for arg in args)
            proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
            self.push(cls(
                proxy=self.create_proxy('call_function', fn.value, proxy_args, proxy_kwargs),
                **options
            ))
        else:
            assert False

    def step(self):
        inst = self.instructions[self.instruction_pointer]
        self.instruction_pointer += 1
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

    def LOAD_CONST(self, inst):
        self.push(ConstantVariable(value=inst.argval,
                                   state=TracingSupported.UNKNOWN))

    def LOAD_GLOBAL(self, inst):
        value = self.f_globals[inst.argval]
        if id(value) in TORCH_OBJECT_IDS:
            # TODO(jansel): if we wanted to be paranoid, we could generate a guard here
            self.push(TorchVariable(
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
            assert False

    def CALL_FUNCTION(self, inst):
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})

    def CALL_METHOD(self, inst):
        args = self.popn(inst.argval)
        self_ptr = self.pop()
        fn = self.pop()
        if isinstance(fn, TorchVariable):
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
            _, options = VariableTracker.combine([fn] + args)
            proxy_args = tuple(x.as_proxy() for x in args)
            self.push(TensorVariable(
                proxy=self.create_proxy('call_module', fn.key, proxy_args, {}),
                **options
            ))
        else:
            assert False

    def get_submodule(self, keys):
        print(keys)
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
        if isinstance(obj, TorchVariable):
            self.push(TorchVariable(
                value=getattr(obj.value, name),
                **options
            ))
            self.push(None)  # Null self ptr
        elif isinstance(obj, TensorVariable):
            self.push(MethodNameVariable(name=name,
                                         state=TracingSupported.UNKNOWN))
            self.push(obj)
        elif isinstance(obj, NNModuleVariable):
            key = f"{obj.key}.{inst.argval}"
            subobj = self.get_submodule(key)
            if isinstance(subobj, torch.nn.Module):
                self.push(NNModuleVariable(
                    key,
                    **options
                ))
                self.push(None)  # Null self ptr
            else:
                assert False
        else:
            assert False

    def RETURN_VALUE(self, inst):
        rv = self.pop()
        if rv.state == TracingSupported.YES:
            self.create_node('output', 'output', (self.create_arg(rv.proxy),), {})
            self.graph.print_tabular()
            print(rv.guards)
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
            assert False

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


def convert_frame_assert(frame: types.FrameType):
    code = frame.f_code
    if code.co_filename.startswith("<eval_with_key>"):
        return code  # skip FX output
    # TODO(jansel): detect and skip other types of generated code
    debug_checks(code)
    print("ORIGINAL")
    print(dis.Bytecode(code).info())
    print(dis.Bytecode(code).dis())

    def transform(instructions, code_options):
        tracer = InstructionTracer(instructions,
                                   frame.f_locals,
                                   frame.f_globals,
                                   frame.f_builtins,
                                   code_options)
        tracer.run()

    code = transform_code_object(frame.f_code, transform)
    print("NEW CODE")
    print(dis.Bytecode(code).info())
    print(dis.Bytecode(code).dis())
    return code
