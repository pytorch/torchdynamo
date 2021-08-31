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

from .bytecode_transformation import debug_checks, transform_code_object, Instruction, create_instruction

TORCH_OBJECT_IDS = {id(torch)}


def _find_torch_objects(module):
    print(module.__name__)
    for name, obj in list(module.__dict__.items()):
        if id(obj) not in TORCH_OBJECT_IDS and not name.startswith("_"):
            if isinstance(obj, types.ModuleType):
                if obj.__name__.startswith("torch."):
                    TORCH_OBJECT_IDS.add(id(obj))
                    _find_torch_objects(obj)
            else:
                TORCH_OBJECT_IDS.add(id(obj))


_find_torch_objects(torch)

_unique_id_counter = itertools.count()


def unique_id(name):
    return f"{name}_{next(_unique_id_counter)}"


class TracingSupported(enum.Enum):
    UNKNOWN = 0
    YES = 1
    NO = 2


def combine_state(a, b):
    return TracingSupported(max(a.value, b.value))


combine_states = functools.partial(functools.reduce, combine_state)


class VariableTracker:
    """ Base class for tracked locals and stack values """

    @staticmethod
    def combine(vars):
        vars = list(vars)
        priority = [TensorVariable, TorchVariable, ConstantVariable]
        vars.sort(key=lambda v: priority.index(type(v)))
        return type(vars[0]), {"state": combine_states(v.state for v in vars)}

    def __init__(self, state):
        super(VariableTracker, self).__init__()
        self.state = state


class TensorVariable(VariableTracker):
    """ Points to a tensor """

    def __init__(self, proxy, **kwargs):
        super(TensorVariable, self).__init__(**kwargs)
        self.proxy = proxy

    def as_proxy(self):
        return self.proxy


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
        self.cnt = -1
        self.graphargs = []
        self.symbolic_locals = {k: self.wrap_local(k, f_locals[k])
                                for k in code_options["co_varnames"]
                                if k in f_locals}
        self.code_options = code_options

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
                state=TracingSupported.YES
            )
        assert False

    def create_graph_input(self, name):
        self.cnt += 1
        placeholders = [n for n in self.graph.nodes if n.op == "placeholder"]
        if placeholders:
            ctx = self.graph.inserting_after(placeholders[-1])
        else:
            ctx = self.graph.inserting_before(None)
        with ctx:
            return self.create_proxy('placeholder', f'{name}_{str(self.cnt)}', (), {})

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
        if isinstance(value, torch.Tensor):
            self.graphargs.append(GlobalArg(inst.argval))
            self.push(TensorVariable(
                proxy=self.create_graph_input(inst.argval),
                state=TracingSupported.YES
            ))


    def RETURN_VALUE(self, inst):
        rv = self.pop()
        if rv.state == TracingSupported.YES:
            self.create_node('output', 'output', (self.create_arg(rv.proxy),), {})
            self.graph.print_tabular()
            gm = GraphModule(dict(), self.graph)
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


"""
        def trace(self, root: Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
            if isinstance(root, torch.nn.Module):
                self.root = root
                fn = type(root).forward
                self.submodule_paths = {mod: name for name, mod in root.named_modules()}
            else:
                self.root = torch.nn.Module()
                fn = root
            self.graph = Graph()
    
            # When we encounter a Tensor value that's not a parameter, we look if it
            # is some other attribute on the model. Construct a dict mapping Tensor
            # values to the qualified name here for efficiency. This is used downstream
            # in create_arg
            self.tensor_attrs: Dict[torch.Tensor, str] = {}
    
            def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
                for k, v in m.__dict__.items():
                    if isinstance(v, (torch.Tensor, ScriptObject)):
                        self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
                for k, v in m.named_children():
                    collect_tensor_attrs(v, prefix_atoms + [k])
    
            collect_tensor_attrs(self.root, [])
    
            assert isinstance(fn, FunctionType)
    
            fn_globals = fn.__globals__  # run before it gets patched
            fn, args = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)
    
            parameter_proxy_cache: Dict[str, Proxy] = {}  # Reduce number of get_attr calls
    
            # Method dispatch on parameters is not recorded unless it's directly used.
            # Thus, we need to insert a proxy when __getattr__ requests a parameter.
            @functools.wraps(_orig_module_getattr)
            def module_getattr_wrapper(mod, attr):
                attr_val = _orig_module_getattr(mod, attr)
                return self._module_getattr(attr, attr_val, parameter_proxy_cache)
    
            @functools.wraps(_orig_module_call)
            def module_call_wrapper(mod, *args, **kwargs):
                def forward(*args, **kwargs):
                    return _orig_module_call(mod, *args, **kwargs)
    
                _autowrap_check(patcher, getattr(getattr(mod, "forward", mod), "__globals__", {}),
                                self._autowrap_function_ids)
                return self.call_module(mod, forward, args, kwargs)
    
            with _CPatchManager(self):
                with _Patcher() as patcher:
                    # allow duplicate patches to support the case of nested calls
                    patcher.patch_method(torch.nn.Module, "__getattr__", module_getattr_wrapper, deduplicate=False)
                    patcher.patch_method(torch.nn.Module, "__call__", module_call_wrapper, deduplicate=False)
                    _patch_wrapped_functions(patcher)
                    _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
                    for module in self._autowrap_search:
                        _autowrap_check(patcher, module.__dict__, self._autowrap_function_ids)
                    self.create_node('output', 'output', (self.create_arg(fn(*args)),), {},
                                     type_expr=fn.__annotations__.get('return', None))
    
            self.submodule_paths = None
            return self.graph
            
        class BoxedValue:
            def __init__(self, value):
                self.value = value
        
        
        def stack_op(fn):
            nargs = len(inspect.signature(fn).parameters)
        
            @functools.wraps(fn)
            def impl(self, val):
                self.stack.append(fn(*self.popn(nargs)))
        
            return impl
        
        
        def load_op(fn):
            @functools.wraps(fn)
            def impl(self, val):
                self.stack.append(fn(self, val))
        
            return impl
        
        
        class FrameInterpreter:
            def LOAD_FAST(self, val):
                self.stack.append(self.f_locals[val].value)
        
            def LOAD_CLOSURE(self, val):
                self.stack.append(self.f_locals[val].value)
        
            def LOAD_FAST(self, val):
                self.stack.append(self.f_locals[val].value)
        
            def STORE_FAST(self, val):
                self.f_locals[val] = BoxedValue(self.stack.pop())
        
            def LOAD_GLOBAL(self, val):
                self.stack.append(self.f_globals[val])
        
            def STORE_DEREF(self, val):
                self.f_locals[val] = BoxedValue(CellType(self.stack.pop()))
        
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
            LOAD_CONST = load_op(lambda self, val: val)
"""


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
