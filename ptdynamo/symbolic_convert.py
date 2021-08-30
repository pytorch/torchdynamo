import dataclasses
import dis
import types
from typing import Union, Callable, Optional, Dict, Any, List
import enum
import functools
import inspect
import torch
from torch.fx import Graph
from torch.fx import GraphModule
from torch import fx

from .bytecode_transformation import debug_checks, transform_code_object, Instruction, create_instruction

"""
def symbolic_trace(root: Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]] = None,
                   enable_cpatching: bool = False) -> GraphModule:
    tracer = MyTracer(enable_cpatching=enable_cpatching)
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)

# out = self.create_proxy('placeholder', f'{name}_{str(cnt)}', (), {})
"""


class TracingSupported(enum.Enum):
    UNKNOWN = 0
    YES = 1
    NO = 2


def combine_state(a, b):
    return TracingSupported(max(a.value, b.value))


combine_states = functools.partial(functools.reduce, combine_state)


@dataclasses.dataclass
class VariableTracker:
    proxy: Optional[fx.Proxy]
    state: TracingSupported


def stack_op(fn):
    nargs = len(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def impl(self, inst):
        inputs = self.popn(nargs)
        state = combine_states(i.state for i in inputs)
        if state == TracingSupported.YES:
            proxy = fn(*[i.proxy for i in inputs])
        else:
            proxy = None
        self.push(VariableTracker(
            proxy=proxy,
            state=state
        ))

    return impl


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
        self.argnames = []
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
        print("argnames  ", self.argnames)

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
            self.cnt += 1
            self.argnames.append(name)
            return VariableTracker(
                proxy=self.create_proxy('placeholder', f'{name}_{str(self.cnt)}', (), {}),
                state=TracingSupported.YES
            )
        return VariableTracker(
            proxy=None,
            state=TracingSupported.NO
        )

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

    def RETURN_VALUE(self, inst):
        rv = self.pop()
        if rv.state == TracingSupported.YES:
            self.create_node('output', 'output', (self.create_arg(rv.proxy),), {})
            self.graph.print_tabular()
            gm = GraphModule(dict(), self.graph)
            gm.recompile()
            name = "__ptdynamo_fn__"
            self.f_globals[name] = gm.forward
            self.code_options["co_names"] = tuple(self.code_options["co_names"]) + (name,)
            self.code_options["co_stacksize"] = len(self.argnames) + 1
            self.instructions[:] = (
                    [self.create_load_global(name)] +
                    list(map(self.create_load_fast, self.argnames)) +
                    [create_instruction("CALL_FUNCTION", len(self.argnames)),
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
    if code.co_name == "forward":
        print("fname", code.co_filename, code.co_firstlineno)
        return code
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
