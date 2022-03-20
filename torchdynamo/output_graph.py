import collections
import copy
import itertools
import operator
import re
import sys
import traceback
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import torch.nn
from torch import fx

import torchdynamo

from . import config
from . import variables
from .bytecode_transformation import Instruction
from .bytecode_transformation import create_instruction
from .bytecode_transformation import unique_id
from .codegen import PyCodegen
from .exc import unimplemented
from .guards import GuardBuilder
from .mutation_guard import is_dynamic_nn_module
from .side_effects import SideEffects
from .source import LocalSource
from .source import Source
from .utils import CleanupHook
from .utils import count_calls
from .utils import counters
from .variables.nn_module import NNModuleVariable
from .variables.tensor import TensorVariable


class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""

    def __init__(self, nn_modules: dict):
        super(FakeRootModule, self).__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)

    def __repr__(self):
        return "FakeRootModule(...)"


class OutputGraph(fx.Tracer):
    """
    Wrapper class to hold outputs of InstructionTranslator.  Mainly the
    generated fx.Graph.
    """

    def __init__(
        self,
        f_globals: Dict[str, Any],
        code_options: Dict[str, Any],
        compiler_fn: Callable,
        root_tx: "torchdynamo.symbolic_convert.InstructionTranslator",
    ):
        super(OutputGraph, self).__init__()

        # Mutable state checkpointed by copy_graphstate()
        self.graph = torch.fx.Graph()
        self.graphargs = []
        self.guards = set()
        self.nn_modules = dict()
        self.side_effects = SideEffects()
        self.code_options = dict(code_options)
        self.output_instructions = []

        # Not checkpointed
        self.compiler_fn = compiler_fn
        self.root_globals = f_globals
        self.root_tx = root_tx
        self.cleanups = []
        self.should_exit = False

    @property
    def output(self):
        return self

    def copy_graphstate(self):
        """Create a checkpoint of the current state by copying everything"""
        graph_nodes = set(self.graph.nodes)
        return (
            graph_nodes,
            list(self.graphargs),
            copy.deepcopy(self.guards),
            dict(self.nn_modules),
            self.side_effects.clone(),
        )

    def restore_graphstate(self, state):
        """Restore a checkpoint created by self.copy_graphstate()"""
        (
            graph_nodes,
            self.graphargs,
            self.guards,
            self.nn_modules,
            self.side_effects,
        ) = state
        # FX deepcopy doesn't work for a partially created graph, so just remove new nodes
        for node in reversed(list(self.graph.nodes)):
            if node not in graph_nodes:
                self.graph.erase_node(node)

    def count_calls(self):
        return count_calls(self.graph)

    def get_submodule(self, keys):
        assert keys
        obj = self.nn_modules
        for k in keys.split("."):
            if isinstance(obj, dict):
                obj = obj[k]
            else:
                obj = getattr(obj, k)
        return obj

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

    def new_var(self, name="tmp"):
        existing = set(self.code_options["co_varnames"])
        for i in itertools.count():
            var = f"___{name}_{i}"
            if var not in existing:
                self.code_options["co_varnames"] = self.code_options["co_varnames"] + (
                    var,
                )
                return var

    def update_co_names(self, name):
        """Ensure self.code_options.co_names contains name"""
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] = tuple(self.code_options["co_names"]) + (
                name,
            )

    def add_submodule(self, mod: torch.nn.Module, *names, **options):
        if is_dynamic_nn_module(mod):
            return variables.UnspecializedNNModuleVariable(mod, **options)

        options = dict(options)
        options["guards"] = set(options.get("guards", []))
        source: Source = options["source"]
        if isinstance(mod, torch.Tensor):
            options["guards"].add(source.create_guard(GuardBuilder.TENSOR_MATCH))

            def wrap_name(module_key):
                return TensorVariable.create(
                    self,
                    self.create_proxy("get_attr", module_key, tuple(), {}),
                    example_value=mod,
                    **options,
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

    def compile_subgraph(self, tx, partial_convert=False):
        """
        Generate a subgraph to continue execution on user code.
        Automatically restore live variables.
        """
        self.partial_convert = partial_convert

        if not all(block.can_restore() for block in tx.block_stack):
            unimplemented("compile_subgraph with block_depth != 0")

        for block in reversed(tx.block_stack):
            block.exit(tx)

        tx.prune_dead_locals()
        stack_values = list(tx.stack)
        root = FakeRootModule(self.nn_modules)

        # Add all the local vars to the "stack" so restore at the end
        restore_vars = []
        val_to_names = collections.OrderedDict()
        if stack_values:
            val_to_names[stack_values[-1]] = list()
        for k, v in tx.symbolic_locals.items():
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
            and self.side_effects.is_empty()
        ):
            # optimization to generate better code in a common case
            self.add_output_instructions(
                self.compile_and_call_fx_graph(tx, list(reversed(stack_values)), root)
                + [create_instruction("UNPACK_SEQUENCE", len(stack_values))]
            )
        else:
            graph_output_var = self.new_var("graph_out")
            pass1 = PyCodegen(tx, root, graph_output_var)
            self.side_effects.codegen(pass1)
            pass1.foreach(stack_values)

            # one more time now that we have established tempvars
            pass2 = PyCodegen(
                tx,
                root,
                graph_output_var,
                tempvars={val: None for val, count in pass1.uses.items() if count > 1},
            )
            self.side_effects.codegen(pass2)
            pass2.foreach(stack_values)

            output = []
            if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
                output.extend(
                    self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
                )

                if len(pass2.graph_outputs) != 0:
                    output.append(pass2.create_store(graph_output_var))
                else:
                    output.append(create_instruction("POP_TOP"))
            self.add_output_instructions(output + pass2.get_instructions())

        # restore all the live local vars
        self.add_output_instructions(
            [PyCodegen(tx).create_store(var) for var in reversed(restore_vars)]
        )

    def compile_and_call_fx_graph(self, tx, rv, root):
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

        if config.dynamic_propagation:
            # free a bit of memory
            for node in self.graph.nodes:
                if "example_value" in node.meta:
                    del node.meta["example_value"]

        gm = fx.GraphModule(root, self.graph)
        gm.recompile()
        name = unique_id("__compiled_fn")
        compiled_fn = self.call_user_compiler(gm)
        compiled_fn = torchdynamo.disable(compiled_fn)
        counters["stats"]["unique_graphs"] += 1
        self.install_global(name, compiled_fn)
        if config.debug:
            print(f"\n{name} {gm.forward.__code__.co_filename}")
            self.graph.print_tabular()

        cg = PyCodegen(tx)
        cg.make_call_generated_code(name)
        return cg.get_instructions()

    def call_user_compiler(self, gm):
        try:
            compiled_fn = self.compiler_fn(gm, self.example_inputs())
            assert callable(compiled_fn), "compiler_fn did not return callable"
        except Exception:
            sys.stderr.write("-" * 40 + "\n")
            sys.stderr.write("TORCHDYNAMO: backend compiler failed\n")
            traceback.print_exc()
            sys.stderr.write("-" * 40 + "\n")
            compiled_fn = gm.forward
        return compiled_fn

    def example_inputs(self):
        result = []
        for arg in self.graphargs:
            result.extend(arg.get_examples())
        return result

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

    def add_output_instructions(self, prefix: List[Instruction]):
        """
        We call this on the creation of a new compiled subgraph that is inserted
        before user code.
        """
        self.output_instructions.extend(prefix)
        self.should_exit = True

    def install_global(self, name, value):
        self.cleanups.append(CleanupHook.create(self.root_globals, name, value))
