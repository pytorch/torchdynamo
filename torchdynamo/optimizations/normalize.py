import builtins
import dataclasses
import itertools
import math
import operator

import torch
from torch.fx.operator_schemas import get_signature_for_torch_op

from torchdynamo import symbolic_convert
from torchdynamo.allowed_functions import _allowed_function_ids

VIEW_OPS = {
    # list taken from https://pytorch.org/docs/stable/tensor_view.html
    "getitem",
    "as_strided",
    "detach",
    "diagonal",
    "expand",
    "expand_as",
    "movedim",
    "narrow",
    "permute",
    "select",
    "squeeze",
    "transpose",
    "t",
    "T",
    "real",
    "imag",
    "view_as_real",
    "view_as_imag",
    "unflatten",
    "unfold",
    "unsqueeze",
    "view",
    "view_as",
    "unbind",
    "split",
    "split_with_sizes",
    "swapaxes",
    "swapdims",
    "chunk",
    "indices",
    "values",
}
MAYBE_VIEW_OPS = {"contiguous", "reshape"}
NORMALIZE_METHODS = {
    # 'size'
    "transpose": torch.transpose,
    "mean": torch.mean,
    "sigmoid": torch.sigmoid,
    "pow": torch.pow,
    "sum": torch.sum,
    "softmax": torch.nn.functional.softmax,
    "unsqueeze": torch.unsqueeze,
    "chunk": torch.chunk,
    "tril": torch.tril,
    "std": torch.std,
    "flatten": torch.flatten,
    "clone": torch.clone,
    "flip": torch.flip,
    "mul_": operator.imul,
    "add_": operator.iadd,
}
DONT_EXPAND_MODULES = {
    # This have internal control flow
    "EmbeddingBag",
    "LSTM",
    "BatchNorm2d",
    "InstanceNorm2d",
    "ConvTranspose2d",
}
FUNCTION_REPLACEMENTS = {
    torch.nn.functional.sigmoid: torch.sigmoid,
    torch.nn.functional.tanh: torch.tanh,
}

F = torch.nn.functional
INPLACE_OPS = {
    F.mish,
    F.silu,
    F.hardsigmoid,
    F.rrelu,
    F.leaky_relu,
    F.celu,
    F.selu,
    F.elu,
    F.relu6,
    F.hardswish,
    F.hardtanh,
    F.relu,
    F.threshold,
}

SKIP_INPLACE = {
    v
    for v in itertools.chain(
        math.__dict__.values(), builtins.__dict__.values(), operator.__dict__.values()
    )
    if callable(v)
}


class InliningTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return False


def expand_module_call(prefix, graph: torch.fx.Graph, module, args, kwargs):
    try:
        assert not kwargs
        arg_index = itertools.count()
        vars = dict()
        for node in InliningTracer().trace(module).nodes:
            if node.op == "placeholder":
                vars[node] = args[next(arg_index)]
            elif node.op == "output":
                assert len(node.args) == 1
                return vars[node.args[0]]
            elif node.op == "get_attr":
                vars[node] = graph.get_attr(f"{prefix}{node.target}")
            else:
                vars[node] = graph.node_copy(node, vars.__getitem__)
        assert False
    except Exception:
        print(f"Error while expanding {module.__class__.__name__}")
        raise


@dataclasses.dataclass
class NodeCounts:
    usages: int = 0


def short_name(gm, node: torch.fx.Node):
    if node.op == "call_function":
        return node.target.__name__
    elif node.op == "call_method":
        return node.target
    elif node.op == "call_module":
        return gm.get_submodule(node.target).__class__.__name__
    elif node.op == "get_attr":
        return node.target
    assert False


def long_name(gm, node: torch.fx.Node):
    name = short_name(gm, node)
    target = node.target
    if node.op == "call_function":
        try:
            return _allowed_function_ids()[id(node.target)]
        except KeyError:
            return f"{getattr(target, '__module__', '')}.{name}"
    elif node.op == "call_method":
        return name
    elif node.op == "call_module":
        target = gm.get_submodule(target).__class__
        return f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', '')}"
    elif node.op == "get_attr":
        return name
    assert False


class Inplacifier:
    def __init__(self, gm: torch.fx.GraphModule):
        self.gm = gm

    def can_be_view(self, node):
        name = short_name(self.gm, node)
        return name in VIEW_OPS or name in MAYBE_VIEW_OPS

    def inplacify(self):
        counts = dict()

        def record_usage(node):
            counts[node].usages += 1
            return node

        for node in self.gm.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                if self.can_be_view(node):
                    # Aliasing
                    counts[node] = counts[node.args[0]]
                elif "out" in node.kwargs:
                    counts[node] = counts[node.kwargs["out"]]
                else:
                    counts[node] = NodeCounts(0)
            else:
                counts[node] = NodeCounts(float("inf"))

        for node in reversed(list(self.gm.graph.nodes)):
            kwargs = dict(node.kwargs)
            if "inplace" in kwargs:
                kwargs.pop("inplace")
            if node.op == "call_function" and len(node.args) + len(kwargs) == 1:
                arg = node.args[0] if node.args else next(kwargs.values())
                if isinstance(arg, torch.fx.Node) and counts[arg].usages == 0:
                    if node.target in SKIP_INPLACE:
                        continue
                    elif node.target in INPLACE_OPS:
                        kwargs["inplace"] = True
                        symbolic_convert.counters["optimizations"]["inplace"] += 1
                    elif " out: torch.Tensor" in repr(
                        get_signature_for_torch_op(node.target)
                    ):
                        kwargs["out"] = arg
                        symbolic_convert.counters["optimizations"]["out"] += 1
                    else:
                        continue
                    with self.gm.graph.inserting_before(node):
                        node.replace_all_uses_with(
                            self.gm.graph.call_function(node.target, node.args, kwargs)
                        )
                    self.gm.graph.erase_node(node)

            torch.fx.map_arg((node.args, node.kwargs), record_usage)


def normalize(gm: torch.fx.GraphModule):
    # gm.graph.print_tabular()
    graph: torch.fx.Graph = gm.graph

    for node in list(graph.nodes):
        with graph.inserting_before(node):
            if node.op == "call_method" and node.target in NORMALIZE_METHODS:
                node.replace_all_uses_with(
                    graph.call_function(
                        NORMALIZE_METHODS[node.target], node.args, node.kwargs
                    )
                )
                graph.erase_node(node)
            elif node.op == "call_module":
                submod = gm.get_submodule(node.target)
                if submod.__class__.__name__ not in DONT_EXPAND_MODULES:
                    node.replace_all_uses_with(
                        expand_module_call(
                            f"{node.target}.", graph, submod, node.args, node.kwargs
                        )
                    )
                    graph.erase_node(node)

    # gm.graph.print_tabular()
