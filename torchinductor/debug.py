import collections
from typing import Any
from typing import List

import torch
from torch import fx as fx

from torchinductor import ir
from torchinductor.scheduler import BaseSchedulerNode
from torchinductor.scheduler import ExternKernelSchedulerNode
from torchinductor.scheduler import FusedSchedulerNode
from torchinductor.scheduler import NopKernelSchedulerNode
from torchinductor.scheduler import OutputNode
from torchinductor.scheduler import SchedulerNode
from torchinductor.scheduler import TemplateSchedulerNode


def draw_buffers(nodes, print_graph=False):
    """
    Draw a graph in fname.svg.
    nodes is a list of SchedulerNode objects.
    """

    from functorch.compile import draw_graph
    from functorch.compile import get_graph_being_compiled
    from torch.fx.graph_module import GraphModule
    from torch.fx.passes.shape_prop import TensorMetadata
    from torch.fx.passes.tools_common import legalize_graph

    fname = get_graph_being_compiled()
    graph = create_fx_from_snodes(nodes)

    for node in graph.nodes:
        if "fusion_meta" not in node.meta:
            continue
        group = node.meta["fusion_meta"].group
        if isinstance(group, tuple):
            group = group[1]

        # gather meta data
        dtype = None
        if isinstance(node, ir.ComputedBuffer):
            dtype = node.data.dtype

        metadata = TensorMetadata(group, dtype, None, None, None, None, None)
        node.meta["tensor_meta"] = metadata

    if print_graph:
        print(graph)
    print("starting creating module")
    gm = GraphModule({}, graph)
    legalize_graph(gm)
    gm.graph.lint()
    print("starting drawing")
    draw_graph(gm, fname, clear_meta=False)


def create_fx_from_snodes(snodes: List[BaseSchedulerNode]) -> fx.Graph:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """

    def get_fake_func(name):
        def func1(*args):
            return 0

        func1.__name__ = name
        return func1

    FusionMeta = collections.namedtuple("FusionMeta", ["group", "snodes", "type"])

    func_dict = {s: get_fake_func(s) for s in ["extern", "nop", "compute", "fused"]}
    buf_to_fx_node = {}
    graph = torch.fx.Graph()
    first_node = None

    outputs = []
    group: Any = None
    # create call_function node for each Buffer and Kernel
    for snode in snodes:
        if isinstance(snode, ExternKernelSchedulerNode):
            node_type = "extern"
            group = node_type
        elif isinstance(snode, TemplateSchedulerNode):
            node_type = "template"
            group = node_type
        elif isinstance(snode, NopKernelSchedulerNode):
            node_type = "nop"
            group = node_type
        elif isinstance(snode, SchedulerNode):
            node_type = "compute"
            group = snode.group
        elif isinstance(snode, FusedSchedulerNode):
            node_type = "fused"
            group = snode.group
        else:
            raise RuntimeError("Unknown node type")
        node_func = func_dict[node_type]
        fx_node = graph.call_function(node_func, args=(), kwargs=None)

        def in_output(snode):
            if isinstance(snode, FusedSchedulerNode):
                return any([in_output(x) for x in snode.snodes])
            return any([isinstance(user.node, OutputNode) for user in snode.users])

        if in_output(snode):
            outputs.append(fx_node)
        name = snode.get_name()
        fx_node.name = name

        fx_node.meta["fusion_meta"] = FusionMeta(group, [snode], node_type)

        if isinstance(snode, FusedSchedulerNode):
            for x in snode.snodes:
                buf_to_fx_node[x.get_name()] = fx_node
        buf_to_fx_node[name] = fx_node

        if first_node is None:
            first_node = fx_node

    # create edges between nodes
    for snode in snodes:
        name = snode.get_name()
        deps = snode.read_writes.reads

        fx_node = buf_to_fx_node[name]
        new_args = []
        for dep in deps:
            if dep.name in buf_to_fx_node:
                dep_node = buf_to_fx_node[dep.name]
            else:
                with graph.inserting_before(first_node):
                    dep_node = graph.placeholder(dep.name)
                    buf_to_fx_node[dep.name] = dep_node
            new_args.append(dep_node)

        fx_node.args = tuple(new_args)

    graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
    return graph
