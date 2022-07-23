import functools
import itertools
import operator
import os
import textwrap
from typing import List

import torch.fx
from functorch.compile import min_cut_rematerialization_partition

import torchdynamo.config
from torchdynamo.optimizations.backends import aot_autograd
from torchdynamo.optimizations.normalize import normalize_ir
from torchdynamo.optimizations.python_key import python_key_normalize
from torchdynamo.testing import same
from torchdynamo.utils import identity
from torchdynamo.utils import init_logging
from functorch._src.partitioners import draw_graph
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from . import ir
from .codegen.cpp import CppScheduling
from .codegen.triton import TritonScheduling
from torch.fx.passes.tools_common import legalize_graph

from . import config
from .decomposition import decompositions
from .graph import GraphLowering
from .virtualized import V


class CheckEachNode(torch.fx.Interpreter):
    def call_function(self, target, args, kwargs):
        expected = target(*args, **kwargs)
        if target in (operator.getitem,):
            return expected

        g = torch.fx.Graph()
        g_args = []
        a_args = []
        for n, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                g_args.append(g.placeholder(f"arg{n}"))
                a_args.append(arg)
            else:
                g_args.append(arg)
        assert all(not isinstance(x, torch.Tensor) for x in kwargs.values())
        node = g.call_function(target, tuple(g_args), kwargs)
        if isinstance(expected, torch.Tensor):
            node = (node,)
        g.output(node)

        gm = torch.fx.GraphModule({}, g)
        graph = GraphLowering(gm)
        with V.set_graph_handler(graph):
            graph.run(*args, **kwargs)
            actual = graph.compile_to_fn()(*a_args)

        if isinstance(expected, torch.Tensor):
            actual = actual[0]

        print(target, same(expected, actual))
        assert same(expected, actual)

        return expected


def dump_to_repro(gm, *args):
    with open(os.path.join(torchdynamo.config.base_dir, "repro.py"), "w") as fd:
        fd.write(
            textwrap.dedent(
                """
                import torch
                import torchdynamo
                from torchdynamo.testing import rand_strided, same
                """
            )
        )
        fd.write("class Repro:\n")
        for i in itertools.count():
            try:
                val = getattr(gm, f"_tensor_constant{i}")
            except AttributeError:
                break
            fd.write(f"    _tensor_constant{i} = {val.item()!r}\n")
        fd.write(textwrap.indent(gm.code, "    "))
        fd.write("\n")

        fd.write("args = (\n")
        for arg in args:
            fd.write(
                f"  rand_strided({tuple(arg.size())!r}, {tuple(arg.stride())!r},"
                f" {arg.dtype!r}, {arg.device.type!r}),"
            )
            fd.write("\n")
        fd.write(")\n")
        fd.write(
            textwrap.dedent(
                """
                expected = Repro().forward(*args)
                with torchdynamo.optimize("inductor", nopython=True):
                    actual = Repro().forward(*args)
                assert same(actual, expected)
                """
            )
        )
        print("wrote repro.py")


def compile_fx_python_key(
    model: torch.fx.GraphModule, example_inputs: List[torch.Tensor], cudagraphs=None
):
    """Alternate version for inference only"""
    assert isinstance(model, torch.fx.GraphModule)
    assert all(isinstance(x, torch.Tensor) for x in example_inputs)

    gm, wrap = python_key_normalize(
        model, example_inputs, decompositions=decompositions
    )

    if config.dce:
        gm.graph.eliminate_dead_code()
    if config.debug:
        gm.graph.print_tabular()

    if os.environ.get("TORCHINDUCTOR_CHECK_OPS") == "1":
        wrap(CheckEachNode(gm).run)(*example_inputs)

    return compile_fx_inner(gm, example_inputs, wrap=wrap, cudagraphs=cudagraphs)


def compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    wrap=identity,
    cudagraphs=None,
    num_fixed=0,
):
    init_logging()

    if cudagraphs is None:
        cudagraphs = config.triton.cudagraphs

    try:
        graph = GraphLowering(gm, num_dynamic_inputs=len(example_inputs))
        with V.set_graph_handler(graph):
            wrap(graph.run)(*example_inputs)
            compiled_fn = wrap(graph.compile_to_fn())

        # make sure it works, causes issues for mutation
        # compiled_fn(*example_inputs)

        if (
            cudagraphs
            and set(graph.device_types) == {"cuda"}
            and not graph.mutated_inputs
        ):
            return cudagraphify(
                compiled_fn, example_inputs, static_input_idxs=range(num_fixed)
            )
        else:
            return compiled_fn
    except Exception:
        if os.environ.get("TORCHINDUCTOR_DUMP_REPRO") == "1":
            wrap(functools.partial(dump_to_repro, gm))(*example_inputs)

        raise


def get_fake_func(name):
    def func1(*args):
        return 0
    func1.__name__ = name
    return func1


def create_fx_graph(nodes, fname, backend = "triton", print_graph = False):

    func_dict = {}
    # import pprint
    # pprint.pprint(nodes)
    name_to_fx_node = {}
    graph = torch.fx.Graph()
    first_node = None

    if backend == "triton":
        group_fn = TritonScheduling(None).group_fn
        group_fn_NHW_C = TritonScheduling(None).group_fn_NHW_C
    else:
        group_fn = CppScheduling(None).group_fn

    # create call_function node for each Buffer and Kernel
    for node in nodes:
        name = node.get_name()      
        node_type = str(type(node)).split(".")[-1].replace("'>","")
        if node_type in func_dict:
            fake_f = func_dict[node_type]
        else:
            fake_f = get_fake_func(node_type)
            func_dict[node_type] = fake_f
        fx_node = graph.call_function(fake_f, args=(), kwargs=None)
        fx_node.name = name

        # gather meta data
        dtype = None
        if isinstance(node, ir.ComputedBuffer):
            dtype = node.data.dtype

        try:
            stride = node.get_stride()
            layout = type(node.layout)
            sizes = node.get_size()
            if isinstance(node, ir.ComputedBuffer):
                sizes, _ = node.simplify_reorder_and_tile()
            elif isinstance(node, ir.ExternKernel):
                sizes, _ = node.get_group_stride()

            if isinstance(node, ir.Convolution):
                group = group_fn_NHW_C(sizes)
            else:
                group = group_fn(sizes)
        except:
            group = torch.Size([0])
            
        metadata = TensorMetadata(group, dtype, False, stride, layout, None, None)
        fx_node.meta["tensor_meta"] = metadata

        name_to_fx_node[name] = fx_node
        if first_node is None:
            first_node = fx_node

    # create edges between nodes
    for node in nodes:
        name = node.get_name()      
        deps = node.get_reads()
        fx_node = name_to_fx_node[node.name]
        
        new_args = []
        for dep in deps:
            if dep.name in name_to_fx_node:
                dep_node = name_to_fx_node[dep.name]
            else:
                with graph.inserting_before(first_node):
                    dep_node = graph.placeholder(dep.name)  # assume it's a placeholder if not a computebox
                    name_to_fx_node[dep.name] = dep_node
            new_args.append(dep_node)

        fx_node.args = tuple(new_args)
    
    outputs = []
    for _,v in name_to_fx_node.items():
        if len(v.users) == 0:
            outputs.append(v)
    graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
    
    
    if print_graph:
        print(graph)
    print("starting creating module")
    gm = GraphModule({}, graph)
    graph = legalize_graph(gm)
    gm.graph.lint()
    # print(gm)
    print("starting drawing")
    draw_graph(gm, fname, clear_meta=False)


def draw_compute_box(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], fname = "image", print_graph = False):
    """
    Dump the graph of a compute box to a file with fname.
    """
    init_logging()
    wrap=identity

    try:
        graph = GraphLowering(gm, num_dynamic_inputs=len(example_inputs))
        with V.set_graph_handler(graph):
            wrap(graph.run)(*example_inputs)
            # import pprint
            # pprint.pprint(graph.buffers)
            # breakpoint()
            create_fx_graph(graph.buffers, fname, print_graph=print_graph)
    except Exception:
        if os.environ.get("TORCHINDUCTOR_DUMP_REPRO") == "1":
            wrap(functools.partial(dump_to_repro, gm))(*example_inputs)

        raise


def cudagraphify(model, inputs, static_input_idxs=()):
    """
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """
    assert isinstance(inputs, (list, tuple))
    static_inputs = [
        torch.zeros_like(x) if idx not in static_input_idxs else inputs[idx]
        for idx, x in enumerate(inputs)
    ]

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    if config.size_asserts:

        def run(*new_inputs):
            assert len(static_inputs) == len(new_inputs)
            for idx, (dst, src) in enumerate(zip(static_inputs, new_inputs)):
                if idx in static_input_idxs:
                    assert dst.data_ptr() == src.data_ptr()
                else:
                    dst.copy_(src)
            graph.replay()
            return static_outputs

    else:
        copy_indices = [
            idx for idx in range(len(static_inputs)) if idx not in static_input_idxs
        ]

        def run(*new_inputs):
            for idx in copy_indices:
                static_inputs[idx].copy_(new_inputs[idx])
            graph.replay()
            return static_outputs

    return run


def count_tangents(fx_g: torch.fx.GraphModule):
    """
    Infers which inputs are static for a backwards graph
    """

    def is_not_gradout(x):
        return "tangents" not in x.name

    arg_count = 0
    static_arg_idxs = []
    for n in fx_g.graph.nodes:
        if n.op == "placeholder":
            if is_not_gradout(n):
                static_arg_idxs.append(arg_count)
            arg_count += 1

    assert static_arg_idxs == list(range(len(static_arg_idxs)))
    return len(static_arg_idxs)

# def get_input_meta(args):
#     input_meta = []
#     if len(args) > 0 and isinstance(args[0], tuple):  # joint input
#         input_meta += get_input_meta(args[0])
#         input_meta += get_input_meta(args[1])
#         return input_meta
#     for arg in args:
#         if(type(arg) == int or type(arg) == float):
#             input_meta.append((type(arg),))
#         else:
#             input_meta.append((type(arg), arg.shape, arg.stride(), arg.dtype, arg.device))
#     return input_meta

model_name = "hf_Bert"

def compile_fx_aot(model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor]):
    """Main entrypoint to a compile given FX graph"""
    model_ = normalize_ir(model_, example_inputs_)
    num_example_inputs = len(example_inputs_)

    def fw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = len(example_inputs) - num_example_inputs
        return compile_fx_inner(model, example_inputs, num_fixed=fixed)

    def bw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = count_tangents(model)
        return compile_fx_inner(model, example_inputs, num_fixed=fixed)


    return aot_autograd(
        model_,
        example_inputs_,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        decompositions=decompositions,
        partition_fn=min_cut_rematerialization_partition,
    )


def compile_fx_aot_dump(model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor]):
    """Main entrypoint to a compile given FX graph"""
    model_ = normalize_ir(model_, example_inputs_)


    def fw_compiler(model: torch.fx.GraphModule, example_inputs):
        # import pickle
        # model.graph.set_codegen(torch.fx.graph.CodeGen())  # remove codegen
        # model.to_folder("hf_Bert_forward_0")
        # input_meta = get_input_meta(example_inputs)
        # pickle.dump(input_meta, open("hf_Bert_forward_0/hf_Bert_forward_0.input", "wb"))  # noqa: E501
        global model_name
        draw_compute_box(model, example_inputs, f"{model_name}_fw", print_graph=True)
        return model
        

    def bw_compiler(model: torch.fx.GraphModule, example_inputs):
        # print(model)
        global model_name
        draw_compute_box(model, example_inputs, f"{model_name}_bw", print_graph=True)
        return model

    return aot_autograd(
        model_,
        example_inputs_,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        decompositions=decompositions,
        partition_fn=min_cut_rematerialization_partition,
    )


def compile_fx(model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor]):
    """Main entrypoint to a compile given FX graph"""
    if config.aot_autograd:
        return compile_fx_aot(model_, example_inputs_)
    else:
        return compile_fx_python_key(model_, example_inputs_)
