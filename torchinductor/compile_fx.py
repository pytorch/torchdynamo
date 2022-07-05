import functools
import itertools
import operator
import os
import textwrap
from typing import List

import torch.fx

import torchdynamo.config
from torchdynamo.optimizations.python_key import python_key_normalize
from torchdynamo.testing import same
from torchdynamo.utils import identity

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


def compile_fx(
    model: torch.fx.GraphModule, example_inputs: List[torch.Tensor], cudagraphs=None
):
    """Main entrypoint to a compile given FX graph"""
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


def no_compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
):
    return gm.forward


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


def compile_fx_training(
    model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor]
):
    from torchdynamo.optimizations.backends import aot_autograd

    def fw_compiler(model: torch.fx.GraphModule, example_inputs):
        # model.graph.print_tabular()
        fixed = len(example_inputs) - len(example_inputs_)
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
    )
