import functools
import itertools
import operator
import os
import textwrap
from typing import List

import torch.fx

import torchdynamo.config
from torchdynamo.optimizations.backends import cudagraphs_inner
from torchdynamo.optimizations.python_key import python_key_normalize
from torchdynamo.testing import same

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
                from torchdynamo.testing import rand_strided,same
                from torchinductor.compile_fx import compile_fx

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
                with torchdynamo.optimize(compile_fx, nopython=True):
                    actual = Repro().forward(*args)
                assert same(actual, expected), (actual[0]-expected[0]).max()
                """
            )
        )
        print("wrote repro.py")


def compile_fx(
    model: torch.fx.GraphModule, example_inputs: List[torch.Tensor], cudagraphs=True
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

    if False:
        wrap(CheckEachNode(gm).run)(*example_inputs)

    if False:
        wrap(functools.partial(dump_to_repro, gm))(*example_inputs)

    graph = GraphLowering(gm, num_dynamic_inputs=len(example_inputs))
    with V.set_graph_handler(graph):
        wrap(graph.run)(*example_inputs)
        compiled_fn = wrap(graph.compile_to_fn())

    if example_inputs[0].device.type == "cuda" and cudagraphs:
        return cudagraphs_inner(compiled_fn, example_inputs)
    else:
        return compiled_fn
