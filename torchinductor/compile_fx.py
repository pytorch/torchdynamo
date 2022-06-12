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
            return cudagraphs_inner(compiled_fn, example_inputs, copy_outputs=False)
        else:
            return compiled_fn
    except Exception:
        if os.environ.get("TORCHINDUCTOR_DUMP_REPRO") == "1":
            wrap(functools.partial(dump_to_repro, gm))(*example_inputs)

        raise


def compile_fx_training(
    model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    from torchdynamo.optimizations.backends import aot_autograd

    return aot_autograd(
        model,
        example_inputs,
        fw_compiler=compile_fx_inner,
        decompositions=decompositions,
    )
