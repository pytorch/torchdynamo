import logging
import operator
import os
from itertools import chain

import torch
from torch.fx import GraphModule
from torch.fx import Node
from torch.nn.utils import _stateless

from ..allowed_functions import _allowed_function_ids
from ..utils import clone_inputs
from ..utils import istype
from .normalize import normalize_ir

log = logging.getLogger(__name__)


def fake_signature(fn, nargs):
    """FX gets confused by varargs, de-confuse it"""
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})


def constant_inputs(n: torch.fx.Node):
    """True if a given FX node has no others as inputs"""

    def visit(other_node):
        nonlocal only_constants
        only_constants = False
        return other_node

    only_constants = True
    torch.fx.map_arg((n.args, n.kwargs), visit)
    return only_constants


def debug_node(n: Node):
    target = n.target
    target = _allowed_function_ids().get(
        id(target),
        f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', '')} {target}",
    )
    return f"{n.op} {target} {n.args} {n.kwargs}"


def python_key_normalize(gm: torch.fx.GraphModule, example_inputs, decompositions={}):
    """
    Use AOT autograd for normalizing IR in inference mode.  This is useful
    for debugging and gives us a common IR for both eval and train modes.
    """
    from functorch._src.aot_autograd import aot_autograd_decompositions
    from functorch._src.aot_autograd import pytree
    from functorch._src.named_members_polyfill import _named_buffers
    from functorch._src.named_members_polyfill import _named_parameters
    from functorch._src.python_key import PythonKeyTracer
    from functorch._src.python_key import PythonTensor
    from functorch._src.python_key import pythonkey_decompose

    example_inputs = clone_inputs(example_inputs)

    # TODO(jansel): remove the need for IR normalization
    gm = normalize_ir(gm, example_inputs)

    params = {
        **dict(_named_parameters(gm, remove_duplicate=False)),
        **dict(_named_buffers(gm, remove_duplicate=False)),
    }
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = tuple(params_flat)
    params_len = len(params_flat)
    nargs = params_len + len(example_inputs)

    class PatchingInterpreter(torch.fx.Interpreter):
        def run_node(self, n: torch.fx.Node):
            try:
                result = super().run_node(n)

                if istype(result, torch.Tensor):
                    # this should a be a PythonTensor proxy, something is wrong

                    if constant_inputs(n):
                        # Tensor creation ops won't be captured because none
                        # of their inputs are PythonTensor proxies.
                        # Explicitly add them to the output graph.
                        result = PythonTensor(
                            result,
                            tracer.create_proxy(n.op, n.target, n.args, n.kwargs),
                        )
                    else:
                        # TODO(jansel): look into lstm getitem bug
                        if (
                            n.target is not operator.getitem
                            or "lstm" not in repr(n.args)
                            or os.environ.get("PYTHONKEY_VERBOSE") == "1"
                        ):
                            log.warning("returning real tensor? %s", debug_node(n))

                return result

            except Exception:
                log.exception("exception running %s", debug_node(n))
                raise

    def fn_for_tracing(*proxy_args):
        assert len(proxy_args) == nargs
        args = [
            PythonTensor(elem, proxy)
            for elem, proxy in zip(chain(params_flat, example_inputs), proxy_args)
        ]
        with _stateless.reparametrize_module(
            gm, pytree.tree_unflatten(args[:params_len], params_spec)
        ):
            out = PatchingInterpreter(gm).run(*args[params_len:])
        assert isinstance(out, (tuple, list)), "graph must output tuple()"
        return tuple(x.proxy for x in out)

    with pythonkey_decompose({**aot_autograd_decompositions, **decompositions}):
        tracer: torch.fx.Tracer = PythonKeyTracer()
        graph = tracer.trace(fake_signature(fn_for_tracing, nargs))
        traced = GraphModule(tracer.root, graph, "python_key_traced")

    traced.recompile()
    # record_graph_stats(traced)

    def make_wrapper(inner):
        def call_fn(*args):
            with torch.no_grad():
                return inner(*params_flat, *args)

        return call_fn

    return traced, make_wrapper


def python_key(gm: torch.fx.GraphModule, example_inputs):
    gm, make_wrapper = python_key_normalize(gm, example_inputs)
    return make_wrapper(gm.forward)
