import logging
import operator
import os
from itertools import chain
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from torch.fx import GraphModule
from torch.fx import Node
from torch.nn.utils import _stateless

from ..allowed_functions import torch_get_name
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
    target = torch_get_name(
        target,
        f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', '')} {target}",
    )
    return f"{n.op} {target} {n.args} {n.kwargs}"


def python_key_normalize(
    gm: torch.fx.GraphModule,
    example_inputs,
    decompositions={},
    PythonTensorOverrideClass=None,
    post_trace_hook=None,
):
    """
    Use AOT autograd for normalizing IR in inference mode.  This is useful
    for debugging and gives us a common IR for both eval and train modes.
    """
    from functorch._src.aot_autograd import aot_autograd_decompositions
    from functorch._src.aot_autograd import pytree
    from functorch._src.named_members_polyfill import _named_buffers
    from functorch._src.named_members_polyfill import _named_parameters
    from torch.fx import Tracer

    PythonTensorClass = PythonTensorOverrideClass
    if not PythonTensorClass:
        from functorch._src.python_key import PythonTensor

        PythonTensorClass = PythonTensor

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
                        result = PythonTensorClass(
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
            PythonTensorClass(elem, proxy)
            for elem, proxy in zip(chain(params_flat, example_inputs), proxy_args)
        ]
        with _stateless.reparametrize_module(
            gm, pytree.tree_unflatten(args[:params_len], params_spec)
        ):
            out = PatchingInterpreter(gm).run(*args[params_len:])

        assert isinstance(out, (tuple, list)), "graph must output tuple()"
        return tuple(x.proxy for x in out)

    class PythonKeyTracer(Tracer):
        def __init__(self):
            super().__init__()

        def call_module(
            self,
            m: torch.nn.Module,
            forward: Callable[..., Any],
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> Any:
            return forward(*args, **kwargs)

        def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
            if isinstance(attr_val, torch.nn.Parameter):
                for n, p in self.root.named_parameters():
                    if attr_val is p:
                        if n not in parameter_proxy_cache:
                            proxy = self.create_proxy("get_attr", n, (), {})
                            parameter_proxy_cache[n] = PythonTensorClass(
                                attr_val, proxy
                            )
                        return parameter_proxy_cache[n]
                return attr_val
            return attr_val

        # We need to do this so that parameters entering the `make_fx` context have
        # a reference to them (and also have requires_grad set on them correctly
        # I'm not actually sure if this is the right thing to do ...
        def create_arg(self, a: Any):
            if isinstance(a, torch.nn.Parameter):
                for n, p in self.root.named_parameters():
                    if a is p:
                        return self.create_node("get_attr", n, (), {})
                qualname: Optional[str] = None

                if not qualname:
                    i = 0
                    while True:
                        qualname = f"_param_constant{i}"
                        if not hasattr(self.root, qualname):
                            break
                        i += 1
                    setattr(self.root, qualname, a)

                return self.create_node("get_attr", qualname, (), {})
            return super().create_arg(a)

    with pythonkey_decompose({**aot_autograd_decompositions, **decompositions}):
        tracer: torch.fx.Tracer = PythonKeyTracer()
        graph = tracer.trace(fake_signature(fn_for_tracing, nargs))
        traced = GraphModule(tracer.root, graph, "python_key_traced")

    traced.recompile()
    # record_graph_stats(traced)
    if post_trace_hook is not None:
        post_trace_hook(tracer, graph)

    def make_wrapper(inner):
        def call_fn(*args):
            with torch.no_grad():
                return inner(*params_flat, *args)

        return call_fn

    return traced, make_wrapper


def python_key(gm: torch.fx.GraphModule, example_inputs):
    gm, make_wrapper = python_key_normalize(gm, example_inputs)
    return make_wrapper(gm.forward)
