import base64
import functools
import hashlib
import io
import itertools
import json
import os
import shutil
import time
from collections import defaultdict

import torch

from .backends import clone_inputs
from .backends import ipex
from .backends import onnxrt
from .backends import optimize_for_inference
from .backends import static_runtime
from .backends import torchscript
from .backends import tvm_compile
from .normalize import long_name
from torchdynamo import config
from torchdynamo.utils import counters, count_calls, warning


def string_key(gm: torch.fx.GraphModule, example_inputs):
    out = io.StringIO()
    node_to_id = defaultdict(iter(itertools.count()).__next__)

    def argkey(n: torch.fx.Node):
        return f"#{node_to_id[n]}"

    def tensorkey(t):
        if isinstance(t, torch.Tensor):
            requires_grad = t.requires_grad and torch.torch.is_grad_enabled()
            return (
                f"{t.__class__.__name__}({t.dtype}, {t.device}, "
                f"{tuple(t.size())}, {tuple(t.stride())}, {requires_grad})"
            )
        return type(t).__name__

    inputs_iter = iter(example_inputs)

    for node in gm.graph.nodes:
        key = argkey(node)
        name = "."
        if node.op == "placeholder":
            name = tensorkey(next(inputs_iter))
        elif node.op == "get_attr":
            val = eval(f"self.{node.target}", {"self": gm})
            name = tensorkey(val)
        elif node.op in ("call_function", "call_method", "call_module"):
            name = long_name(gm, node)
        out.write(
            f"{key} {node.op} {name} "
            f"{torch.fx.map_arg(node.args, argkey)!r} "
            f"{torch.fx.map_arg(node.kwargs, argkey)!r}\n"
        )
    return out.getvalue()


def graph_hash(gm: torch.fx.GraphModule, example_inputs):
    return "g" + base64.urlsafe_b64encode(
        hashlib.sha256(string_key(gm, example_inputs).encode("utf-8")).digest()
    )[:39].decode("utf-8")


def folder_name(gm: torch.fx.GraphModule, example_inputs):
    base = os.path.join(config.base_dir, "subgraphs")
    if not os.path.exists(base):
        os.mkdir(base)
        open(os.path.join(base, "__init__.py"), "w").close()
    return os.path.join(base, graph_hash(gm, example_inputs))


def user_compiler(gm: torch.fx.GraphModule, example_inputs):
    if torch.is_grad_enabled():
        if any(
            getattr(x, "requires_grad", False)
            for x in itertools.chain(example_inputs, gm.parameters(True))
        ):
            warning("not optimizing requires_grad=True")
            return gm.forward

    # normalize(gm)
    # gm = NormalizeOperators(gm).transform()
    # Inplacifier(gm).inplacify()

    for node in gm.graph.nodes:
        if node.op in ("call_function", "call_method", "call_module"):
            counters[node.op][long_name(gm, node)] += 1
        elif node.op in ("placeholder", "output", "get_attr"):
            pass
        else:
            assert False, node.op

    gm.recompile()

    path = folder_name(gm, example_inputs)
    if not os.path.exists(path):
        if count_calls(gm.graph) >= config.minimum_call_count:
            try:
                gm.to_folder(path)
                with open(os.path.join(path, "key"), "w") as fd:
                    fd.write(string_key(gm, example_inputs))
                with open(os.path.join(path, "example_inputs.pt"), "wb") as fd:
                    torch.save(example_inputs, fd)
                open(os.path.join(path, "timestamp"), "w").write(str(time.time()))
            except Exception:
                shutil.rmtree(path)
                raise
    else:
        open(os.path.join(path, "timestamp"), "w").write(str(time.time()))
        if os.path.exists(os.path.join(path, "perf.json")):
            ts = functools.partial(torchscript, gm, example_inputs)
            backends = {
                "eager": lambda: gm.forward,
                "torchscript": ts,
                "freezing": lambda: optimize_for_inference(ts(), example_inputs),
                "static_runtime": lambda: static_runtime(ts(), example_inputs),
                "onnxrt": lambda: onnxrt(ts(), example_inputs),
                "tvm": lambda: tvm_compile(ts(), example_inputs),
                "ipex": lambda: ipex(ts(), example_inputs),
                "ansor20k": lambda: tvm_compile(
                    ts(),
                    example_inputs,
                    os.path.join(path, "ansor20k"),
                ),
            }
            perf = json.loads(open(os.path.join(path, "perf.json")).read())
            best = "eager"
            best_sec = float("inf")
            for name, sec in perf.items():
                assert name in backends
                if sec < best_sec and name in backends:
                    best = name
                    best_sec = sec
            if best != "eager":
                example_inputs = clone_inputs(example_inputs)
            return backends[best]()

    return gm.forward
