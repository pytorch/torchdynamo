import base64
import hashlib
import io
import itertools
import json
import os
import shutil
import time
from collections import defaultdict

import torch
from torch.fx.experimental.normalize import NormalizeOperators

from torchdynamo import config
from torchdynamo.utils import clone_inputs
from torchdynamo.utils import count_calls
from torchdynamo.utils import counters
from torchdynamo.utils import torchscript
from torchdynamo.utils import warning
from .analysis import ShapeAliasingAndMutationProp
from .backends import BACKENDS
from .normalize import Functionalization
from .normalize import long_name
from .normalize import normalize
from ..variable_tracker import typestr


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


def record_graph_stats(gm):
    for node in gm.graph.nodes:
        if node.op in ("call_function", "call_method", "call_module"):
            counters[node.op][long_name(gm, node)] += 1
        elif node.op in ("placeholder", "output", "get_attr"):
            pass
        else:
            assert False, node.op


def user_compiler(gm: torch.fx.GraphModule, example_inputs):
    if torch.is_grad_enabled():
        if any(
            getattr(x, "requires_grad", False)
            for x in itertools.chain(example_inputs, gm.parameters(True))
        ):
            warning("not optimizing requires_grad=True")
            return gm.forward

    any_cuda = any(
        x.is_cuda for x in itertools.chain(example_inputs, gm.parameters(True))
    )
    if any_cuda:
        assert all(
            x.is_cuda for x in itertools.chain(example_inputs, gm.parameters(True))
        )

    example_inputs = clone_inputs(example_inputs)

    if config.normalize_ir:
        normalize(gm)
        gm = NormalizeOperators(gm).transform()
        ShapeAliasingAndMutationProp(gm).run(*example_inputs)
        gm = Functionalization(gm).transform()

    record_graph_stats(gm)
    gm.recompile()

    path = folder_name(gm, example_inputs)
    if not os.path.exists(path):
        if count_calls(gm.graph) >= config.minimum_call_count:
            try:
                gm.to_folder(path)
                jit_model = torchscript(gm, example_inputs, verbose=False)
                if jit_model is not None:
                    torch.jit.save(jit_model, os.path.join(path, "model.ts"))
                with open(os.path.join(path, "key"), "w") as fd:
                    fd.write(string_key(gm, example_inputs))
                with open(os.path.join(path, "example_inputs.pt"), "wb") as fd:
                    torch.save(example_inputs, fd)
                with open(os.path.join(path, "example_outputs.pt"), "wb") as fd:
                    torch.save(gm(*example_inputs), fd)
                open(os.path.join(path, "timestamp"), "w").write(str(time.time()))
                with open(os.path.join(path, "metadata.json"), "w") as fd:
                    json.dump(
                        {
                            "is_cuda": any_cuda,
                        },
                        fd,
                    )
            except Exception:
                shutil.rmtree(path)
                raise
    else:
        open(os.path.join(path, "timestamp"), "w").write(str(time.time()))
        if os.path.exists(os.path.join(path, "perf.json")):
            best = argmin(json.loads(open(os.path.join(path, "perf.json")).read()))
            counters["backend"][best] += 1
            return BACKENDS[best](gm, example_inputs)

    return gm.forward


def argmin(perf):
    best = "eager"
    best_sec = float("inf")
    for name, sec in perf.items():
        if sec < best_sec:
            best = name
            best_sec = sec
    return best
