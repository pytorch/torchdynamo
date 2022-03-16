import base64
import hashlib
import io
import itertools
import json
import logging
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import torch

from torchdynamo import config
from torchdynamo.utils import check_is_cuda
from torchdynamo.utils import checkpoint_params
from torchdynamo.utils import clone_inputs
from torchdynamo.utils import count_calls
from torchdynamo.utils import counters
from torchdynamo.utils import timed

from ..exc import warning
from .backends import BACKENDS
from .normalize import long_name
from .normalize import normalize_ir

log = logging.getLogger(__name__)


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


def check_requires_grad(gm, example_inputs):
    if torch.is_grad_enabled():
        if any(
            getattr(x, "requires_grad", False)
            for x in itertools.chain(example_inputs, gm.parameters(True))
        ):
            return True
    return False


def jit_trace(gm, example_inputs):
    """Wrapper around jit.trace to handle hooks"""
    restore_backward_hooks = []

    def visit(mod):
        if mod._backward_hooks:
            restore_backward_hooks.append((mod, mod._backward_hooks))
            mod._backward_hooks = []

    if not check_requires_grad(gm, example_inputs):
        # in inference mode it is safe to ignore backwards hooks to allow tracing
        gm.apply(visit)

    try:
        return torch.jit.trace(gm.forward, example_inputs)
    finally:
        for mod, hooks in restore_backward_hooks:
            mod._backward_hooks = hooks


def same(left, right):
    return len(left) == len(right) and all(
        torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(left, right)
    )


class TorchScriptStrategy(object):
    """Common base for backend strategies that use TorchScript"""

    @classmethod
    def compile_fn(cls, gm: torch.fx.GraphModule, example_inputs):
        if count_calls(gm.graph) < 2:
            return gm.forward  # no point for tiny graphs
        return cls(gm, example_inputs).verified_candidate()

    def __init__(self, gm: torch.fx.GraphModule, example_inputs):
        super(TorchScriptStrategy, self).__init__()
        self.restore = checkpoint_params(gm)
        self.original_example_inputs = example_inputs
        self.correct = gm.forward(*self.example_inputs)
        self.gm = normalize_ir(gm, self.original_example_inputs)
        self.scripted = jit_trace(self.gm, self.example_inputs)

    @property
    def example_inputs(self):
        return clone_inputs(self.original_example_inputs)

    def verified_candidate(self):
        try:
            candidate = self.candidate()
            if candidate is None or candidate is self.gm.forward:
                return self.gm.forward

            self.restore()
            result = candidate(*self.example_inputs)

            if same(result, self.correct):
                return candidate

            print(f"incorrect candidate {self}")

            return self.gm.forward
        except Exception:
            log.exception("error in verified_candidate()")
            return self.gm.forward
        finally:
            self.restore()

    def candidate(self):
        raise NotImplementedError()


class FixedStrategy1(TorchScriptStrategy):
    def candidate(self):
        return self.scripted


fixed_strategy1 = FixedStrategy1.compile_fn


class FixedStrategy2(TorchScriptStrategy):
    def candidate(self):
        cg = BACKENDS["cudagraphs_ts"](self.scripted, self.example_inputs)
        if cg is not None:
            return cg
        return self.scripted


fixed_strategy2 = FixedStrategy2.compile_fn


class OfflineAutotuner(TorchScriptStrategy):
    def candidate(self):
        gm = self.gm
        if check_requires_grad(gm, self.original_example_inputs):
            warning("not optimizing requires_grad=True")
            return None

        path = folder_name(gm, self.original_example_inputs)
        if not os.path.exists(path):
            # a new graph! lets save it for offline tuning
            try:
                gm.to_folder(path)
                torch.jit.save(self.scripted, os.path.join(path, "model.ts"))

                open(os.path.join(path, "key"), "w").write(
                    string_key(gm, self.original_example_inputs)
                )
                save_pt(path, "example_inputs.pt", self.original_example_inputs)
                save_pt(path, "example_outputs.pt", self.correct)
                save_pt(path, "rng_state.pt", torch.get_rng_state())
                save_metadata(path, self.gm, self.original_example_inputs)
                touch_timestamp(path)
            except Exception:
                shutil.rmtree(path)
                raise
        else:
            touch_timestamp(path)
            if os.path.exists(os.path.join(path, "perf.json")):
                best = argmin(json.loads(open(os.path.join(path, "perf.json")).read()))
                counters["backend"][best] += 1
                if best != "eager":
                    return BACKENDS[best](self.scripted, self.example_inputs)

        return None


offline_autotuner = OfflineAutotuner.compile_fn


def save_pt(path, name, data):
    with open(os.path.join(path, name), "wb") as fd:
        torch.save(data, fd)


def save_metadata(path, gm, example_inputs):
    with open(os.path.join(path, "metadata.json"), "w") as fd:
        json.dump(
            {
                "is_cuda": check_is_cuda(gm, example_inputs),
            },
            fd,
        )


def touch_timestamp(path):
    open(os.path.join(path, "timestamp"), "w").write(str(time.time()))


def argmin(perf):
    best = "eager"
    best_sec = float("inf")
    for name, sec in perf.items():
        if sec < best_sec:
            best = name
            best_sec = float(sec)
            if name == "eager":
                # small bias torwards using eager since it is more robust
                best_sec *= 0.99
    return best


class OnlineAutotuner(TorchScriptStrategy):
    repeat = 15

    def candidate(self):
        if check_requires_grad(self.gm, self.original_example_inputs):
            warning("not optimizing requires_grad=True")
            return None
        self.scripted = self.scripted.eval()
        example_inputs_copy = self.example_inputs
        models = [("eager", self.gm.forward)]
        for name in self.select_backends():
            try:
                compiled_model = BACKENDS[name](self.scripted, example_inputs_copy)
                if compiled_model is None:
                    continue
                self.restore()
                result = compiled_model(*self.example_inputs)
                assert same(result, self.correct)
                models.append((name, compiled_model))
            except AssertionError:
                logging.exception(f"incorrect while running {name}")
            except Exception:
                logging.exception(f"error while running {name}")

        timings = np.zeros((self.repeat, len(models)), np.float64)
        for rep in range(timings.shape[0]):
            # interleave the runs to handle frequency scaling and load changes
            for i, (n, m) in enumerate(models):
                result, timings[rep, i] = timed(m, example_inputs_copy)
        median = np.median(timings, axis=0)
        median[0] *= 0.99  # a bias towards eager
        best = int(np.argmin(median))
        counters["backend"][models[best][0]] += 1
        return models[best][1]

    def select_backends(self):
        if check_is_cuda(self.gm, self.original_example_inputs):
            backend_names = [
                "ts",
                "cudagraphs_ts_ofi",
                "nnc_ofi",
                "tensorrt",
            ]
        else:
            backend_names = ["ofi", "onnxrt_cpu"]
        return backend_names


online_autotuner = OnlineAutotuner.compile_fn


BACKENDS["autotune"] = online_autotuner
