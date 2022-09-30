import contextlib
import dis
import functools
import logging
import os.path
import time
import types
import unittest
from typing import Any
from typing import Callable
from typing import List
from typing import Set
from unittest.mock import patch

import torch
from torch import fx

import torchdynamo

from . import config
from .bytecode_transformation import create_instruction
from .bytecode_transformation import debug_checks
from .bytecode_transformation import is_generator
from .bytecode_transformation import transform_code_object
from .guards import CheckFunctionManager
from .guards import GuardedCode
from .optimizations import BACKENDS
from .utils import same

unsupported = torchdynamo.eval_frame.unsupported
three = 3

log = logging.getLogger(__name__)


def clone_me(x):
    if x is None:
        return None
    return x.detach().clone().requires_grad_(x.requires_grad)


def collect_results(model, prediction, loss, example_inputs):
    results = []
    results.append(prediction)
    results.append(loss)
    if isinstance(loss, torch.Tensor) and loss.item() > 1:
        log.warning(
            f"High loss value alert - {loss:.2f}. Can result in unstable gradients."
        )

    grads = dict()
    for name, param in model.named_parameters():
        grad = clone_me(param.grad)
        # Treat None and zero grad as same
        if param.grad is None:
            grad = torch.zeros_like(param)
        grads[name + ".grad"] = grad
    results.append(grads)
    for example in example_inputs:
        if isinstance(example, (tuple, list)):
            for inp in example:
                if isinstance(inp, torch.Tensor):
                    results.append(clone_me(inp.grad))
        else:
            if isinstance(example, torch.Tensor):
                results.append(clone_me(example.grad))
    return results


def requires_bwd_pass(out):
    if isinstance(out, torch.Tensor):
        return out.requires_grad
    elif isinstance(out, (list, tuple)):
        return any([requires_bwd_pass(x) for x in out])
    raise NotImplementedError("Don't know how to reduce", type(out))


def reduce_to_scalar_loss(out):
    """Reduce the output of a model to get scalar loss"""
    if isinstance(out, torch.Tensor):
        # Mean does not work on integer tensors
        return out.sum() / out.numel()
    elif isinstance(out, (list, tuple)):
        return sum([reduce_to_scalar_loss(x) for x in out]) / len(out)
    elif type(out).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
    ):
        return reduce_to_scalar_loss(out.logits)
    elif type(out).__name__ == "SquashedNormal":
        return out.mean.sum()
    elif isinstance(out, dict):
        return sum([reduce_to_scalar_loss(value) for value in out.values()]) / len(
            out.keys()
        )
    raise NotImplementedError("Don't know how to reduce", type(out))


def debug_dir():
    path = os.path.join(os.path.dirname(__file__), "../debug")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def debug_dump(name, code: types.CodeType, extra=""):
    with open(os.path.join(debug_dir(), name), "w") as fd:
        fd.write(
            f"{dis.Bytecode(code).info()}\n\n{dis.Bytecode(code).dis()}\n\n{extra}\n"
        )


def debug_insert_nops(frame, cache_size):
    """used to debug jump updates"""

    def insert_nops(instructions, code_options):
        instructions.insert(0, create_instruction("NOP"))
        instructions.insert(0, create_instruction("NOP"))

    if is_generator(frame.f_code):
        return None

    debug_checks(frame.f_code)
    code = transform_code_object(frame.f_code, insert_nops)

    return GuardedCode(code, CheckFunctionManager().check_fn)


class CompileCounter:
    def __init__(self):
        self.frame_count = 0
        self.op_count = 0

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    def clear(self):
        self.frame_count = 0
        self.op_count = 0


class CompileCounterWithBackend:
    def __init__(self, backend):
        self.frame_count = 0
        self.op_count = 0
        self.backend = backend

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        if self.backend == "inductor":
            from torchinductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)
        return BACKENDS[self.backend](gm, example_inputs)


def standard_test(self, fn, nargs, expected_ops=None, expected_ops_dynamic=None):
    if torchdynamo.config.dynamic_shapes and expected_ops_dynamic is not None:
        expected_ops = expected_ops_dynamic

    actual = CompileCounter()
    if expected_ops is None:
        expected = CompileCounter()
        try:
            gm = torch.fx.symbolic_trace(fn)
            expected(gm)
            print("\nfx.symbolic_trace graph:")
            gm.graph.print_tabular()
            expected_ops = expected.op_count
        except Exception:
            pass  # Silently ignore FX errors (not our issue)

    args1 = [torch.randn(10, 10) for _ in range(nargs)]
    args2 = [torch.randn(10, 10) for _ in range(nargs)]
    correct1 = fn(*args1)
    correct2 = fn(*args2)
    torchdynamo.reset()
    opt_fn = torchdynamo.optimize_assert(actual)(fn)
    val1a = opt_fn(*args1)
    val2a = opt_fn(*args2)
    val1b = opt_fn(*args1)
    val2b = opt_fn(*args2)
    torchdynamo.reset()
    self.assertTrue(same(val1a, correct1))
    self.assertTrue(same(val1b, correct1))
    self.assertTrue(same(val2a, correct2))
    self.assertTrue(same(val2b, correct2))
    self.assertEqual(actual.frame_count, 1)
    if expected_ops is not None:
        self.assertEqual(actual.op_count, expected_ops)


def evaluate(
    f: Callable,
    backends: Set[str] = [None, "inductor"],
    inputs: List[Any] = [],
    *,
    warmups: List[Any] = [],
    file=None,
    nopython=False,
):
    assert f is not None, "Must provide a callabke to evaluate"
    assert len(backends) > 0, "Must provide at least 1 backend"
    assert len(inputs) > 0, "Must provide at least 1 input"

    import cProfile
    from pstats import Stats

    backends = set(backends)

    print(f"Starting evaluation of {len(backends)} backends: {backends}", file=file)

    assert isinstance(backends, Set), "Backends must be a set"

    result_lookup_table = []
    for input_idx in range(0, len(inputs)):
        result_lookup_table.append({})

    for backend in backends:
        if backend is None:
            backend = "No Dynamo"

        # TODO(voz): If backend is inductor, get data from the inductor trace util

        print(f"EVALUATING {backend}", file=file)
        torchdynamo.reset()
        backend_fn = f
        # A None backend indicates a run without TorchDynamo
        if backend != "No Dynamo":
            backend_fn = torchdynamo.optimize(backend, nopython=nopython)(f)

        first_input = inputs[0]
        (
            explanation,
            _,
            _,
            _,
            _,
        ) = torchdynamo.explain(f, *first_input)
        print(f"EXPLAIN FOR {backend}: {explanation}", file=file)
        torchdynamo.reset()
        try:
            for warmup in warmups:
                backend_fn(*warmup)
            if len(warmups) > 0:
                print(f"WARMUP FINISHED FOR {backend}", file=file)

        except Exception as e:
            print(f"WARMUP FAILED FOR {backend}", file=file)
            raise e

        try:
            pr = cProfile.Profile()
            pr.enable()

            start = time.time()
            for input_idx in range(0, len(inputs)):
                input = inputs[input_idx]
                result = backend_fn(*input)
                result_lookup_table[input_idx][backend] = result

            end = time.time()

            pr.disable()
            stats = Stats(pr, stream=file)
            # TODO(voz): Memory profiling
            # TODO(voz): CUDA Memory profiling
            stats.sort_stats("tottime").print_stats(10)
            print(f"RUN FINISHED FOR {backend}", file=file)
            print(
                f"   RUN FOR {backend} TOOK: {end - start} seconds for {len(inputs)} inputs and {len(warmups)} warmups",
                file=file,
            )

        except Exception as e:
            print(f"RUN FAILED FOR {backend}", file=file)
            raise e

        all_equal = True
        for idx in range(0, len(result_lookup_table)):
            results_by_backend = result_lookup_table[idx]
            result = next(iter(results_by_backend.values()))
            equal = all(same(r, result) for r in results_by_backend.values())
            if not equal:
                print("MISMATCH AT INPUT IDX {idx}", file=file)
                all_equal = False

        if all_equal:
            print("ALL RESULTS EQUAL ACROSS ALL BACKENDS", file=file)


class TestCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = contextlib.ExitStack()
        cls._exit_stack.enter_context(
            patch.object(config, "raise_on_backend_error", True)
        )
        cls._exit_stack.enter_context(
            patch.object(config, "raise_on_ctx_manager_usage", True)
        )

    def setUp(self):
        torchdynamo.reset()
        torchdynamo.utils.counters.clear()

    def tearDown(self):
        for k, v in torchdynamo.utils.counters.items():
            print(k, v.most_common())
        torchdynamo.reset()
        torchdynamo.utils.counters.clear()


def dummy_fx_compile(gm: fx.GraphModule, example_inputs):
    return gm.forward


def format_speedup(speedup, pvalue, is_correct=True, pvalue_threshold=0.1):
    if not is_correct:
        return "ERROR"
    if pvalue > pvalue_threshold:
        return f"{speedup:.3f}x SAME"
    return f"{speedup:.3f}x p={pvalue:.2f}"


def requires_static_shapes(fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        if torchdynamo.config.dynamic_shapes:
            raise unittest.SkipTest("requires static shapes")
        return fn(*args, **kwargs)

    return _fn


def rand_strided(size, stride, dtype=torch.float32, device="cpu"):
    needed_size = sum((shape - 1) * stride for shape, stride in zip(size, stride)) + 1
    if dtype.is_floating_point:
        buffer = torch.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)
    return torch.as_strided(buffer, size, stride)


def _make_fn_with_patches(fn, *patches):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            for attr, val in patches:
                stack.enter_context(patch.object(torchdynamo.config, attr, val))

            return fn(*args, **kwargs)

    return _fn


def make_test_cls_with_patches(cls, cls_prefix, fn_suffix, *patches):
    class DummyTestClass(cls):
        pass

    DummyTestClass.__name__ = f"{cls_prefix}{cls.__name__}"

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                continue
            new_name = f"{name}{fn_suffix}"
            fn = _make_fn_with_patches(fn, *patches)
            fn.__name__ = new_name
            setattr(DummyTestClass, name, None)
            setattr(DummyTestClass, new_name, fn)

    return DummyTestClass
