import contextlib
import dis
import functools
import os.path
import types
import unittest
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
from .utils import same

unsupported = torchdynamo.eval_frame.unsupported
three = 3


def clone_me(x):
    if x is None:
        return None
    return x.detach().clone().requires_grad_(x.requires_grad)


def collect_results(model, prediction, loss, example_inputs):
    results = []
    results.append(prediction)
    results.append(loss)
    grads = dict()
    for name, param in model.named_parameters():
        grads[name + ".grad"] = clone_me(param.grad)
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
    raise NotImplementedError("Don't know how to reduce")


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
    with torchdynamo.optimize_assert(actual):
        val1a = fn(*args1)
        val2a = fn(*args2)
        val1b = fn(*args1)
        val2b = fn(*args2)
    self.assertTrue(same(val1a, correct1))
    self.assertTrue(same(val1b, correct1))
    self.assertTrue(same(val2a, correct2))
    self.assertTrue(same(val2b, correct2))
    self.assertEqual(actual.frame_count, 1)
    if expected_ops is not None:
        self.assertEqual(actual.op_count, expected_ops)


class TestCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = contextlib.ExitStack()
        cls._exit_stack.enter_context(patch.object(config, "debug", True))
        cls._exit_stack.enter_context(
            patch.object(config, "raise_on_backend_error", True)
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
        buffer = torch.randint(
            low=0, high=2, size=[needed_size], dtype=dtype, device=device
        )
    return torch.as_strided(buffer, size, stride)
