import dis
import functools
import os.path
import types
import unittest

import torch
from torch import fx

import torchdynamo

from .bytecode_transformation import create_instruction
from .bytecode_transformation import debug_checks
from .bytecode_transformation import is_generator
from .bytecode_transformation import transform_code_object
from .guards import GuardedCode

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


def same(a, b):
    """Check correctness to see if a and b match"""
    if isinstance(a, (list, tuple, torch.nn.ParameterList, torch.Size)):
        assert isinstance(b, (list, tuple)), f"type mismatch {type(a)} {type(b)}"
        return len(a) == len(b) and all(same(ai, bi) for ai, bi in zip(a, b))
    elif isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a.keys()) == set(
            b.keys()
        ), f"keys mismatch {set(a.keys())} == {set(b.keys())}"
        for k in a.keys():
            if not (same(a[k], b[k])):
                print("Accuracy failed for key name", k)
                return False
        return True
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        return torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    elif isinstance(a, (str, int, float, type(None), bool, torch.device)):
        return a == b
    elif type(a).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
        "LongformerMaskedLMOutput",
        "Instances",
        "SquashedNormal",
        "Boxes",
        "Normal",
        "TanhTransform",
        "Foo",
        "Variable",
    ):
        assert type(a) is type(b)
        return all(same(getattr(a, key), getattr(b, key)) for key in a.__dict__.keys())
    else:
        raise RuntimeError(f"unsupported type: {type(a).__name__}")


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

    return GuardedCode(code)


class CompileCounter:
    def __init__(self):
        self.frame_count = 0
        self.op_count = 0

    def __call__(self, gm: torch.fx.GraphModule):
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
        torchdynamo.reset()
        torchdynamo.config.debug = cls.prior_debug

    @classmethod
    def setUpClass(cls):
        torchdynamo.reset()
        cls.prior_debug = torchdynamo.config.debug
        torchdynamo.config.debug = True

    def setUp(self):
        torchdynamo.utils.counters.clear()

    def tearDown(self):
        for k, v in torchdynamo.utils.counters.items():
            print(k, v.most_common())
        torchdynamo.utils.counters.clear()


def dummy_fx_compile(gm: fx.GraphModule):
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
