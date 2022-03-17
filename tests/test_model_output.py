#!/usr/bin/env pytest
import dataclasses
import unittest.mock

import torch

import torchdynamo.testing
from torchdynamo.testing import same

try:
    from transformers import modeling_outputs
    from transformers.file_utils import ModelOutput
    from transformers.modeling_outputs import BaseModelOutput
except ImportError:
    modeling_outputs = None


def maybe_skip(fn):
    if modeling_outputs is None:
        return unittest.skip("requires HuggingFace")(fn)
    return fn


class TestModelOutput(torchdynamo.testing.TestCase):
    @maybe_skip
    def test_mo_create(self):
        def fn(a, b):
            tmp = BaseModelOutput(a + 1, attentions=b + 3)
            return tmp

        torchdynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=2)

    @maybe_skip
    def test_mo_assign(self):
        def fn(a, b):
            tmp = BaseModelOutput(last_hidden_state=b + 3)
            tmp.hidden_states = a + 7
            tmp["attentions"] = a + b + 6
            return tmp

        args = [torch.randn(10), torch.randn(10)]
        obj1 = fn(*args)

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize_assert(cnts):
            obj2 = fn(*args)
        self.assertTrue(same(obj1.last_hidden_state, obj2.last_hidden_state))
        self.assertTrue(same(obj1.hidden_states, obj2.hidden_states))
        self.assertTrue(same(obj1.attentions, obj2.attentions))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def _common(self, fn, op_count):
        args = [
            BaseModelOutput(
                last_hidden_state=torch.randn(10), attentions=torch.randn(10)
            )
        ]
        obj1 = fn(*args)
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize_assert(cnts):
            obj2 = fn(*args)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, op_count)

    @maybe_skip
    def test_mo_getattr(self):
        def fn(obj: BaseModelOutput):
            x = obj.last_hidden_state * 10
            if obj.hidden_states is not None:
                x += obj.hidden_states
            if obj.attentions is not None:
                x += obj.attentions
            return x

        self._common(fn, 2)

    @maybe_skip
    def test_mo_getitem(self):
        def fn(obj: BaseModelOutput):
            x = obj["last_hidden_state"] * 10
            if "hidden_stats" in obj:
                x += obj["hidden_states"]
            if "attentions" in obj:
                x += obj["attentions"]
            return x

        self._common(fn, 2)

    @maybe_skip
    def test_mo_tuple(self):
        def fn(obj: BaseModelOutput):
            a, b = obj.to_tuple()
            return a + b * 10

        self._common(fn, 2)

    @maybe_skip
    def test_mo_index(self):
        def fn(obj: BaseModelOutput):
            return obj[0] * 10 + obj[1]

        self._common(fn, 2)

    @maybe_skip
    def test_mo_init(self):
        @dataclasses.dataclass
        class MyDataClass(ModelOutput):
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None
            d: torch.Tensor = None
            e: torch.Tensor = None

        def fn(obj):
            class_fields = dataclasses.fields(obj)
            assert len(class_fields)
            assert all(field.default is None for field in class_fields[1:])
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            assert not other_fields_are_none

            total = getattr(obj, class_fields[0].name)
            for field in class_fields[1:]:
                v = getattr(obj, field.name)
                if v is not None:
                    total += v

            return total

        tensors = [torch.randn(10), torch.randn(10), torch.randn(10)]
        obj1 = MyDataClass(*tensors)
        correct1 = fn(obj1)

        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            obj2 = MyDataClass(*tensors)
            self.assertTrue(same(fn(obj2), correct1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)
