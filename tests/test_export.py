import torch

import torchdynamo.testing
from torchdynamo.spec import Spec


class ExportTests(torchdynamo.testing.TestCase):
    def allclose(lhs, rhs, rtol=1e-5, atol=1e-8):
        r"""
        Unlike torch.allocse which only handles Tensor arguments, allclose handles
        list, tuple, dict and nesting of these as well.
        """
        if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
            return torch.allclose(lhs, rhs, rtol, atol)
        if isinstance(lhs, (tuple, list)) and isinstance(rhs, (tuple, list)):
            return len(lhs) == len(rhs) and all(
                ExportTests.allclose(a, b, rtol, atol) for a, b in zip(lhs, rhs)
            )
        if isinstance(lhs, dict) and isinstance(rhs, dict):
            lhs_keys = set(lhs.keys())
            rhs_keys = set(rhs.keys())
            if lhs_keys != rhs_keys:
                return False
            return all(allclose(lhs[k], rhs[k], rtol, atol) for k in lhs)
        else:
            raise RuntimeError(
                f"Unexpected types: lhs type {type(lhs)}, rhs type {type(rhs)}"
            )

    def test_export_spec_capture(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[1]
            lc_val = state[2]
            bar = []
            for i in range(0, 4):
                bar2 = []
                for j in range(0, 3):
                    bar2.append(lc_key + lc_val + torch.randn([1, 1]))
                bar.append(bar2)

            return bar

        def func():
            mems = torch.randn([1, 1, 96])
            state = [torch.randn([1, 1, 96])] * 7
            return pre_attention_state_ops(torch.randn(3, 4), mems, state)

        exported = torchdynamo.export(func)
        out_spec = exported[3]
        self.assertEqual(f"{out_spec}", "List[List[TTT]List[TTT]List[TTT]List[TTT]]")

    def test_non_export_spec_capture(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[1]
            lc_val = state[2]
            bar = []
            for i in range(0, 4):
                bar2 = []
                for j in range(0, 3):
                    bar2.append(lc_key + lc_val + torch.randn([1, 1]))
                bar.append(bar2)

            return bar

        def func():
            mems = torch.randn([1, 1, 96])
            state = [torch.randn([1, 1, 96])] * 7
            return pre_attention_state_ops(torch.randn(3, 4), mems, state)

        with torchdynamo.optimize("eager", nopython=True):
            real_result = func()

        real_spec = Spec.describe_spec(real_result)

        self.assertEqual(f"{real_spec}", "List[List[TTT]List[TTT]List[TTT]List[TTT]]")

    def test_spec_apply(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[0]
            lc_val = state[1]
            bar = []
            for i in range(0, 4):
                bar2 = []
                for j in range(0, 3):
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                bar.append(bar2)

            return bar

        def func():
            mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
            state = [
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            ]
            i = torch.tensor(
                [
                    [0.0313, -0.1487, -0.3846, -0.5321],
                    [-1.7073, 1.3331, -0.0890, -1.4935],
                    [-0.8314, -0.1862, -0.5935, 1.5232],
                ]
            )
            return pre_attention_state_ops(i, mems, state)

        with torchdynamo.optimize("eager", nopython=True):
            real_result = func()

        real_spec = Spec.describe_spec(real_result)

        torchdynamo.reset()

        exported = torchdynamo.export(func)
        out_graph = exported[0]
        out_spec = exported[3]

        dynamo_result = list(out_graph())
        unflattened = Spec.apply_spec(out_spec, dynamo_result)

        # Specs should match
        self.assertEqual(f"{real_spec}", "List[List[TTT]List[TTT]List[TTT]List[TTT]]")
        self.assertEqual(f"{out_spec}", "List[List[TTT]List[TTT]List[TTT]List[TTT]]")

        # Unflattened dynamo result should be identical to the real traced result!
        self.assertTrue(ExportTests.allclose(unflattened, real_result))
