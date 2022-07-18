import torch

import torchdynamo.testing
from torchdynamo.spec import Spec


class ExportTests(torchdynamo.testing.TestCase):
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
