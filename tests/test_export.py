import torch
import torch.utils._pytree as pytree

import torchdynamo.testing


class ExportTests(torchdynamo.testing.TestCase):
    def test_export(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[0]
            lc_val = state[1]
            bar = {}
            for i in range(0, 4):
                bar2 = []
                for j in range(0, 3):
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                bar[f"key_str_{i}"] = bar2

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

        with torchdynamo.optimize("eager", nopython=False):
            real_result = func()

        torchdynamo.reset()

        exported = torchdynamo.export(func)
        out_graph = exported[0]
        out_spec = exported[3]

        dynamo_result = list(out_graph())
        dynamo_result_flat, dynamo_spec = pytree.tree_flatten(dynamo_result)
        real_result_flat, real_spec = pytree.tree_flatten(real_result)
        self.assertEqual(out_spec, real_spec)
        self.assertTrue(
            torchdynamo.utils.recursive_allclose(real_result_flat, dynamo_result_flat)
        )
