import torch
import torch.utils._pytree as pytree

import torchdynamo.testing


class ExportTests(torchdynamo.testing.TestCase):
    def test_export(self):
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

        torchdynamo.reset()

        exported = torchdynamo.export(func)
        out_graph = exported[0]

        dynamo_result = out_graph()
        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_mismatched_out(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        with torchdynamo.optimize("eager", nopython=True):
            real_result = func(torch.tensor([[[1.3737, 0.1]]]))

        torchdynamo.reset()

        exported = torchdynamo.export(func, torch.tensor([[[1.3737, 0.1]]]))
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))
        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_bypass(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second

        with torchdynamo.optimize("eager", nopython=True):
            real_result = func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))
