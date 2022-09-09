#!/usr/bin/env pytest
from unittest.mock import patch

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import make_fx

import torchdynamo.testing


class ExportTests(torchdynamo.testing.TestCase):
    # TODO(voz): Refactor to a shared test function.
    # The tests in this file are a little redundant,
    # They all take a func, run it with eager, then export it, then compare
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

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func()

        torchdynamo.reset()

        exported = torchdynamo.export(func)
        out_graph = exported[0]

        dynamo_result = out_graph()
        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_mismatched_out(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

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

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_list_unpack(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return x[0], first * second, x[1], x[2]

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_mismatched_out_2(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torchdynamo.reset()

        exported = torchdynamo.export(func, torch.tensor([[[1.3737, 0.1]]]))
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_list(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second, x

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_complex_reorder(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[0]
            second = x[1]
            third = x[2]
            return third, first, second, first * second, first * third

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dupes(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dupes_2(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.4, 0.4])
        inps = [inp, inp2]

        def func(x, z):
            y = x + 1
            return y, y, z

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_with_non_tensor_arg(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y, y, z

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_reorder_with_non_tensor_arg(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return z, y, y

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    def test_dupes_and_bypass_with_non_tensor_output(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y[0].item(), y, z

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_zeroes_in_and_out_different_shape_on_test(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return [[a], [b, c], [a + b], [[c + c]]]

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps_rand)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps_rand)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    def test_zeroes_in_new_shape_scalar_out(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return a[0].item() + b[0].item() + c[0].item()

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps_rand)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps_rand)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    def test_zeroes_in_new_shape_scalar_out_permute(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return b[0].item() + c[0].item() + a[0].item() + a[0].item()

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps_rand)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps_rand)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    def test_zeroes_in_new_shape_scalar_out_permute_dupe_and_bypass(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return a, b[0].item() + c[0].item() + a[0].item() + a[0].item(), a

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps_rand)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps_rand)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_func_return(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c

            def func2(y):
                return x * y

            return func2(x)

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps_rand)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps_rand)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dict_return(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c
            return {"a": x}

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps_rand)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps_rand)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_with_aten_graph(self):
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

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func()

        torchdynamo.reset()

        exported = torchdynamo.export(func, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph()
        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_mismatched_out_with_aten_graph(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torchdynamo.reset()

        exported = torchdynamo.export(
            func, torch.tensor([[[1.3737, 0.1]]]), aten_graph=True
        )
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_bypass_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_list_unpack_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return x[0], first * second, x[1], x[2]

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_mismatched_out_2_with_aten_graph(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torchdynamo.reset()

        exported = torchdynamo.export(
            func, torch.tensor([[[1.3737, 0.1]]]), aten_graph=True
        )
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_list_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second, x

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_complex_reorder_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[0]
            second = x[1]
            third = x[2]
            return third, first, second, first * second, first * third

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dupes_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dupes_2_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(inp)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inp)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.4, 0.4])
        inps = [inp, inp2]

        def func(x, z):
            y = x + 1
            return y, y, z

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_with_non_tensor_arg_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y, y, z

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_reorder_with_non_tensor_arg_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return z, y, y

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    def test_dupes_and_bypass_with_non_tensor_output_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y[0].item(), y, z

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_zeroes_in_and_out_different_shape_on_test_with_aten_graph(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return [[a], [b, c], [a + b], [[c + c]]]

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps_rand)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps_rand)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_func_return_with_aten_graph(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c

            def func2(y):
                return x * y

            return func2(x)

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps_rand)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps_rand)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_dict_return_with_aten_graph(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c
            return {"a": x}

        opt_func = torchdynamo.optimize("eager", nopython=True)(func)
        real_result = opt_func(*inps_rand)

        torchdynamo.reset()

        exported = torchdynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]
        flat_input, _ = pytree.tree_flatten(inps_rand)

        dynamo_result = out_graph(*flat_input)

        self.assertTrue(torchdynamo.utils.same(real_result, dynamo_result))

    def test_export_with_stack_trace(self):
        inp = torch.tensor([0.1, 0.1])
        linear = torch.nn.Linear(2, 2)

        def func(x):
            x = x + 1
            y = x.t()
            y = y.relu()
            y = linear(y)
            return y

        exported = torchdynamo.export(func, inp, aten_graph=False)
        out_graph = exported[0]

        for node in out_graph.graph.nodes:
            if node.op not in {"placeholder", "output"}:
                self.assertTrue(node.stack_trace is not None)

        torchdynamo.reset()

        exported = torchdynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]
        for node in out_graph.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(node.stack_trace is not None)

    def test_export_compare_optimize_with_make_fx(self):
        inp = torch.tensor([0.1, 0.1])
        linear = torch.nn.Linear(2, 2)

        def func(x):
            x = x + 1
            y = x.t()
            y = y.relu()
            y = linear(y)
            return y

        exported = torchdynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]
        export_result = out_graph(inp)

        torchdynamo.reset()

        def compiler(gm, sample_inputs):
            aten_gm = make_fx(gm)(*sample_inputs)

            self.assertEqual(len(aten_gm.graph.nodes), len(out_graph.graph.nodes))
            for node1, node2 in zip(aten_gm.graph.nodes, out_graph.graph.nodes):
                self.assertEqual(node1.op, node2.op)
                if node1.op == "call_function":
                    self.assertEqual(node1.target, node2.target)
                    self.assertEqual(len(node1.args), len(node2.args))
                    for arg1, arg2 in zip(node1.args, node2.args):
                        self.assertEqual(type(arg1), type(arg2))

            return aten_gm.forward

        opt_func = torchdynamo.optimize(compiler, nopython=True)(func)
        make_fx_result = opt_func(inp)

        self.assertTrue(torchdynamo.utils.same(make_fx_result, export_result))
