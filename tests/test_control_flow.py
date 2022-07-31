from unittest.mock import patch

import torch

import torchdynamo.testing
from torchdynamo.logic.control_flow import cond


class ControlFlowTests(torchdynamo.testing.TestCase):
    @patch.object(torchdynamo.config, "fake_tensor_propagation", False)
    def test_simple_condition(self):
        from_true = None
        from_false = None

        @torchdynamo.optimize("eager", nopython=True)
        def foo():
            nonlocal from_true
            nonlocal from_false

            def compute(pred, z):
                def when_true(x):
                    return x + x

                def when_false(x):
                    return x * x

                return cond(pred, when_true, when_false, (z,))

            a = torch.tensor([0.5, 0.5])

            from_true = compute(True, a)
            from_false = compute(False, a)

        foo()
        self.assertTrue(
            torch.allclose(from_true, torch.tensor([1.0, 1.0]))
        )  # True adds
        self.assertTrue(
            torch.allclose(from_false, torch.tensor([0.25, 0.25]))
        )  # False multiplies

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    @patch.object(torchdynamo.config, "fake_tensor_propagation", False)
    def test_simple_condition_nested(self):
        from_true_a = None
        from_true_b = None
        from_false = None

        @torchdynamo.optimize("eager")
        def foo():
            nonlocal from_true_a
            nonlocal from_true_b
            nonlocal from_false

            def nested_when_true():
                return -1

            def nested_when_false():
                return 1

            def compute(pred, z):
                def when_true(x):
                    inner_pred = x.item() < 0.5
                    pos_or_neg = cond(
                        inner_pred, nested_when_true, nested_when_false, tuple()
                    )
                    return x * pos_or_neg

                def when_false(x):
                    return x * x

                return cond(pred, when_true, when_false, (z,))

            a = torch.tensor([0.5])
            b = torch.tensor([0.25])

            from_true_a = compute(
                True, a
            )  # True -> when_true -> not less than .5 -> nested_when_false -> .5 * 1
            from_true_b = compute(
                True, b
            )  # True -> when_true -> less than .5 -> nested_when_true -> .25 * -1
            from_false = compute(False, a)  # False -> when_false -> .5 * .5

        foo()
        self.assertTrue(torch.allclose(from_true_a, torch.tensor([0.5])))
        self.assertTrue(torch.allclose(from_true_b, torch.tensor([-0.25])))
        self.assertTrue(torch.allclose(from_false, torch.tensor([0.25])))

    @patch.object(torchdynamo.config, "capture_scalar_outputs", True)
    def test_simple_condition_nested_no_dynamo(self):
        from_true_a = None
        from_true_b = None
        from_false = None

        def foo():
            nonlocal from_true_a
            nonlocal from_true_b
            nonlocal from_false

            def nested_when_true():
                return -1

            def nested_when_false():
                return 1

            def compute(pred, z):
                def when_true(x):
                    inner_pred = x.item() < 0.5
                    pos_or_neg = cond(
                        inner_pred, nested_when_true, nested_when_false, tuple()
                    )
                    return x * pos_or_neg

                def when_false(x):
                    return x * x

                return cond(pred, when_true, when_false, (z,))

            a = torch.tensor([0.5])
            b = torch.tensor([0.25])

            from_true_a = compute(
                True, a
            )  # True -> when_true -> not less than .5 -> nested_when_false -> .5 * 1
            from_true_b = compute(
                True, b
            )  # True -> when_true -> less than .5 -> nested_when_true -> .25 * -1
            from_false = compute(False, a)  # False -> when_false -> .5 * .5

        foo()
        self.assertTrue(torch.allclose(from_true_a, torch.tensor([0.5])))
        self.assertTrue(torch.allclose(from_true_b, torch.tensor([-0.25])))
        self.assertTrue(torch.allclose(from_false, torch.tensor([0.25])))
