import unittest

import torch
from torch.fx.tensor_type import Dyn

import torchdynamo
import torchdynamo.testing

try:
    import z3  # noqa
    from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import (  # noqa
        evaluate_conditional_with_constraints,
    )

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

skipIfNoZ3 = unittest.skipIf(not HAS_Z3, "no z3")


class TorchDynamoUseCases(unittest.TestCase):
    @skipIfNoZ3
    def test_reshape(self):
        """
        Here, we expect a single graph because
        we proved that the conditional is always false
        """

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: Dyn):
                y = x.view(100)
                tmp = y.size()[0]
                if tmp < 100:
                    return torch.dropout(x, p=0.5, train=False)
                else:
                    return torch.relu(x)

        torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            BasicBlock().forward(torch.rand(50, 2))

        # initial op count is 5
        # intiial frame count is 2
        self.assertEqual(cnts.op_count, 3)
        self.assertEqual(cnts.frame_count, 1)

    @skipIfNoZ3
    def test_fake_condition(self):
        """
        We use a gt node, but it is not actually
        a conditional. Here, we should do nothing.
        """

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: Dyn):
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                view = x.view(-1, getitem)
                lt = arange > view
                masked_fill = x.masked_fill_(lt, 0)
                return masked_fill

        torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            BasicBlock().forward(torch.rand(50, 2))

        # nothing should change here
        self.assertEqual(cnts.op_count, 6)
