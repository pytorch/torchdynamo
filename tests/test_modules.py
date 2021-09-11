#!/usr/bin/env pytest
import unittest

import torch
from torch.nn import functional as F

import torchdynamo
from torchdynamo import eval_frame
from torchdynamo.symbolic_convert import convert_frame_assert, dummy_fx_compile
from torchdynamo.testing import same

torchdynamo.symbolic_convert.DEBUG = True


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


class FnMember(torch.nn.Module):
    def __init__(self):
        super(FnMember, self).__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = F.relu

    def forward(self, x):
        x = self.linear1(x)
        if self.activation:
            x = self.activation(x)
        return x


class FnMemberCmp(torch.nn.Module):
    def __init__(self, activation):
        super(FnMemberCmp, self).__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = activation

    def forward(self, x):
        x = self.linear1(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.activation is None:
            x = F.sigmoid(x)
        return x


class SubmoduleExample(torch.nn.Module):
    def __init__(self):
        super(SubmoduleExample, self).__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x * self.scale


class IsTrainingCheck(torch.nn.Module):
    def __init__(self):
        super(IsTrainingCheck, self).__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.train(True)

    def forward(self, x):
        if self.training:
            mod = self.linear1
        else:
            mod = self.linear2
        return F.relu(mod(x))


class IsEvalCheck(IsTrainingCheck):
    def __init__(self):
        super(IsEvalCheck, self).__init__()
        self.train(False)


class ModuleMethodCall(torch.nn.Module):
    def __init__(self):
        super(ModuleMethodCall, self).__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        return x * self.scale

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        x2 = self.call_and_scale(self.layer2, x)
        return x1 + x2


def make_test(fn):
    def test_fn(self):
        torch.fx.symbolic_trace(fn).graph.print_tabular()
        args1 = [torch.randn(10, 10)]
        args2 = [torch.randn(10, 10)]
        correct1 = fn(*args1)
        correct2 = fn(*args2)
        with eval_frame.optimize(convert_frame_assert(dummy_fx_compile)):
            val1a = fn(*args1)
            val2a = fn(*args2)
            val1b = fn(*args1)
            val2b = fn(*args2)
        self.assertTrue(same(val1a, correct1))
        self.assertTrue(same(val1b, correct1))
        self.assertTrue(same(val2a, correct2))
        self.assertTrue(same(val2b, correct2))

    return test_fn


class NNModuleTests(unittest.TestCase):
    test_basicmodule1 = make_test(BasicModule())
    test_basicmodule2 = make_test(BasicModule())
    test_submodules1 = make_test(SubmoduleExample())
    test_submodules2 = make_test(SubmoduleExample())
    test_modulemethod1 = make_test(ModuleMethodCall())
    test_modulemethod2 = make_test(ModuleMethodCall())
    test_fnmember = make_test(FnMember())
    test_fnmembercmp = make_test(FnMemberCmp(F.relu))
    test_fnmembercmp = make_test(FnMemberCmp(None))

    # TODO(jansel): these ones aren't implemented yet
    # test_istraining1 = make_test(IsTrainingCheck())
    # test_istraining2 = make_test(IsTrainingCheck())
    # test_iseval1 = make_test(IsEvalCheck())
    # test_iseval2 = make_test(IsEvalCheck())

    # TODO(jansel): we should make sure to expand nn.Sequential
