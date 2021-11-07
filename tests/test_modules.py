#!/usr/bin/env pytest

import torch
from torch.nn import functional as F

import torchdynamo.testing
from . import test_functions

from torchdynamo._eval_frame import unsupported


class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


class FnMember(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = F.relu

    def forward(self, x):
        x = self.linear1(x)
        if self.activation:
            x = self.activation(x)
        return x


class FnMemberCmp(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = activation

    def forward(self, x):
        x = self.linear1(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.activation is None:
            x = torch.sigmoid(x)
        return x


class SubmoduleExample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x * self.scale


class IsTrainingCheck(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
        super().__init__()
        self.train(False)


class ModuleMethodCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
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


class UnsupportedMethodCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        x = x * self.scale
        return unsupported(x, x)

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        return x + x1


class UnsupportedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x) * self.scale
        return unsupported(x, x)


class UnsupportedModuleCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = UnsupportedModule()

    def forward(self, x):
        return 1 + self.mod(x * 1.5)


class ModuleStaticMethodCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    @staticmethod
    def call_and_scale(scale, mod, x):
        x = mod(x)
        return x * scale

    def forward(self, x):
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleClassMethodCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    @classmethod
    def call_and_scale(cls, scale, mod, x):
        x = mod(x)
        return x * scale

    def forward(self, x):
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleProperty(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.randn(1, 10)

    @property
    def scale_alias(self):
        return self.scale

    def forward(self, x):
        return x * self.scale_alias


class ConstLoop(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.count = 3

    def forward(self, x):
        for i in range(self.count):
            x = torch.sigmoid(self.linear1(x))
        return x


class ViaModuleCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return test_functions.constant3(torch.sigmoid(self.linear1(x)), x)


class IsNoneLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = None
        self.train(True)

    def forward(self, x):
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        return x


class LayerList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModuleList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        for layer in self.layers:
            x = layer(x)

        for layer, val in zip(self.layers, (x, x, x, x)):
            x = layer(x) + val

        for layer, val in zip(self.layers, (1, 2, 3, 4)):
            x = layer(x) + val

        for idx, layer in enumerate(self.layers):
            x = layer(x) * idx

        return x


class TensorList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = (
            torch.randn((1, 10)),
            torch.randn((10, 1)),
            torch.randn((1, 10)),
            torch.randn((10, 1)),
        )

    def forward(self, x):
        for layer in self.layers:
            x = x * layer
        return x


class IntArg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)

    def forward(self, x, offset=1):
        x = F.relu(self.layer1(x)) + offset
        return x


class Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Cfg:
    def __init__(self):
        self.val = 0.5
        self.count = 3


class CfgModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = Cfg()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        for i in range(self.cfg.count):
            x = self.layer(x + self.cfg.val)
        return x


class StringMember(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.mode = "some_string"

    def forward(self, x):
        if self.mode == "some_string":
            return F.relu(self.linear1(x))


def make_test(fn, expected_ops=None):
    def test_fn(self):
        return torchdynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=expected_ops
        )

    return test_fn


class SuperModule(BasicModule):
    def forward(self, x):
        x = super().forward(x)
        return x + 10.0


class NNModuleTests(torchdynamo.testing.TestCase):
    test_seq = make_test(Seq())
    test_basicmodule1 = make_test(BasicModule())
    test_basicmodule2 = make_test(BasicModule())
    test_submodules1 = make_test(SubmoduleExample())
    test_submodules2 = make_test(SubmoduleExample())
    test_modulemethod1 = make_test(ModuleMethodCall())
    test_modulemethod2 = make_test(ModuleMethodCall())
    test_module_static_method = make_test(ModuleStaticMethodCall())
    test_fnmember = make_test(FnMember())
    test_fnmembercmp1 = make_test(FnMemberCmp(F.relu))
    test_fnmembercmp2 = make_test(FnMemberCmp(None))
    test_constloop = make_test(ConstLoop())
    test_istraining1 = make_test(IsTrainingCheck())
    test_istraining2 = make_test(IsTrainingCheck())
    test_iseval1 = make_test(IsEvalCheck())
    test_iseval2 = make_test(IsEvalCheck())
    test_viamodulecall = make_test(ViaModuleCall())
    test_isnonelayer = make_test(IsNoneLayer())
    test_layerlist = make_test(LayerList())
    test_tensorlist = make_test(TensorList(), expected_ops=8)
    test_intarg = make_test(IntArg())
    test_cfgmod = make_test(CfgModule())
    test_stringmember = make_test(StringMember())
    test_modulelist = make_test(ModuleList())
    test_super1 = make_test(SuperModule())

    def test_unsupportedmethod(self):
        m = UnsupportedMethodCall()
        i = torch.randn(10)
        cnt = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnt):
            r = m(i)
        self.assertTrue(torchdynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 5)

    def test_unsupportedmodule(self):
        m = UnsupportedModuleCall()
        i = torch.randn(10)
        cnt = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnt):
            r = m(i)
        self.assertTrue(torchdynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 6)

    # not implemented yet:
    # test_module_class_method = make_test(ModuleClassMethodCall())
    # test_module_property = make_test(ModuleProperty()))
