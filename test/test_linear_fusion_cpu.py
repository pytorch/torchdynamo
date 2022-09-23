import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_quantization import NodeSpec as ns, QuantizationTestCase

import torchdynamo        
from torchdynamo.testing import same
from torchinductor.overrides import fuse_fx, LinearEltwise

torchdynamo.config.raise_on_backend_error = False

def _eltwise_list():
    eltwise_list = [
        nn.ReLU(),
        nn.Sigmoid(),
        nn.Tanh(),
        nn.Hardswish(),
        nn.LeakyReLU(0.1, inplace=False),
        nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False),
        nn.GELU(approximate="none"),
        nn.GELU(approximate="tanh"),
    ]
    return eltwise_list

class TestFuseFx(QuantizationTestCase):
    def _test_linear_eltwise(self, eltwise_fn, bias, input_shape):
        class M(nn.Module):
            def __init__(self, eltwise_fn, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias, **kwargs)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.linear(x)
                x = self.eltwise(x)
                return x    

        mod = M(eltwise_fn, input_shape[-1], 10, bias).eval()          

        @torchdynamo.optimize("inductor")
        def fn(x):
            return mod(x)

        v = torch.randn(input_shape)

        fused_gm = fuse_fx(torch.fx.symbolic_trace(mod), [v])
        expected_nodes = [
            ns.call_module(LinearEltwise)
        ]
        self.checkGraphModuleNodes(
            fused_gm,
            expected_node_list=expected_nodes)   

        result = fn(v)
        assert same(result, mod(v))

    def test_linear_eltwise(self):
        for eltwise_fn in _eltwise_list():
            for input_shape in [
                [2, 3, 10],
                [2, 10]]:
                for bias in [
                    True,
                    False]:
                    self._test_linear_eltwise(eltwise_fn, bias, input_shape)

if __name__ == '__main__':
    run_tests()