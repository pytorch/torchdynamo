import torch
import torch.nn as nn
import numpy as np
from torch.nn import *
import torchdynamo
import copy

from torchdynamo.testing import rand_strided
from torchdynamo.testing import collect_results
from torchdynamo.utils import clone_inputs
from torchdynamo.utils import same
from torch.utils._pytree import tree_map


import functorch.compile
functorch.compile.config.debug_joint = True


def cast_to(dtype, model, inputs):
    # cast model and inputs to fp16
    model = model.to(dtype)

    inputs = tree_map(
        lambda x: x.to(dtype)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x,
        inputs,
    )
    return model, inputs


def cast_to_fp64(model, inputs):
    return cast_to(torch.float64, model, inputs)



def forward_and_backward_pass(mod, inputs, collect_outputs=True):
    cloned_inputs = clone_inputs(inputs)
    mod.zero_grad(True)
    pred = mod(*cloned_inputs)
    if isinstance(pred, tuple):
        pred = pred[0]
    loss = pred.sum()
    loss.backward()
    if collect_outputs:
        return collect_results(mod, pred, loss, [])
    return None


class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _stack0_0_ : torch.Tensor):
        np.random.seed(0)
        contiguous = _stack0_0_.contiguous();  _stack0_0_ = None
        view = contiguous.view(1, 1024, -1);  contiguous = None
        return (view,)

class Bar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_output_dense = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, _stack0_0_ : torch.Tensor):
        self_output_dense = self.self_output_dense(_stack0_0_);  _stack0_0_ = None
        return (self_output_dense,)

class BigBirdAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.foo = Foo()
        self.bar = Bar()

    def forward(self, a, b):
        temp = self.foo(a)
        return self.bar(temp[0])


torch.manual_seed(1337)
a = rand_strided(torch.Size([1, 1024, 12, 64]), (786432, 64, 65536, 1), torch.float32, "cuda").requires_grad_(True)
b = rand_strided(torch.Size([1, 1024, 768]), (786432, 768, 1), torch.float32, "cuda").requires_grad_(True)
inputs = [a, b]

mod = BigBirdAttention().to(device="cuda")
opt_mod = torchdynamo.optimize("aot_eager")(copy.deepcopy(mod))

# mod_fp64, inputs_fp64 = cast_to_fp64(copy.deepcopy(mod), clone_inputs(inputs))

ref = forward_and_backward_pass(mod, inputs)
res = forward_and_backward_pass(opt_mod, inputs)
# assert same(ref, ref1, None)
# ref_fp64 = forward_and_backward_pass(mod_fp64, inputs_fp64)

# assert same(ref, res, ref_fp64)
# assert same(ref, res, None)
