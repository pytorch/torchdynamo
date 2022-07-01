import torch
import torch.fx

import torchdynamo

inp = torch.randn(2, 2)
l = torch.nn.Linear(2, 2)
net = torch.nn.Sequential(l, l)
x = inp

# torchdynamo.config.debug = True
def foo(k):
    global x
    for idx, m in enumerate(net.named_modules()):
        x = x + m[1](inp) * k


out = torchdynamo.export(foo, 3)
print(x)
print(out)
