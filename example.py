from typing import List

import torch

import torchdynamo


def toy_example(a, b):
   x = a / (torch.abs(a) + 1)
   if b.sum() < 0:
       b = b * -1
   return x * b


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
   print("my_compiler() called with FX graph:")
   gm.graph.print_tabular()
   return gm.forward  # return a python callable


torchdynamo.config.debug = True
torchdynamo.config.trace = True


with torchdynamo.optimize(my_compiler) as enter_ret:
   for i in range(1):
       toy_example(torch.randn(10), torch.randn(10))

"""
try:
   context = torchdynamo.optimize(my_compiler)
   enter_ret = context.__enter__()

   for _ in range(10):
   toy_example(torch.randn(10), torch.randn(10))

finally:
   context.__exit__(..)
"""
