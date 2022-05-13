from typing import List
import torch
import torchdynamo
import threading
import time

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    gm.graph.print_tabular()
    if not hasattr(gm, 'count_voz'):
        setattr(gm, 'count_voz', 1)
    else:    
        setattr(gm, 'count_voz', getattr(gm, 'count_voz') + 1)

    print(f" Total calls: {getattr(gm, 'count_voz') + 1}")
    return gm.forward  # return a python callable


def foo():
    with torchdynamo.optimize(my_compiler, nopython=False):
        toy_example(torch.randn(10), torch.randn(10))

print(f"Main on {threading.get_ident()}")
ts = list()
for _ in range(30):
    foo()
    x = threading.Thread(target=foo)
    x.start()
    x.join()


time.sleep(3)
