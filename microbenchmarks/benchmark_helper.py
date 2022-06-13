import torch
import time
from torch.utils.benchmark import Timer

def time_with_torch_timer(fn, args, string_id="function", kwargs={}, 
                            iters=100, mean=True, median=False):
    env = {"args": args, "kwargs": kwargs, "fn": fn}
    fn_call = "fn(*args, **kwargs)"
    
    # Measure end-to-end time
    timer = Timer(stmt=f"{fn_call}", globals=env)
    
    tt = timer.timeit(iters)
    if mean:
        print(f"{string_id}\t mean: {tt.mean * 1000:.4f} ms")
    if median:
        print(f"{string_id}\t median: {tt.median * 1000:.4f} ms")
