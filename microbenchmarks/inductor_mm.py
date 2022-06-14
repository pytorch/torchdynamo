import torch
from benchmark_helper import time_with_torch_timer
import torchdynamo
import torchdynamo.config
import torchinductor.config as config


@torchdynamo.optimize("inductor", nopython=True)
def inductor_aten_mm(a, b):
    return torch.mm(a, b)

@torchdynamo.optimize("inductor", nopython=True)
def inductor_triton_mm(a, b):
    return torch.mm(a, b)

def torch_mm(a, b):
    return torch.matmul(a, b)

def alexnet_matmuls():
    a_shapes = [[128, 9216], [128, 4096], [128, 4096]]
    b_shapes = [[9216, 4096], [4096, 4096], [4096, 1000]]
    for i in range(len(a_shapes)):
        a_shape = a_shapes[i]
        b_shape = b_shapes[i]
        print("Shape:", a_shape, 'x', b_shape)
        a = torch.randn(a_shape, device='cuda', dtype=torch.float16)
        b = torch.randn(b_shape, device='cuda', dtype=a.dtype)

        config.triton.use_mm = False
        inductor_aten_mm(a, b)

        config.triton.use_mm = True
        inductor_triton_mm(a, b)

        time_with_torch_timer(torch_mm, (a, b), string_id="torch mm")

        config.triton.use_mm = False
        time_with_torch_timer(inductor_aten_mm, (a, b), string_id="inductor aten mm")

        config.triton.use_mm = True
        time_with_torch_timer(inductor_triton_mm, (a, b), string_id="inductor triton mm")
      

if __name__ == "__main__":
    alexnet_matmuls()

      




# Results preview
'''
Shape: [128, 9216] x [9216, 4096]
torch mm         mean: 0.0880 ms
inductor aten mm         mean: 0.2163 ms
inductor triton mm       mean: 0.2043 ms
Shape: [128, 4096] x [4096, 4096]
torch mm         mean: 0.0341 ms
inductor aten mm         mean: 0.0997 ms
inductor triton mm       mean: 0.1002 ms
Shape: [128, 4096] x [4096, 1000]
torch mm         mean: 0.0215 ms
inductor aten mm         mean: 0.0310 ms
inductor triton mm       mean: 0.0375 ms
'''
