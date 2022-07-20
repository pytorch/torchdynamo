from benchmarks.microbenchmarks import model as model
import torch
import torchdynamo
import torchinductor.config
from torchdynamo.testing import same
from torchdynamo.testing import rand_strided

torchinductor.config.debug = True
torchinductor.config.triton.dense_indexing = True
# torchinductor.config.triton.cudagraphs = False
torch.manual_seed(0)

class Func(object):
    @torchdynamo.optimize("inductor")
    def mm(x, w):
        y =  torch.mm(x, w)
        return y

    # mm+bias
    @torchdynamo.optimize("inductor")
    def mm_add(x, w, bias):
        y =  torch.mm(x, w)
        return y + bias

    # relu(mm)
    @torchdynamo.optimize("inductor")
    def mm_relu(x, w):
        y =  torch.mm(x, w)
        return torch.relu(y)

    # relu(mm+bias)
    @torchdynamo.optimize("inductor")
    def mm_add_relu(x, w, bias):
        y =  torch.mm(x, w)
        y += bias
        return torch.relu(y)


def test():
    dtype=torch.float32
    a = torch.ones((128, 9216), device='cuda', dtype=dtype) / 10
    b = torch.ones((9216, 4096), device='cuda', dtype=dtype) / 10

    
    mm_fusion = getattr(Func, f"mm_relu")
    # torchinductor from template
    torchinductor.config.triton.use_mm = True
    torchinductor.metrics.reset()
    y = mm_fusion(a, b)
    assert torchinductor.metrics.generated_kernel_count == 1, f"codegen #kernel != 1"
    # baseline
    # reset to force code gen new python code
    torchdynamo.reset()
    torchinductor.config.triton.use_mm = False
    y_correct = mm_fusion(a, b)
    print("y", y[0,0:128])
    print("y_correct", y_correct[0,0:128])
    assert(same(y, y_correct, cos_similarity=True))

shapes = [
    # alexnet
    ([128, 9216], [9216, 4096]),
]

print(f"testing correctness of mm+relu")
test()
print("passed")
