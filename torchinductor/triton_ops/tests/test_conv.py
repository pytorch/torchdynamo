import benchmarks.microbenchmarks.model as model
import itertools
import torch
import triton
import pytest

import torchinductor.triton_ops

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False
from torchdynamo.testing import same


def test_conv(
        # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W,
        KERNEL_N, KERNEL_H, KERNEL_W,
        # parameters of conv
        stride, padding,
        dilation, groups,
        # others,
        dtype,
        # MATA
        BLOCK_M, BLOCK_N, BLOCK_K,
        NSTAGE, NWARP,
        layout,
        provider="conv",
    ):

    torch.manual_seed(0)
    # nuke kernel decorators -- will set meta-parameters manually
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K}
    configs = [triton.Config(kwargs=kwargs, num_warps=NWARP, num_stages=NSTAGE)]
    kernel = torchinductor.triton_ops._conv.kernel
    decorators = kernel.kernel_decorators
    kernel.kernel_decorators = []
    triton.autotune(configs, [])(kernel)
    kernel.kernel_decorators += decorators[1:]

    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device='cuda')
    w = torch.randn((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
                    dtype=dtype, device='cuda')
    bias = torch.randn((KERNEL_N), dtype=dtype, device='cuda')
    if layout == "nhwc":
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    y = torchinductor.triton_ops.conv(
        x, w, bias, stride, padding, dilation, False, (0, 0), groups
    )

    y_correct = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
    # print("y", y[0])
    # print("y_correct", y_correct[0])
    assert(same(y, y_correct, cos_similarity=True))
    print("passed")

for layer in (model.resnet50_layers + model.alexnet_layers):
    
    dilation = (1, 1)
    groups = 1
    dtype = torch.float32
    BLOCK_M, BLOCK_N, BLOCK_K, NSTAGE, NWARP = 128, 128, 64, 2, 4
    BATCH = 128
    IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = layer
    # 32, 3, 224, 224, 64, 3, 3, (2, 2), (0, 0), (1, 1), 1, torch.float16, 128, 16, 32, 2, 4
    # 32, 3, 224, 224, 32, 3, 3, (1, 1), (1, 1), (1, 1), 1, torch.float16, 128, 16, 32, 2, 4
    # 
    # 32, 128, 32, 32,32, 3, 3, (1, 1), (0, 0), (1, 1), 1, torch.float16, 128, 16, 32, 2, 4
        
    test_conv(
            # Tensor dimensions
            BATCH, IN_C, IN_H, IN_W,
            KERNEL_N, KERNEL_H, KERNEL_W,
            # parameters of conv
            stride, padding,
            dilation, groups,
            # others,
            dtype,
            # MATA
            BLOCK_M, BLOCK_N, BLOCK_K,
            NSTAGE, NWARP,
            layout="nchw")
