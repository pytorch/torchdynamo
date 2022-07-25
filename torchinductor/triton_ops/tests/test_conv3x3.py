import benchmarks_.microbenchmarks.model as model
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
torchinductor.config.triton.convolution = "triton"
torchinductor.config.triton.dense_indexing = True
torch.manual_seed(0)


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
    x = torch.ones((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device='cuda')
    w = torch.ones((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W), 
                    dtype=dtype, device='cuda')
    # bias = torch.randn((KERNEL_N), dtype=dtype, device='cuda')
    bias = None
    if layout == "nhwc":
        x = x.to(memory_format=torch.channels_last)
        # w = w.to(memory_format=torch.channels_last)
    y = torchinductor.triton_ops.conv3x3(
        x, w, bias, stride, padding, dilation, False, (0, 0), groups
    )

    y_correct = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
    print("y", y[0])
    print("y_correct", y_correct[0])
    assert(same(y, y_correct, cos_similarity=True))
    print("passed")

# for layer in (model.resnet50_layers + model.alexnet_layers):
for i in range(1):
    
    dilation = (1, 1)
    groups = 1
    dtype = torch.float32
    BLOCK_M, BLOCK_N, BLOCK_K, NSTAGE, NWARP = 128, 32, 64, 2, 4
    BATCH = 128
    # IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = layer
    # 32, 3, 224, 224, 64, 3, 3, (2, 2), (0, 0), (1, 1), 1, torch.float16, 128, 16, 32, 2, 4
    # 32, 3, 224, 224, 32, 3, 3, (1, 1), (1, 1), (1, 1), 1, torch.float16, 128, 16, 32, 2, 4
    # 32, 128, 32, 32,32, 3, 3, (1, 1), (0, 0), (1, 1), 1, torch.float16, 128, 16, 32, 2, 4
    IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = 8, 8, 16, 3, 3, 32, (1, 1), (0, 0)
    if KERNEL_H != 3 or KERNEL_W != 3:
        continue
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
