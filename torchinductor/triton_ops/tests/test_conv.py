import itertools
import torch
import triton
import pytest

import torchinductor.triton_ops

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False


@pytest.mark.parametrize(
    "BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, stride, padding, dilation, groups, dtype,\
        BLOCK_NHW, BLOCK_K, BLOCK_CRS, NSTAGE, NWARP",
    itertools.chain(
        *[
            [
                # 1 warp
                (16, 16, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 16, 16, BLOCK_CRS, NSTAGE, 1),
                (16, 16, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_CRS, NSTAGE, 1),
                (16, 32, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 16, 16, BLOCK_CRS, NSTAGE, 1),
                (16, 64, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_CRS, NSTAGE, 1),
                (16, 64, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 1),
                (16, 16, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_CRS, NSTAGE, 1),
                (16, 16, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 1),
                (16, 32, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_CRS, NSTAGE, 1),
                (16, 64, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 1),
                (16, 64, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 1),

                # 2 wrap
                (16, 128, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_CRS, NSTAGE, 2),
                (16, 128, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 2),
                (16, 256, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 2),
                (16, 256, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 2),
                (16, 64, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 2),
                (16, 64, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 2),
                (16, 64, 16, 16, 32, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 2),
                (16, 64, 16, 16, 32, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 32, BLOCK_CRS, NSTAGE, 2),
                (16, 16, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_CRS, NSTAGE, 2),
                (16, 16, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 2),

                # 4 wrap
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 256, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 256, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 32, 32, 32, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 32, 32, 32, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 32, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 32, 32, 64, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 32, 32, 64, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 32, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 32, 32, 64, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 64, BLOCK_CRS, NSTAGE, 4),
                (16, 16, 64, 64, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 16, 64, 64, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 16, 64, 64, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 8, 128, 128, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 8, 128, 128, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 16, BLOCK_CRS, NSTAGE, 4),

                # 8 wrap
                (16, 256, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 8),
                (16, 256, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 16, BLOCK_CRS, NSTAGE, 8),
                (16, 128, 64, 64, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 8),
                (16, 128, 64, 64, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 16, BLOCK_CRS, NSTAGE, 8),
                (16, 128, 128, 128, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_CRS, NSTAGE, 8),
                (16, 128, 128, 128, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 16, BLOCK_CRS, NSTAGE, 8),
                (16, 128, 32, 32, 128, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 64, BLOCK_CRS, NSTAGE, 8),
                (16, 128, 32, 32, 128, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 128, BLOCK_CRS, NSTAGE, 8),

            ] for DTYPE in [torch.float32]
            for BLOCK_CRS in [32]
            for NSTAGE in [2, 3, 4]
        ],
        # kernel, stride, padding, dilation
        *[
            [
                # kernel > 1
                (16, 128, 16, 16, 16, 2, 2, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 3, 3, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 4, 4, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 5, 5, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 7, 7, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),

                # stride > 1
                (16, 128, 16, 16, 16, 1, 1, (2, 2), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 1, 1, (3, 3), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 1, 1, (4, 4), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 1, 1, (5, 5), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),

                # padding > 0
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (1, 1), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (2, 2), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (3, 3), (1, 1), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),

                # dilation > 1
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (1, 1), (2, 2), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (1, 1), (3, 3), 1, DTYPE, 64, 16, BLOCK_CRS, NSTAGE, 4),

            ] for DTYPE in [torch.float32]
            for BLOCK_CRS in [32]
            for NSTAGE in [2, 3, 4]
        ],
    ),
)
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
        BLOCK_NHW, BLOCK_K, BLOCK_CRS,
        NSTAGE, NWARP):

    conv2d_layer = torch.nn.Conv2d(IN_C, KERNEL_N, (KERNEL_H, KERNEL_W),
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=groups)

    torch.manual_seed(0)
    # nuke kernel decorators -- will set meta-parameters manually
    kwargs = {'BLOCK_NHW': BLOCK_NHW, 'BLOCK_K': BLOCK_K, 'BLOCK_CRS': BLOCK_CRS}
    configs = [triton.Config(kwargs=kwargs, num_warps=NWARP, num_stages=NSTAGE)]
    kernel = torchinductor.triton_ops._conv.kernel
    decorators = kernel.kernel_decorators
    kernel.kernel_decorators = []
    triton.autotune(configs, [])(kernel)
    kernel.kernel_decorators += decorators[1:]

    # allocate inputs
    x = torch.randn((BATCH, IN_H, IN_W, IN_C), dtype=dtype, device='cuda')
    w = torch.randn((KERNEL_N, KERNEL_H, KERNEL_W, IN_C // groups),
                    dtype=dtype, device='cuda')
    bias = torch.randn((KERNEL_N), dtype=dtype, device='cuda')
    y = torchinductor.triton_ops.conv(x, w, bias, stride, padding, dilation,
                                      False, (0, 0), groups)

    conv2d_layer.weight.data = w.permute((0, 3, 1, 2))
    conv2d_layer.bias.data = bias
    y_correct = conv2d_layer(x.permute((0, 3, 1, 2))).permute((0, 2, 3, 1))
    triton.testing.assert_almost_equal(y, y_correct, decimal=1)

# BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, stride, padding, dilation, groups, dtype, BLOCK_NHW, BLOCK_K, BLOCK_CRS, NSTAGE, NWARP = \
#     16, 128, 16, 16, 16, 4, 4, (1, 1), (0, 0), (1, 1), 1, torch.float32, 64, 16, 32, 2, 4
# test_conv(
#         # Tensor dimensions
#         BATCH, IN_C, IN_H, IN_W,
#         KERNEL_N, KERNEL_H, KERNEL_W,
#         # parameters of conv
#         stride, padding,
#         dilation, groups,
#         # others,
#         dtype,
#         # MATA
#         BLOCK_NHW, BLOCK_K, BLOCK_CRS,
#         NSTAGE, NWARP)