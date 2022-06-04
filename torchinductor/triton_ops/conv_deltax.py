import numpy as np
import torch
import triton
import triton.language as tl

from torchinductor.triton_ops.utils import _extract_strides
from torchinductor.triton_ops.utils import _roundup
from torchinductor.triton_ops.utils import _unpack

from .conv_perf_model import early_config_prune
from .conv_perf_model import estimate_conv_time

# a python implmentation of https://gist.github.com/ptillet/7f20c2006db1a30470675fad05cebdfe
# BLOCK_K has to be fixed because of _delta_x() need to pass BLOCK_K param


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 16, "BLOCK_K": 32}, num_stages=4, num_warps=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        # good for int8
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        # triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        # triton.Config({"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 128}, num_stages=4, num_warps=4),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=4),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=4),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=4, num_warps=2),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=2),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 64}, num_stages=4, num_warps=2),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 64}, num_stages=5, num_warps=2),
    ],
    # the above configs will be evaluated anytime the key changes
    key=[
        "BATCH",
        "IN_C",
        "IN_H",
        "IN_W",
        "KERNEL_N",
        "KERNEL_H",
        "KERNEL_W",
        "OUT_H",
        "OUT_W",
        # parameters of conv
        "stride_h",
        "stride_w",
        "padding_h",
        "padding_w",
        "dilation_h",
        "dilation_w",
        "output_padding_h",
        "output_padding_w",
        "groups",
    ],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_conv_time,
        "top_k": 10,
    },
)
@triton.jit
def _kernel(
    x,
    w,
    bias,
    y,
    # stride of tensor
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wh,
    stride_ww,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    stride_biasn,
    # pointer inc for x
    delta_x_ptr,
    inc_x_ptr,
    # Tensor dimensions
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    OUT_H,
    OUT_W,
    # parameters of conv
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
    # Metaparameters
    ACC_TYPE: tl.constexpr,
    # blocks in different dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # reduction tiling parameter for matmul
    BLOCK_K: tl.constexpr,
    # Super-blocking for better L2 peformance
    GROUP_M: tl.constexpr,
):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid = tl.program_id(0)

    grid_m = (BATCH * OUT_H * OUT_W + BLOCK_M - 1) // BLOCK_M
    grid_n = (KERNEL_N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # offset for output y
    off_y_k = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W
    off_y_w = off_y_hw % OUT_W

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)

    CRS = IN_C * KERNEL_H * KERNEL_W
    # load initial pos for the first BLOCK_K of x
    delta_x_ptrs = delta_x_ptr + off_x_crs
    off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS)
    x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]

    # now delta_x_ptrs is at beginning of (nextk -  currentk) LUT
    # delta_x_ptrs += BLOCK_K
    # // pointers for A look-up table
    # int rklut[TK] = rk % LUT_SIZE;
    # int* padiff[TK] = ADIFF + rklut;
    # int* padelta[TK] = ADELTA + TK + rklut + off_uw * LUT_SIZE + off_uh * LUT_SIZE * upsample_w;
    # int adiff[TK] = *padiff;
    # int adelta[TK] = *padelta;
    rklut = off_x_crs
    diff_x_ptrs = inc_x_ptr + rklut
    delta_x_ptrs = delta_x_ptr + BLOCK_K + rklut
    diff_x = tl.load(diff_x_ptrs, mask=rklut < CRS)
    delta_x = tl.load(delta_x_ptrs, mask=rklut < CRS)

    # x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]
    mask_x = (
        (off_x_n < BATCH)
        & (off_x_h >= 0)
        & (off_x_h < IN_H)
        & (off_x_w >= 0)
        & (off_x_w < IN_W)
    )[:, None] & (off_x_crs < CRS)[None, :]

    # offset for the inital ptr for w
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] * stride_wc + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # ------ load x ------
    matrix_x = tl.load(x_ptrs, mask=mask_x)
    # ------ load w ------
    matrix_w = tl.load(w_ptrs, mask=mask_w)

    # -----------------------------------------------------------
    # allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):

        # ------ matrix multiplication ------
        acc += tl.dot(matrix_x, matrix_w)
        # ------ update ptrs ------
        w_ptrs += BLOCK_K
        # load inc ptr of x, upade x_ptrs
        off_x_crs = crs + tl.arange(0, BLOCK_K)
        # off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS)
        # delta_x_ptrs += BLOCK_K
        # x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        # x_ptrs += BLOCK_K
        x_ptrs += delta_x[None, :]
        # ------ increment X LUT ------
        delta_x_ptrs += diff_x
        delta_x = tl.load(delta_x_ptrs, mask=off_x_crs < CRS)
        diff_x_ptrs += diff_x
        diff_x = tl.load(diff_x_ptrs, mask=off_x_crs < CRS)

        mask_x = (
            (off_x_n < BATCH)
            & (off_x_h >= 0)
            & (off_x_h < IN_H)
            & (off_x_w >= 0)
            & (off_x_w < IN_W)
        )[:, None] & (off_x_crs + BLOCK_K < CRS)[None, :]
        mask_w = (off_x_crs + BLOCK_K < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
        # ------ prefetch ------
        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w)

    # add bias if is not None
    if bias is not None:
        off_bias_k = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_ptrs = bias + off_bias_k * stride_biasn
        mask_bias = off_bias_k < KERNEL_N
        _bias = tl.load(bias_ptrs, mask=mask_bias)
        acc += _bias[None, :]

    acc = acc.to(y.dtype.element_ty)

    # rematerialize -- this saves some registers
    # offset for output y
    off_y_k = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    # consider output padding
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # y ptrs in the block of [BLOCK_M, BLOCK_N]
    y_ptrs = (
        y
        + off_y_n[:, None] * stride_yn
        + off_y_h[:, None] * stride_yh
        + off_y_w[:, None] * stride_yw
        + off_y_k[None, :] * stride_yc
    )

    # out-of-bounds check
    mask_y = (
        (off_y_n < BATCH)[:, None]
        & (off_y_h < OUT_H + output_padding_h)[:, None]
        & (off_y_w < OUT_W + output_padding_w)[:, None]
        & (off_y_k < KERNEL_N)[None, :]
    )

    tl.store(y_ptrs, acc, mask=mask_y)

    return


class _conv_deltax:
    kernel = _kernel

    @staticmethod
    def _delta_x_ptr(
        wc,
        wh,
        ww,
        shape_w,
        dilation_h,
        dilation_w,
        stride_xh,
        stride_xw,
        stride_xc,
        BLOCK_K,
    ):

        # get the order of axes in w, innermost dimension outward
        order = sorted([wc, wh, ww], reverse=True)
        c, h, w = [order.index(x) for x in [wc, wh, ww]]
        # the innermost two dimension of w
        window_contiguous_size = shape_w[order[0]] * shape_w[order[1]]
        # LUT size
        K = _roundup(BLOCK_K, window_contiguous_size)

        # Initial k
        ki = np.arange(BLOCK_K, dtype=np.int32)[None, None, None, :]
        currentk = _unpack(ki, order, shape_w)
        resulti = 0
        resulti += currentk[c] * stride_xc
        resulti += currentk[h] * stride_xh
        resulti += currentk[w] * stride_xw

        # delta k
        k = np.arange(K, dtype=np.int32)[None, None, None, :]
        currentk = _unpack(k, order, shape_w)
        nextk = _unpack(k + BLOCK_K, order, shape_w)

        # Compute memory stride
        result = 0
        result += (nextk[c] - currentk[c]) * stride_xc
        result += (nextk[h] - currentk[h]) * stride_xh
        result += (nextk[w] - currentk[w]) * stride_xw

        return np.concatenate((resulti, result), axis=-1)

    @staticmethod
    def _call(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        layout_x,
        layout_w,
        layout_y,
    ):
        # Q: should we check x, w, bias dtypes?
        device = x.device
        # input shapes
        shape_x = x.shape
        shape_w = w.shape
        shape_bias = bias.shape if bias is not None else None

        # indicies for the layeout
        xn, xc, xh, xw = [layout_x.find(x) for x in "nchw"]
        yn, yc, yh, yw = [layout_y.find(x) for x in "nchw"]
        wn, wc, wh, ww = [layout_w.find(x) for x in "nchw"]

        # out_channel, in_channel, kernel_height, kernel_width
        kernel_size = [shape_w[wh], shape_w[ww]]
        input_size = [shape_x[xh], shape_x[xw]]
        assert (
            not shape_bias or shape_bias[0] == shape_w[wn]
        ), f"bias shape did not match{shape_bias} != {shape_w[wn]}"
        in_channel = shape_w[wc] * groups

        assert shape_x[xc] % groups == 0, "in_channels must be divisible by groups"
        assert shape_w[wn] % groups == 0, "out_channels must be divisible by groups"
        assert (
            shape_x[xc] == in_channel
        ), f"in_channel did not match {shape_x[xc]} != {in_channel}"

        assert (
            len(stride)
            == len(padding)
            == len(dilation)
            == len(output_padding)
            == len(kernel_size)
            == len(input_size)
        )

        # delta_x_ptr does not support dilation > 1...
        assert dilation[0] == 1 and dilation[1] == 1, "only support dilation == 1"

        # output shape
        shape_y = [0] * 4
        shape_y[yn] = shape_x[xn]
        shape_y[yc] = shape_w[wn]
        shape_y[yh] = (
            input_size[0]
            + 2 * padding[0]
            - dilation[0] * (kernel_size[0] - 1)
            - 1
            + stride[0]
        ) // stride[0] + 2 * output_padding[0]
        shape_y[yw] = (
            input_size[1]
            + 2 * padding[1]
            - dilation[1] * (kernel_size[1] - 1)
            - 1
            + stride[1]
        ) // stride[1] + 2 * output_padding[1]

        BATCH = shape_x[xn]
        IN_C = shape_x[xc]
        IN_H = shape_x[xh]
        IN_W = shape_x[xw]
        KERNEL_N = shape_w[wn]
        KERNEL_H = shape_w[wh]
        KERNEL_W = shape_w[ww]
        OUT_H = shape_y[yh]
        OUT_W = shape_y[yw]

        # get strides for tensors
        stride_x = _extract_strides(shape_x)
        stride_w = _extract_strides(shape_w)
        stride_y = _extract_strides(shape_y)
        stride_bias = _extract_strides(shape_bias) if shape_bias else None
        stride_biasn = stride_bias[0] if stride_bias else None

        assert (
            (stride_x[0] >= stride_x[1])
            and (stride_x[1] >= stride_x[2])
            and (stride_x[2] > stride_x[3])
        ), "stride and layout does not match for x"
        assert (
            (stride_w[0] >= stride_w[1])
            and (stride_w[1] >= stride_w[2])
            and (stride_w[2] > stride_w[3])
        ), "stride and layout does not match for w"
        assert (
            (stride_y[0] >= stride_y[1])
            and (stride_y[1] >= stride_y[2])
            and (stride_y[2] > stride_y[3])
        ), "stride and layout does not match for y"

        # allocate output
        y = torch.empty(shape_y, device=device, dtype=x.dtype)
        # allocate tmp
        # WINDOW_SIZE = KERNEL_H * KERNEL_W * IN_C
        # tmp_x = torch.empty((BATCH * OUT_H * OUT_W, WINDOW_SIZE), device=device, dtype=x.dtype)
        # tmp_w = torch.empty((WINDOW_SIZE, KERNEL_N), device=device, dtype=w.dtype)
        # accumulator types
        ACC_TYPE = (
            tl.float32
            if x.dtype in [torch.float16, torch.bfloat16, torch.float32]
            else tl.int32
        )
        # if input activation is channel-last, it"s good to tiling output
        # into block of [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] because of better
        # if stride_x[xc] == 1 and stride_x > 1 and stride_y > 1:
        if True:
            # fixed BLOCK_K
            BLOCK_K = 32
            delta_x = _conv_deltax._delta_x_ptr(
                wc,
                wh,
                ww,
                shape_w,
                dilation[0],
                dilation[1],
                stride_x[xh],
                stride_x[xw],
                stride_x[xc],
                BLOCK_K,
            )
            delta_x = torch.from_numpy(delta_x).cuda()
            # delta increments for x
            inc_x = np.arange(delta_x.shape[-1] - BLOCK_K, dtype=np.int32)
            inc_x = ((inc_x + BLOCK_K) % inc_x.size) - inc_x
            inc_x = torch.from_numpy(inc_x).cuda()
            # print("delta_x", delta_x)
            # print("inc_x", inc_x)

            # launch kernel, 1-dim
            def grid(META):
                return (
                    triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"])
                    * triton.cdiv(KERNEL_N, META["BLOCK_N"]),
                )

            _kernel[grid](
                x,
                w,
                bias,
                y,
                # stride nchw for x,w,y tensor
                stride_x[xn],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                stride_w[wn],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_y[yn],
                stride_y[yc],
                stride_y[yh],
                stride_y[yw],
                stride_biasn,
                # pointer inc for x
                delta_x,
                inc_x,
                # Tensor dimensions
                BATCH,
                IN_C,
                IN_H,
                IN_W,
                KERNEL_N,
                KERNEL_H,
                KERNEL_W,
                OUT_H,
                OUT_W,
                # conv parameters
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                output_padding[0],
                output_padding[1],
                groups,
                # Metaparameters
                ACC_TYPE=ACC_TYPE,
                # BLOCK_M=128, BLOCK_N=32,
                # BLOCK_K=32,
                GROUP_M=8,
                # LUT_SIZE=delta_x.shape[-1],
            )
        # do matrix multiplication
        # else:
        #    _kernel_mm[grid]()
        return y

    @staticmethod
    def forward(
        x,
        w,
        bias,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        transposed=False,
        output_padding=(0, 0),
        groups=1,
        layout_x="nhwc",
        layout_w="nhwc",
        layout_y="nhwc",
    ):
        if groups != 1:
            print(f"Do not support groups = {groups}")
            return
        if transposed:
            print("Do not support transposed")
        return _conv_deltax._call(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            layout_x,
            layout_w,
            layout_y,
        )


conv_deltax = _conv_deltax.forward
