import torch
import triton
import triton.language as tl

from .autotune import conv_heuristics


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@conv_heuristics(pre_hook=init_to_zero("y"))
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
    SPLIT_K: tl.constexpr,
):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_window = tl.program_id(2)

    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W
    off_y_w = off_y_hw % OUT_W

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_inc = tl.arange(0, BLOCK_K)

    # load inc ptr of x, upade x_ptrs
    delta_xh = pid_window // KERNEL_W
    delta_xw = pid_window % KERNEL_W
    delta_xc = off_x_inc
    # c, h, w: IN_C, KERNEL_H, KERNEL_W
    off_x_crs_unpacked = (
        delta_xh * dilation_h * stride_xh
        + delta_xw * dilation_w * stride_xw
        + delta_xc * stride_xc
    )
    x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]

    mask_x = (
        (off_x_n < BATCH)[:, None]
        & (off_x_inc < IN_C)[None, :]
        & (off_x_h + (delta_xh * dilation_h) >= 0)[:, None]
        & (off_x_h + (delta_xh * dilation_h) < IN_H)[:, None]
        & (off_x_w + (delta_xw * dilation_w) >= 0)[:, None]
        & (off_x_w + (delta_xw * dilation_w) < IN_W)[:, None]
    )

    # offset for the inital ptr for w
    off_w_crs = delta_xh * stride_wh + delta_xw * stride_ww + delta_xc * stride_wc
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    # tell triton not to vectorize, otherwise misaligned address error
    # w_ptrs = tl.multiple_of(w_ptrs, [1, 1])
    mask_w = (off_x_inc < IN_C)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # ------ load x ------
    matrix_x = tl.load(x_ptrs, mask=mask_x)  # BLOCK_M * crs_mul_of_KERNEL
    # ------ load w ------
    matrix_w = tl.load(w_ptrs, mask=mask_w)  # crs_mul_of_KERNEL * BLOCK_N

    # -----------------------------------------------------------
    # allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    # acc += tl.dot(matrix_x, matrix_w)
    for inc in range(0, IN_C, BLOCK_K):

        # ------ matrix multiplication ------
        acc += tl.dot(matrix_x, matrix_w)
        # ------ update ptrs ------
        off_x_inc = inc + BLOCK_K + tl.arange(0, BLOCK_K)
        x_ptrs += BLOCK_K * stride_xc
        w_ptrs += BLOCK_K * stride_wc

        mask_x = (
            (off_x_n < BATCH)[:, None]
            & (off_x_inc < IN_C)[None, :]
            & (off_x_h[:, None] + (delta_xh * dilation_h) >= 0)
            & (off_x_h[:, None] + (delta_xh * dilation_h) < IN_H)
            & (off_x_w[:, None] + (delta_xw * dilation_w) >= 0)
            & (off_x_w[:, None] + (delta_xw * dilation_w) < IN_W)
        )
        mask_w = (off_x_inc < IN_C)[:, None] & (off_w_k < KERNEL_N)[None, :]
        # ------ prefetch ------
        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w)

    # add bias if is not None
    # if bias is not None and pid_window == 0:
    #     off_bias_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    #     bias_ptrs = bias + off_bias_k * stride_biasn
    #     mask_bias = off_bias_k < KERNEL_N
    #     _bias = tl.load(bias_ptrs, mask=mask_bias)
    #     acc += _bias[None, :]

    acc = acc.to(y.dtype.element_ty)

    # rematerialize -- this saves some registers
    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
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

    if SPLIT_K == 1:
        tl.store(y_ptrs, acc, mask=mask_y)
    else:
        tl.atomic_add(y_ptrs, acc, mask=mask_y)

    return


class _conv_split:
    kernel = _kernel

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
    ):
        # Q: should we check x, w, bias dtypes?
        device = x.device
        # input shapes
        shape_x = x.shape
        shape_w = w.shape
        shape_bias = bias.shape if bias is not None else None

        # indicies for the layeout
        xn, xc, xh, xw = 0, 1, 2, 3
        yn, yc, yh, yw = 0, 1, 2, 3
        wn, wc, wh, ww = 0, 1, 2, 3

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
        # assert kernel_size == [3, 3], "should be _split kernel"

        assert (
            len(stride)
            == len(padding)
            == len(dilation)
            == len(output_padding)
            == len(kernel_size)
            == len(input_size)
        )

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

        # allocate output
        y = torch.empty(shape_y, device=device, dtype=x.dtype)

        # get strides for tensors
        stride_x = x.stride()
        stride_w = w.stride()
        stride_bias = bias.stride() if shape_bias else None
        stride_biasn = stride_bias[0] if stride_bias else None

        # output layout should be the same as x
        if stride_x[xc] < stride_x[xh] and stride_x[xc] < stride_x[xw]:
            y = y.to(memory_format=torch.channels_last)
        stride_y = y.stride()

        # accumulator types
        ACC_TYPE = (
            tl.float32
            if x.dtype in [torch.float16, torch.bfloat16, torch.float32]
            else tl.int32
        )

        # launch kernel, 2-dim, batch*h*w, kernel
        def grid(META):
            return (
                triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"]),
                triton.cdiv(KERNEL_N, META["BLOCK_N"]),
                # split over sliding window KERNEL_H * KERNEL_W; atomic_add
                KERNEL_H * KERNEL_W,
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
            # BLOCK_M=128,
            # BLOCK_N=32,
            # BLOCK_K=BLOCK_K,
            SPLIT_K=KERNEL_H * KERNEL_W,
        )
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
    ):
        if groups != 1:
            print(f"Do not support groups = {groups}")
            return
        if transposed:
            print("Do not support transposed")
        return _conv_split._call(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )


conv_split = _conv_split.forward
