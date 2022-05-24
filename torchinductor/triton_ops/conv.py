import torch
import triton

import triton.language as tl


@triton.jit
def _kernel(x, w, bias, y,
            # stride of tensor
            stride_xn, stride_xc, stride_xh, stride_xw,
            stride_wn, stride_wc, stride_wh, stride_ww,
            stride_yn, stride_yc, stride_yh, stride_yw,
            # Tensor dimensions
            BATCH, IN_C, IN_H, IN_W,
            KERNEL_N, KERNEL_H, KERNEL_W,
            OUT_H, OUT_W,
            # parameters of conv
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            output_padding_h, output_padding_w,
            groups, ACC_TYPE,
            # Metaparameters
            # blocks in different dimension
            BLOCK_BATCH: tl.constexpr, BLOCK_H: tl.constexpr,
            BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
            # Super-blocking for better L2 peformance
            GROUP_H: tl.constexpr):
    '''
    each program instance computes a [BLOCK_BATCH, BLOCK_K, BLOCK_H, BLOCK_W] block of y
    '''
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    # program-id for current grid
    pid_batch = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_hw = tl.program_id(2)
    # number of program ids along different axes
    grid_h = tl.cdiv(OUT_H, BLOCK_H)
    grid_w = tl.cdiv(OUT_W, BLOCK_W)

    batch = pid_batch * BLOCK_BATCH
    k = pid_k * BLOCK_K

    # re-order program ID for better L2 performance
    width = GROUP_H * grid_w
    group_id = pid_hw // width
    group_size = min(grid_h - group_id * GROUP_H, GROUP_H)
    # *within groups*, programs are ordered in a column-major order
    # row-id of the program in the *launch grid*
    pid_h = group_id * GROUP_H + (pid_hw % group_size)
    # col-id of the program in the *launch grid*
    pid_w = (pid_hw % width) // (group_size)

    # get starting y_ptr for current batch, h and w
    y_h_start = pid_h * BLOCK_H
    y_w_start = pid_w * BLOCK_W
    y_ptr = y + batch * stride_yn + y_h_start * stride_yh \
        + y_w_start * stride_yw + k * stride_yc
    # get starting w_ptr for current k
    w_ptr = w + k * stride_wn
    # get starting x_ptr for the window that correspondes to the y_ptr output
    x_h_start = (y_h_start - 1) * stride_h + 1 - 2 * padding_h + dilation_h * (KERNEL_H - 1)
    x_w_start = (y_w_start - 1) * stride_w + 1 - 2 * padding_w + dilation_w * (KERNEL_W - 1)
    x_ptr = x + batch * stride_xc + x_h_start * stride_xh \
        + x_w_start * stride_xw

    # -----------------------------------------------------------
    # do the computation for current block
    r_windowh = tl.arange(0, KERNEL_H)
    r_windoww = tl.arange(0, KERNEL_W)
    rb = tl.arange(0, BLOCK_BATCH)
    rh = tl.arange(0, BLOCK_H)
    rw = tl.arange(0, BLOCK_W)
    rc = tl.arange(0, BLOCK_C)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_BATCH * BLOCK_H * BLOCK_W, BLOCK_K), dtype=ACC_TYPE)
    for ci in range(0, IN_C, BLOCK_C):
        w_ptrs = w_ptr + rk[:, None, None, None] * stride_wn + r_windowh[None, :, None, None] * stride_wh \
            + r_windoww[None, None, :, None] * stride_ww + (ci + rc[None, None, None, :]) * stride_wc \

        mask_w = (k + rk < KERNEL_N)[:, None, None, None] \
            & (ci + rc < IN_C)[None, None, None, :] 

        matrix_w = tl.load(w_ptrs, mask=mask_w).reshape((BLOCK_K, KERNEL_H * KERNEL_W * BLOCK_C))

        x_ptrs = tl.empty((BLOCK_BATCH, BLOCK_H, BLOCK_W, KERNEL_H * KERNEL_W * BLOCK_C))
        mask_x = tl.empty((BLOCK_BATCH, BLOCK_H, BLOCK_W, KERNEL_H * KERNEL_W * BLOCK_C), dtype=torch.bool)
        for hi in range(BLOCK_H):
            for wi in range(BLOCK_W):
                x_window_start_ptr = x_ptr + hi * stride_h * stride_xh \
                    + wi * stride_w * stride_xw
                x_window_ptrs = x_window_start_ptr + rb[:, None, None, None] * stride_xn \
                    + r_windowh[None, :, None, None] * stride_xh \
                    + r_windoww[None, None, :, None] * stride_xw \
                    + (ci + rc[None, None, None, :]) * stride_xc
                x_ptrs[:, hi, wi, :] = x_window_ptrs.reshape((BLOCK_BATCH, KERNEL_H * KERNEL_W * BLOCK_C))
                mask_x_window = (batch + rb < BATCH)[:, None, None, None] \
                    & (x_h_start + hi + r_windowh < IN_H)[None, :, None, None] \
                    & (x_w_start + wi + r_windoww < IN_W)[None, None, :, None] \
                    & (ci + rc < IN_C)[None, None, None, :]
                mask_x[:, hi, wi, :] = tl.reshape(mask_x_window, (BLOCK_BATCH, KERNEL_H * KERNEL_W * BLOCK_C))

        matrix_x = tl.load(x_ptrs, mask=mask_x).reshape(BLOCK_BATCH * BLOCK_H * BLOCK_W, KERNEL_H * KERNEL_W * BLOCK_C)

        # dot of matrix_x and matrix_w
        acc += tl.dot(matrix_x, matrix_w)

    acc = tl.reshape(acc, (BLOCK_BATCH, BLOCK_H, BLOCK_W, BLOCK_K))

    y_ptrs = y_ptr + rb[:, None, None, None] * stride_yn + rh[None, :, None, None] * stride_yh \
        + rw[None, None, :, None] * stride_yw + rk[None, None, None, :] * stride_yc

    mask_y = (batch + rb < BATCH)[:, None, None, None] \
        & (pid_h * BLOCK_H + rh < OUT_H)[None, :, None, None] \
        & (pid_w * BLOCK_W + rw < OUT_W)[None, None, :, None] \
        & (k + rk < KERNEL_N)[None, None, None, :] \

    tl.store(y_ptrs, acc, mask=mask_y)


class _conv(torch.autograd.Function):
    kernel = _kernel

    @staticmethod
    def _extract_strides(shape):
        rank = len(shape)
        ret = [1] * rank
        for i in range(rank - 1, 0, -1):
            ret[i - 1] = ret[i] * shape[i]
        return ret

    @staticmethod
    def _call(x, w, bias,
              stride, padding,
              dilation, transposed,
              output_padding, groups,
              layout_x, layout_w, layout_y):
        # Q: should we check x, w, bias dtypes?
        device = x.device
        # input shapes
        shape_x = x.shape
        shape_w = w.shape
        shape_bias = bias.shape if bias else None

        # indicies for the layeout
        xn, xc, xh, xw = [layout_x.find(x) for x in 'nchw']
        yn, yc, yh, yw = [layout_y.find(x) for x in 'nchw']
        wn, wc, wh, ww = [layout_w.find(x) for x in 'nchw']

        # out_channel, in_channel, kernel_height, kernel_width
        kernel_size = [shape_w[yh], shape_w[yw]]
        input_size = [shape_x[xh], shape_x[xw]]
        assert not shape_bias or shape_bias == shape_w[wn], \
            f"bias shape did not match{shape_bias} != {shape_w[wn]}"
        in_channel = shape_w[wc] * groups

        assert(shape_x[xc] == in_channel), \
            f"in_channel did not match {shape_x[xc]} != {in_channel}"

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
        shape_y[yh] = (input_size[yh] + 2 * padding[yh] - dilation[yh] * (kernel_size[yh] - 1)
                       - 1 + stride[yh]) // stride[yh] + 2 * output_padding[yh]
        shape_y[yw] = (input_size[yw] + 2 * padding[yw] - dilation[yw] * (kernel_size[yw] - 1)
                       - 1 + stride[yw]) // stride[yw] + 2 * output_padding[yw]

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
        stride_x = _conv._extract_strides(shape_x)
        stride_w = _conv._extract_strides(shape_w)
        stride_y = _conv._extract_strides(shape_y)

        # allocate output
        y = torch.empty(shape_y, device=device, dtype=x.dtype)
        # accumulator types
        ACC_TYPE = tl.float32 if x.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # launch kernel, 3-dim, batch, kernel num, h*w
        grid = lambda META: (tl.cdiv(BATCH, META['BLOCK_BATCH']), tl.cdiv(KERNEL_N, META['BLOCK_K']),
                             tl.cdiv(OUT_H, META['BLOCK_H']) * tl.cdiv(OUT_W, META['BLOCK_W']))
        return _kernel[grid](x, w, bias, y,
                             # stride nchw for x,w,y tensor
                             stride_x[xn], stride_x[xc], stride_x[xh], stride_x[xw],
                             stride_w[wn], stride_w[wc], stride_w[wh], stride_w[ww],
                             stride_y[yn], stride_y[yc], stride_y[yh], stride_y[xw],
                             # Tensor dimensions
                             BATCH, IN_C, IN_H, IN_W,
                             KERNEL_N, KERNEL_H, KERNEL_W,
                             OUT_H, OUT_W,
                             # conv parameters
                             stride[0], stride[1],
                             padding[0], padding[1],
                             dilation[0], dilation[1],
                             output_padding[0], output_padding[1],
                             groups, ACC_TYPE
                             )

    @staticmethod
    def forward(ctx, x, w, bias,
                stride, padding,
                dilation, transposed,
                output_padding, groups,
                layout_x='nhwc', layout_w='nhwc',
                layout_y='nhwc'):
        return _conv._call(x, w, bias,
                           stride, padding,
                           dilation, transposed,
                           output_padding, groups,
                           layout_x, layout_w, layout_y)


conv = _conv.apply
