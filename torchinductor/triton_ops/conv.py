import torch
import triton
import numpy as np

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
            groups,
            # ptr increment for x
            delta_x_c_ptr,
            delta_x_h_ptr,
            delta_x_w_ptr,
            # Metaparameters
            ACC_TYPE: tl.constexpr,
            # blocks in different dimension
            BLOCK_BATCH: tl.constexpr, BLOCK_H: tl.constexpr,
            BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr,
            # tiling parameter for matmul
            TK: tl.constexpr,
            # Super-blocking for better L2 peformance
            GROUP_H: tl.constexpr
            ):
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
    # TODO: consider dilation and group
    x_h_start = y_h_start * stride_h
    x_w_start = y_w_start * stride_w
    x_ptr = x + batch * stride_xn + x_h_start * stride_xh \
        + x_w_start * stride_xw

    # -----------------------------------------------------------
    # do the computation for current block)
    rbhw = tl.arange(0, BLOCK_BATCH * BLOCK_H * BLOCK_W)
    rb = rbhw // (BLOCK_H * BLOCK_W)
    rhw = rbhw % (BLOCK_H * BLOCK_W)
    rh = rhw // BLOCK_W
    rw = rhw % BLOCK_W
    rk = tl.arange(0, BLOCK_K)
    rtk = tl.arange(0, TK)
    WINDOW_SIZE = KERNEL_H * KERNEL_W * IN_C

    # allocate accumulator
    acc = tl.zeros((BLOCK_BATCH * BLOCK_H * BLOCK_W, BLOCK_K), dtype=ACC_TYPE)

    for tki in range(0, WINDOW_SIZE, TK):

        # weight
        # load contigous weight pointers
        w_ptrs = w_ptr + (tki + rtk)[:, None] + rk[None, :] * stride_wn

        mask_w = (tki + rtk < WINDOW_SIZE)[:, None] \
            & (k + rk < KERNEL_N)[None, :]
        # load (TK, BLOCK_K) of weights
        matrix_w = tl.load(w_ptrs, mask=mask_w)

        # x window
        # range of the sliding window
        rx_k_c = tl.load(delta_x_c_ptr + tki + rtk)
        rx_k_h = tl.load(delta_x_h_ptr + tki + rtk)
        rx_k_w = tl.load(delta_x_w_ptr + tki + rtk)
        rx_k = rx_k_h * stride_xh + rx_k_w * stride_xw + rx_k_c * stride_xc

        x_window_ptrs = x_ptr + rb[:, None] * stride_xn \
            + rh[:, None] * stride_h * stride_xh \
            + rw[:, None] * stride_w * stride_xw \
            + rx_k[None, :]
        # check if the sliding window x exceeds bounds
        x_window_in = (x_h_start + rx_k_h < IN_H) & (x_w_start + rx_k_w < IN_W) \
            & (rx_k_c < IN_C) & (tki + rtk < WINDOW_SIZE)
        x_window_in = (tki + rtk < WINDOW_SIZE)
        mask_x_window = (batch + rb < BATCH)[:, None] \
            & (y_h_start + rh < OUT_H)[:, None] \
            & (y_w_start + rw < OUT_W)[:, None] \
            & x_window_in[None, :]
        matrix_x_window = tl.load(x_window_ptrs, mask=mask_x_window)

        # dot product
        # (BLOCK_BATCH * BLOCK_H * BLOCK_W, TK) DOT (TK, BLOCK_K)
        # output shape (BLOCK_BATCH * BLOCK_H * BLOCK_W, BLOCK_K)
        acc += tl.dot(matrix_x_window, matrix_w)

    acc = acc.to(y.dtype.element_ty)

    # y ptrs in the block of [BLOCK_BATCH, BLOCK_H, BLOCK_W, BLOCK_K]
    y_ptrs = y_ptr + rb[:, None] * stride_yn + rh[:, None] * stride_yh \
        + rw[:, None] * stride_yw + rk[None, :] * stride_yc

    mask_y = (batch + rb < BATCH)[:, None] \
        & (y_h_start + rh < OUT_H)[:, None] \
        & (y_w_start + rw < OUT_W)[:, None] \
        & (k + rk < KERNEL_N)[None, :]

    tl.store(y_ptrs, acc, mask=mask_y)

    return


class _conv(torch.autograd.Function):

    @staticmethod
    def _extract_strides(shape):
        rank = len(shape)
        ret = [1] * rank
        for i in range(rank - 1, 0, -1):
            ret[i - 1] = ret[i] * shape[i]
        return ret

    @staticmethod
    def _roundup(x, div):
        return (x + div - 1) // div * div

    @staticmethod
    def _unpack(idx, order, shape):
        _12 = idx  // shape[order[0]]
        _0   = idx   % shape[order[0]]
        _2  = _12 // shape[order[1]]
        _1   = _12  % shape[order[1]]
        return _0, _1, _2

    # for the contigous order of w ptr, what's the corresponding
    # ptr changes for x in a sliding window
    @staticmethod
    def _delta_x_ptr(wc, wh, ww,
                     shape_w):
        # get the order of axes in w, innermost dimension outward
        order = sorted([wc, wh, ww], reverse=True)
        c, h, w = [order.index(x) for x in [wc, wh, ww]]
        window_size = shape_w[order[0]] * shape_w[order[1]] * shape_w[order[2]]
        delta_x_c, delta_x_h, delta_x_w = [], [], []

        for i in range(0, window_size):
            current_k = _conv._unpack(i, order, shape_w)
            delta_x_c.append(current_k[c])
            delta_x_h.append(current_k[h])
            delta_x_w.append(current_k[w])

        delta_x_c = np.asarray(delta_x_c)
        delta_x_h = np.asarray(delta_x_h)
        delta_x_w = np.asarray(delta_x_w)
        return delta_x_c, delta_x_h, delta_x_w

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
        shape_y[yh] = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1)
                       - 1 + stride[0]) // stride[0] + 2 * output_padding[0]
        shape_y[yw] = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1)
                       - 1 + stride[1]) // stride[1] + 2 * output_padding[1]

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

        assert((stride_x[0] >= stride_x[1]) and (stride_x[1] >= stride_x[2])
               and (stride_x[2] > stride_x[3])), "stride and layout does not match for x"
        assert((stride_w[0] >= stride_w[1]) and (stride_w[1] >= stride_w[2])
               and (stride_w[2] > stride_w[3])), "stride and layout does not match for w"
        assert((stride_y[0] >= stride_y[1]) and (stride_y[1] >= stride_y[2])
               and (stride_y[2] > stride_y[3])), "stride and layout does not match for y"

        # allocate output
        y = torch.empty(shape_y, device=device, dtype=x.dtype)
        # accumulator types
        ACC_TYPE = tl.float32 if x.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # if input activation is channel-last, it's good to tiling output
        # into block of [BLOCK_BATCH, BLOCK_K, BLOCK_H, BLOCK_W] because of better
        #if stride_x[xc] == 1 and stride_x > 1 and stride_y > 1:
        if True:
            delta_x_c, delta_x_h, delta_x_w = _conv._delta_x_ptr(wc, wh, ww, shape_w)
            delta_x_c_ptr = torch.from_numpy(delta_x_c).cuda()
            delta_x_h_ptr = torch.from_numpy(delta_x_h).cuda()
            delta_x_w_ptr = torch.from_numpy(delta_x_w).cuda()
            # launch kernel, 3-dim, batch, kernel num, h*w
            grid = lambda META: (triton.cdiv(BATCH, META['BLOCK_BATCH']), triton.cdiv(KERNEL_N, META['BLOCK_K']),
                                 triton.cdiv(OUT_H, META['BLOCK_H']) * triton.cdiv(OUT_W, META['BLOCK_W']))
            _kernel[grid](x, w, bias, y,
                          # stride nchw for x,w,y tensor
                          stride_x[xn], stride_x[xc], stride_x[xh], stride_x[xw],
                          stride_w[wn], stride_w[wc], stride_w[wh], stride_w[ww],
                          stride_y[yn], stride_y[yc], stride_y[yh], stride_y[yw],
                          # Tensor dimensions
                          BATCH, IN_C, IN_H, IN_W,
                          KERNEL_N, KERNEL_H, KERNEL_W,
                          OUT_H, OUT_W,
                          # conv parameters
                          stride[0], stride[1],
                          padding[0], padding[1],
                          dilation[0], dilation[1],
                          output_padding[0], output_padding[1],
                          groups,
                          # ptr increment for x
                          delta_x_c_ptr,
                          delta_x_h_ptr,
                          delta_x_w_ptr,
                          # Metaparameters
                          ACC_TYPE=ACC_TYPE,
                          BLOCK_BATCH=16, BLOCK_H=16, BLOCK_W=16, BLOCK_K=8,
                          TK=8,
                          GROUP_H=1
                          )
        # do matrix multiplication
        # else:
        #    _kernel_mm[grid]()
        return y

    @staticmethod
    def forward(ctx, x, w, bias,
                stride=[1, 1], padding=[0, 0],
                dilation=[1, 1], transposed=False,
                output_padding=[0, 0], groups=1,
                layout_x='nhwc', layout_w='nhwc',
                layout_y='nhwc'):
        if groups != 1:
            print(f"doesn't support groups = {groups}")
        return _conv._call(x, w, bias,
                           stride, padding,
                           dilation, transposed,
                           output_padding, groups,
                           layout_x, layout_w, layout_y)


conv = _conv.apply
