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
            delta_x_ptr,
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
    x_h_start = (y_h_start - 1) * stride_h + 1 - 2 * padding_h + dilation_h * (KERNEL_H - 1)
    x_w_start = (y_w_start - 1) * stride_w + 1 - 2 * padding_w + dilation_w * (KERNEL_W - 1)
    x_ptr = x + batch * stride_xc + x_h_start * stride_xh \
        + x_w_start * stride_xw

    # -----------------------------------------------------------
    # do the computation for current block
    r_windowh = tl.arange(0, KERNEL_H)
    r_windoww = tl.arange(0, KERNEL_W)
    rb = tl.arange(0, BLOCK_BATCH)
    # rh = tl.arange(0, BLOCK_H)
    # rw = tl.arange(0, BLOCK_W)
    # rc = tl.arange(0, BLOCK_C)
    rk = tl.arange(0, BLOCK_K)
    rtk = tl.arange(0, TK)
    WINDOW_SIZE = KERNEL_H * KERNEL_W * IN_C

    # print("loop", BATCH, BLOCK_H, BLOCK_W, IN_C, BLOCK_C)
    for hi in range(BLOCK_H):
        for wi in range(BLOCK_W):
            y_ptrs = y_ptr + rb[:, None] * stride_yn + hi * stride_yh \
                + wi * stride_yw + rk[None, :] * stride_yc

            mask_y = (batch + rb < BATCH)[:, None] \
                & (y_h_start + hi < OUT_H) \
                & (y_w_start + wi < OUT_W) \
                & (k + rk < KERNEL_N)[None, :]

            # allocate accumulator
            acc = tl.zeros((BLOCK_BATCH, BLOCK_K), dtype=ACC_TYPE) + 1
            
            for tki in range(0, WINDOW_SIZE, TK):
                acc = acc.to(y.dtype.element_ty)
                # weight
                # assume kn is not the last channel, then the TK elements in weights are contigous
                '''
                w_ptrs = w_ptr + (tki + rtk)[:, None] + rk[None, :] * stride_wn

                mask_w = (tki + rtk < WINDOW_SIZE)[:, None] \
                    & (k + rk < KERNEL_N)[None, :]
                # load (TK, BLOCK_K) of weights
                matrix_w = tl.load(w_ptrs, mask=mask_w)

                # x window
                x_window_start_ptr = x_ptr + hi * stride_h * stride_xh \
                    + wi * stride_w * stride_xw
                delta_x_ptr_window = tl.load(delta_x_ptr + tki + rtk)
                x_window_ptrs = x_window_start_ptr + rb[:, None] * stride_xn \
                    + delta_x_ptr_window[None, :]
                mask_x_window = (batch + rb < BATCH)[:, None] \
                    & (tki + rtk < WINDOW_SIZE)[None, :]
                matrix_x_window = tl.load(x_window_ptrs, mask=mask_x_window)

                # dot product
                # (BLOCK_BATCH, TK) DOT (TK, BLOCK_K)
                # output shape (BLOCK_BATCH, BLOCK_K)
                acc += tl.dot(matrix_x_window, matrix_w)
                '''
            
            acc = acc.to(y.dtype.element_ty)
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
    # ptr changes for x
    @staticmethod
    def _delta_x_ptr(xc, xh, xw,
                     wc, wh, ww,
                     stride_x, shape_w,
                     TK):
        # get the order of axes in w, innermost dimension outward
        order = sorted([wc, wh, ww], reverse=True)
        c, h, w = [order.index(x) for x in [wc, wh, ww]]

        # PTR_LUT_SIZE
        PTR_LUT_SIZE = _conv._roundup(shape_w[order[0]] * shape_w[order[1]] * shape_w[order[2]], TK)
        r_lut = np.arange(PTR_LUT_SIZE, dtype=np.int32)
        # len is kernel_w * kernel_w * in_c
        current_k = _conv._unpack(r_lut, order, shape_w)

        # compute memory stride
        delta_x = current_k[c] * stride_x[xc]
        delta_x += current_k[h] * stride_x[xh]
        delta_x += current_k[w] * stride_x[xw]

        return delta_x

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
        print(y.shape, 'stride_y', stride_y[yn], stride_y[yc], stride_y[yh], stride_y[yw])
        # accumulator types
        ACC_TYPE = tl.float32 if x.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # if input activation is channel-last, it's good to tiling output
        # into block of [BLOCK_BATCH, BLOCK_K, BLOCK_H, BLOCK_W] because of better
        #if stride_x[xc] == 1 and stride_x > 1 and stride_y > 1:
        if True:
            delta_x_ptr = _conv._delta_x_ptr(xc, xh, xw,
                                             wc, wh, ww,
                                             stride_x, shape_w,
                                             TK=2)
            delta_x_ptr = torch.from_numpy(delta_x_ptr).cuda()
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
                          delta_x_ptr,
                          # Metaparameters
                          ACC_TYPE=ACC_TYPE,
                          BLOCK_BATCH=32, BLOCK_H=16, BLOCK_W=16, BLOCK_K=8,
                          TK=8,
                          GROUP_H=1
                          )
        # do matrix multiplication
        #else:
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
