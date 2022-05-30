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
            stride_biasn,
            # order of dim of w
            # shape_w_o0, shape_w_o1, # shape_w_o2,
            # oc, oh, ow,
            delta_x_ptr,
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
            # Metaparameters
            ACC_TYPE: tl.constexpr,
            # blocks in different dimension
            BLOCK_NHW: tl.constexpr, BLOCK_K: tl.constexpr,
            # reduction tiling parameter for matmul
            BLOCK_CRS: tl.constexpr,
            # Super-blocking for better L2 peformance
            GROUP_H: tl.constexpr
            ):
    '''
    each program instance computes a [BLOCK_BATCH, BLOCK_K, BLOCK_H, BLOCK_W] block of y
    '''
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)

    # offset for output y
    off_y_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    off_y_nhw = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W
    off_y_w = off_y_hw % OUT_W

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_CRS)

    CRS = IN_C * KERNEL_H * KERNEL_W
    # load inc ptr of x, upade x_ptrs
    off_x_crs_unpacked = tl.load(delta_x_ptr + off_x_crs, mask=off_x_crs < CRS)

    x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    mask_x = ((off_x_n < BATCH) & (off_x_h >= 0) & (off_x_h < IN_H + padding_h)
              & (off_x_w >= 0) & (off_x_w < IN_W + padding_w))[:, None] \
        & (off_x_crs < CRS)[None, :]

    # offset for the inital ptr for w
    off_w_crs = tl.arange(0, BLOCK_CRS)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] * stride_wc + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # -----------------------------------------------------------
    # allocate accumulator
    acc = tl.zeros((BLOCK_NHW, BLOCK_K), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_CRS):
        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w)
        # ------ matrix multiplication ------
        acc += tl.dot(matrix_x, matrix_w)
        # ------ update ptrs ------
        w_ptrs += BLOCK_CRS
        # load inc ptr of x, upade x_ptrs
        off_x_crs = crs + tl.arange(0, BLOCK_CRS)
        off_x_crs_unpacked = tl.load(delta_x_ptr + off_x_crs, mask=off_x_crs < CRS)
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        # x_ptrs += BLOCK_CRS

    # add bias if is not None
    if bias is not None:
        off_bias_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        bias_ptrs = bias + off_bias_k * stride_biasn
        mask_bias = off_bias_k < KERNEL_N
        _bias = tl.load(bias_ptrs, mask=mask_bias)
        acc += _bias[None, :]

    acc = acc.to(y.dtype.element_ty)

    # rematerialize -- this saves some registers
    # offset for output y
    off_y_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    off_y_nhw = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    # consider output padding
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # y ptrs in the block of [BLOCK_NHW, BLOCK_K]
    y_ptrs = y + off_y_n[:, None] * stride_yn + off_y_h[:, None] * stride_yh \
        + off_y_w[:, None] * stride_yw + off_y_k[None, :] * stride_yc

    # out-of-bounds check
    mask_y = (off_y_n < BATCH)[:, None] \
        & (off_y_h < OUT_H + output_padding_h)[:, None] \
        & (off_y_w < OUT_W + output_padding_w)[:, None] \
        & (off_y_k < KERNEL_N)[None, :]

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
                     shape_w,
                     dilation_h, dilation_w,
                     stride_xh, stride_xw, stride_xc):
        # get the order of axes in w, innermost dimension outward
        order = sorted([wc, wh, ww], reverse=True)
        c, h, w = [order.index(x) for x in [wc, wh, ww]]
        window_size = shape_w[order[0]] * shape_w[order[1]] * shape_w[order[2]]

        r_window = np.arange(0, window_size)
        window_unpack = _conv._unpack(r_window, order, shape_w)
        window_unpack_h = window_unpack[h]
        window_unpack_w = window_unpack[w]
        window_unpack_c = window_unpack[c]
        r_dilation_h = dilation_h * window_unpack_h[:, None, None]
        r_dilation_w = dilation_w * window_unpack_w[None, :, None]
        r_inc = window_unpack_c[None, None, :]
        delta_x = r_dilation_h * stride_xh + r_dilation_w * stride_xw \
            + r_inc * stride_xc
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
        shape_bias = bias.shape if bias is not None else None

        # indicies for the layeout
        xn, xc, xh, xw = [layout_x.find(x) for x in 'nchw']
        yn, yc, yh, yw = [layout_y.find(x) for x in 'nchw']
        wn, wc, wh, ww = [layout_w.find(x) for x in 'nchw']

        # out_channel, in_channel, kernel_height, kernel_width
        kernel_size = [shape_w[yh], shape_w[yw]]
        input_size = [shape_x[xh], shape_x[xw]]
        assert not shape_bias or shape_bias[0] == shape_w[wn], \
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
        stride_bias = _conv._extract_strides(shape_bias) if shape_bias else None
        stride_biasn = stride_bias[0] if stride_bias else None

        assert((stride_x[0] >= stride_x[1]) and (stride_x[1] >= stride_x[2])
               and (stride_x[2] > stride_x[3])), "stride and layout does not match for x"
        assert((stride_w[0] >= stride_w[1]) and (stride_w[1] >= stride_w[2])
               and (stride_w[2] > stride_w[3])), "stride and layout does not match for w"
        assert((stride_y[0] >= stride_y[1]) and (stride_y[1] >= stride_y[2])
               and (stride_y[2] > stride_y[3])), "stride and layout does not match for y"

        # allocate output
        y = torch.empty(shape_y, device=device, dtype=x.dtype)
        # allocate tmp
        # WINDOW_SIZE = KERNEL_H * KERNEL_W * IN_C
        # tmp_x = torch.empty((BATCH * OUT_H * OUT_W, WINDOW_SIZE), device=device, dtype=x.dtype)
        # tmp_w = torch.empty((WINDOW_SIZE, KERNEL_N), device=device, dtype=w.dtype)
        # accumulator types
        ACC_TYPE = tl.float32 if x.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # if input activation is channel-last, it's good to tiling output
        # into block of [BLOCK_BATCH, BLOCK_K, BLOCK_H, BLOCK_W] because of better
        # if stride_x[xc] == 1 and stride_x > 1 and stride_y > 1:
        if True:
            delta_x = \
                _conv._delta_x_ptr(wc, wh, ww, shape_w,
                                   dilation[0], dilation[1],
                                   stride_x[xh], stride_x[xw], stride_x[xc])
            delta_x = torch.from_numpy(delta_x).cuda()
            # launch kernel, 2-dim, batch*h*w, kernel
            grid = lambda META: (triton.cdiv(BATCH * OUT_H * OUT_W, META['BLOCK_NHW']),
                                 triton.cdiv(KERNEL_N, META['BLOCK_K']))
            _kernel[grid](x, w, bias, y,
                          # stride nchw for x,w,y tensor
                          stride_x[xn], stride_x[xc], stride_x[xh], stride_x[xw],
                          stride_w[wn], stride_w[wc], stride_w[wh], stride_w[ww],
                          stride_y[yn], stride_y[yc], stride_y[yh], stride_y[yw],
                          stride_biasn,
                          # pointer inc for x
                          delta_x,
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
                          # Metaparameters
                          ACC_TYPE=ACC_TYPE,
                          BLOCK_NHW=128, BLOCK_K=32,
                          BLOCK_CRS=16,
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
