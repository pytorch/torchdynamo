import builtins

import torch
import triton

import torchinductor
import torchinductor.triton_ops

aten = torch.ops.aten
triton_ops = torchinductor.triton_ops


class Autotuner:
    def __init__(self):

        self.cache = dict()

    def _bench(self, kernel, *args, **kwargs):
        def kernel_call():
            kernel(*args, **kwargs)

        return triton.testing.do_bench(kernel_call, warmup=10, rep=50)


autotune = Autotuner()


def tuned_conv(
    x, w, bias, stride, padding, dilation, transposed, output_padding, groups
):

    BATCH, IN_C, IN_H, IN_W = x.shape
    KERNEL_N, _, KERNEL_H, KERNEL_W = w.shape
    stride_x = x.stride()
    stride_w = w.stride()
    # the identifiable args for the layers
    id_args = [
        BATCH,
        IN_C,
        IN_H,
        IN_W,
        KERNEL_N,
        KERNEL_H,
        KERNEL_W,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        stride_x,
        stride_w,
    ]
    use_cuda = x.is_cuda

    # gen_key
    key = tuple([arg for arg in id_args])
    key = ("conv",) + key

    # candidate kernels
    kernels = [aten.convolution]
    if use_cuda:
        kernels += [triton_ops.conv, triton_ops.conv1x1]

    # filter kernels that args/kwargs does not meet requirements
    remove_kernels = []
    if groups > 1 or transposed:
        remove_kernels += [triton_ops.conv, triton_ops.conv1x1]
    # triton_ops.conv1x1 could only deal with nhwc 1x1 kernel
    if KERNEL_H > 1 or KERNEL_W > 1:
        remove_kernels += [triton_ops.conv1x1]
    kernels = [k for k in kernels if k not in remove_kernels]

    # if only one choice, return that kernel
    if len(kernels) == 1:
        kernel = kernels[0]
        return kernel(
            x, w, bias, stride, padding, dilation, transposed, output_padding, groups
        )
    if key not in autotune.cache:
        # bench_start = time.time()
        timings = {
            kernel: autotune._bench(
                kernel,
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
            for kernel in kernels
        }
        # bench_end = time.time()
        # bench_time = bench_end - bench_start
        autotune.cache[key] = builtins.min(timings, key=timings.get)
        if torchinductor.config.debug:
            print("for key = ", key)
            print("timing", timings)
            print("best_kernel", autotune.cache[key])
    best_kernel = autotune.cache[key]
    return best_kernel(
        x, w, bias, stride, padding, dilation, transposed, output_padding, groups
    )
