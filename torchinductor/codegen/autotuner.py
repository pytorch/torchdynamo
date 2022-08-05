import builtins

import torch
import triton

import torchinductor
import torchinductor.triton_ops
from torchdynamo.testing import rand_strided

from ..virtualized import V

aten = torch.ops.aten
triton_ops = torchinductor.triton_ops


def str2func(str):
    module, name = str.split(".")
    if module == "aten":
        return getattr(aten, name)
    elif module == "triton_ops":
        return getattr(triton_ops, name)
    else:
        raise Exception(f"{str} could not be called")


class Autotuner:
    def __init__(self):

        self.cache = dict()

    def _bench(self, kernel, *args, **kwargs):
        def kernel_call():
            kernel(*args, **kwargs)

        return triton.testing.do_bench(kernel_call, warmup=10, rep=50)


autotune = Autotuner()


def tuned_conv(
    x_shape,
    w_shape,
    x_stride,
    w_stride,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    device,
    dtype,
    adjust_triton=0.95,
):
    """
    Return the best kernel name given inputs and layer parameters;
    Considering potential pointwise fusion of conv, we could adjust triton timing
    by multiplying adjust_triton (default=0.95)
    """

    sizevars = V.graph.sizevars
    x_shape = [sizevars.size_hint(s) for s in x_shape]
    w_shape = [sizevars.size_hint(s) for s in w_shape]
    x_stride = [sizevars.size_hint(s) for s in x_stride]
    w_stride = [sizevars.size_hint(s) for s in w_stride]
    x = rand_strided(x_shape, x_stride, device=device, dtype=dtype)
    w = rand_strided(w_shape, w_stride, device=device, dtype=dtype)
    # the identifiable args for the layers
    id_args = [
        *x_shape,
        *w_shape,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        # *x_stride,
        # *w_stride,
    ]
    use_cuda = x.is_cuda

    # gen_key
    key = tuple([arg for arg in id_args])
    key = ("conv",) + key

    # candidate kernels
    kernels = ["aten.convolution"]
    if use_cuda:
        kernels += ["triton_ops.conv"]

    # filter kernels that args/kwargs does not meet requirements
    remove_kernels = []
    if groups > 1 or transposed:
        remove_kernels += ["triton_ops.conv"]
    kernels = [k for k in kernels if k not in remove_kernels]

    # if only one choice, return that kernel
    if len(kernels) == 1:
        kernel = kernels[0]
        # return kernel(
        #     x, w, stride, padding, dilation, transposed, output_padding, groups
        # )
        return kernel
    timings = {}
    if key not in autotune.cache:
        # bench_start = time.time()
        for kernel in kernels:
            runnable_kernel = str2func(kernel)
            if "triton_ops" in kernel:
                # because we use nhwc layout by default for triton conv
                x = x.to(memory_format=torch.channels_last)
            timing, _, _ = autotune._bench(
                runnable_kernel,
                x,
                w,
                None,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            )
            if "triton_ops" in kernel:
                timing = timing * adjust_triton
            timings[kernel] = timing
        # bench_end = time.time()
        # bench_time = bench_end - bench_start
        autotune.cache[key] = builtins.min(timings, key=timings.get)
        # if torchinductor.config.debug:
        print("for key = ", key)
        print("timing", timings)
        print("best_kernel", autotune.cache[key])
    best_kernel = autotune.cache[key]
    return best_kernel
    # return best_kernel(
    #     x, w, stride, padding, dilation, transposed, output_padding, groups
    # )
