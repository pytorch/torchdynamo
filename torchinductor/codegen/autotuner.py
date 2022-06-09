import builtins

import torch
import triton

import torchinductor
import torchinductor.triton_ops

aten = torch.ops.aten
triton_ops = torchinductor.triton_ops

# Instatiatie Autotuner class object -- init the cache, and sort of global

# Autotuning an op
#   1) Register that Node in the codegen of ExternKernelOP
#   2) Autotuner
#   1) register that IRNode "ExternKerneNode" in Autotuner - autotune()
#
# def autotune():
#     """
#     Decorator for auto-tuning a function.

#     .. highlight:: python
#     .. code-block:: python

#         @triton.autotune()
#         wrap_conv(x, w, bias, stride, ...)

#     :note: For the wrap function, the candidate kernels will run multiple time.
#     """
#     def inner(fn):
#         autotuner = Autotuner(fn)
#         return autotuner

#     return inner


class Autotuner:
    def __init__(self):

        self.cache = dict()

    def _bench(self, *args, kernel, **kwargs):
        def kernel_call():
            kernel(*args, **kwargs)

        return triton.testing.do_bench(kernel_call, warmup=10, rep=50)

    # def __call__(self, *args, **kwargs):
    #     # collect input shape/stride/device, layer params
    #     self.fn.set_args(*args, **kwargs)
    #     # get candidate kernels for fn
    #     self.kernels = self.fn.candidate_kernels()
    #     # filter kernels that args/kwargs does not meet requirements
    #     self.fn.filter_kernels(self.kernels)
    #     # if only one choice, return that kernel
    #     if len(self.kernels) == 1:
    #         kernel = self.kernels[0]
    #         return kernel(*args, **kwargs)
    #     # use input shape, stride, layer parameters as key to cache the best kernel to use
    #     key = self.fn.gen_key()
    #     if key not in self.cache:
    #         bench_start = time.time()
    #         timings = {
    #             kernel: self._bench(*args, kernel=kernel, **kwargs) for kernel in self.kernels
    #         }
    #         bench_end = time.time()
    #         self.bench_time = bench_end - bench_start
    #         self.cache[key] = builtins.min(timings, key=timings.get)
    #         self.kernels_timings = timings
    #         kernel = self.cache[key]
    #     self.best_kernel = kernel
    #     return self.best_kernel(*args, **kwargs)


autotune = Autotuner()


def tuned_conv(*args, **kwargs):

    # gather info
    x = args[0]
    w = args[1]
    # bias = args[2]
    stride = args[3]
    padding = args[4]
    dilation = args[5]
    transposed = args[6]
    output_padding = args[7]
    groups = args[8]

    BATCH, IN_C, IN_H, IN_W = x.shape
    KERNEL_N, _, KERNEL_H, KERNEL_W = w.shape
    stride = stride
    padding = padding
    dilation = dilation
    transposed = transposed
    output_padding = output_padding
    groups = groups
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
    kernels = ["aten.convolution"]
    if use_cuda:
        kernels += ["triton_ops.conv", "triton_ops.conv1x1"]

    # filter kernels that args/kwargs does not meet requirements
    remove_kernels = []
    if groups > 1 or transposed:
        remove_kernels += ["triton_ops.conv", "triton_ops.conv1x1"]
    # triton_ops.conv1x1 could only deal with nhwc 1x1 kernel
    if stride_x[1] > 1 or KERNEL_H > 1 or KERNEL_W > 1:
        remove_kernels += ["triton_ops.conv1x1"]
    kernels = [k for k in kernels if k not in remove_kernels]

    # str to callable functions
    kernel_dict = {
        "aten.convolution": aten.convolution,
        "triton_ops.conv": triton_ops.conv,
        "triton_ops.conv1x1": triton_ops.conv1x1,
    }

    # if only one choice, return that kernel
    if len(kernels) == 1:
        kernel = kernel_dict[kernels[0]]
        return kernel(*args, **kwargs)
    if key not in autotune.cache:
        # bench_start = time.time()
        timings = {
            kernel: autotune._bench(*args, kernel=kernel_dict[kernel], **kwargs)
            for kernel in kernels
        }
        # bench_end = time.time()
        # bench_time = bench_end - bench_start
        autotune.cache[key] = builtins.min(timings, key=timings.get)
        if torchinductor.config.debug:
            print("for key = ", key)
            print("best_kernel", autotune.cache[key])
        # kernels_timings = timings
    best_kernel_name = autotune.cache[key]
    best_kernel = kernel_dict[best_kernel_name]
    return best_kernel(*args, **kwargs)
