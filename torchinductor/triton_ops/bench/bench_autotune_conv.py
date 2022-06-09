import torch
import triton

import torchdynamo
import model
import torchdynamo.config as config

# enable autotune conv kernels
config.autotune = True
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


# conv benchmarks
conv_confs = [
    triton.testing.Benchmark(
        x_names=["layout"],
        x_vals=["nchw", "nhwc"],
        line_arg="provider",
        line_vals=["cublas", "autotune"],
        line_names=["cuBLAS", "autotune"],
        ylabel="TFLOPS",
        plot_name=f"resnet50-conv{i}-perf",
        args={"BATCH": BATCH, "IN_H": IN_H, "IN_W": IN_W, "IN_C": IN_C, "KERNEL_N": KERNEL_N,
              "KERNEL_H": KERNEL_H, "KERNEL_W": KERNEL_W, "stride": stride, "padding": padding},
    ) for i, (IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding) in enumerate(model.resnet50_layers)
    for BATCH in [32]
]


@triton.testing.perf_report(conv_confs)
def bench_op(
        # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W,
        KERNEL_N, KERNEL_H, KERNEL_W,
        # provider
        provider,
        # parameters of conv
        stride=(1, 1), padding=(0, 0),
        dilation=(1, 1), groups=1,
        dtype=torch.float32, layout="nhwc",
        warmup=25, rep=75):

    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device='cuda')
    w = torch.randn((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
                    dtype=dtype, device='cuda')
    bias = torch.randn((KERNEL_N), dtype=dtype, device='cuda')
    if layout == "nhwc":
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    OUT_H = (IN_H + 2 * padding[0] - dilation[0] * (KERNEL_H - 1) - 1 + stride[0]) // stride[0]
    OUT_W = (IN_W + 2 * padding[1] - dilation[1] * (KERNEL_W - 1) - 1 + stride[1]) // stride[1]

    tflops = lambda ms: 2. * BATCH * OUT_H * OUT_W * IN_C * KERNEL_H * KERNEL_W * KERNEL_N / ms * 1e-9
    if provider == "cublas":
        fn = lambda: torch.conv2d(x, w, bias, stride, padding, dilation, groups)
    if provider == "autotune":
        @torchdynamo.optimize("inductor")
        def wrap_conv(x, w, bias, stride, padding, dilation, groups):
            return torch.conv2d(x, w, bias, stride, padding, dilation, groups)

        fn = lambda: wrap_conv(x, w, bias, stride, padding, dilation, groups)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


bench_op.run(print_data=True)
