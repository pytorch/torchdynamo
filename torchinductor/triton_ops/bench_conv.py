import math
import torch
import triton

import torchinductor.triton_ops


def rounded_linspace(low, high, steps, div):
    ret = torch.linspace(low, high, steps)
    ret = (ret.int() + div - 1) // div * div
    ret = torch.unique(ret)
    return list(map(int, ret))


def powspace(start, stop, pow, steps):
    start = math.log(start, pow)
    stop = math.log(stop, pow)
    ret = torch.pow(pow, torch.linspace(start, stop, steps))
    return list(map(int, ret))


# conv benchmarks
conv_confs = [
    triton.testing.Benchmark(
        x_names=["IN_H", "IN_W"],
        x_vals=[8, 16, 32, 64, 128, 256], #powspace(8, 256, 2, 1),
        line_arg="provider",
        line_vals=["cublas", "triton"],
        line_names=["cuBLAS", "Triton"],
        ylabel="TFLOPS",
        plot_name=f"conv-performance-BATCH={BATCH}-IN_C={IN_C}-KN={KERNEL_N}-KH=KW={KERNEL_H}",
        args={"BATCH": BATCH, "IN_C": IN_C, "KERNEL_N": KERNEL_N,
              "KERNEL_H": KERNEL_H, "KERNEL_W": KERNEL_W},
    ) for BATCH in [32]
    for IN_C in [128]  #powspace(16, 256, 2, 1)
    for KERNEL_N in [32]  #powspace(16, 256, 2, 1)
    for KERNEL_H in [2]
    for KERNEL_W in [2]
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
        dtype=torch.float32, warmup=25, rep=75):

    x = torch.randn((BATCH, IN_H, IN_W, IN_C), dtype=dtype, device='cuda')
    w = torch.randn((KERNEL_N, KERNEL_H, KERNEL_W, IN_C // groups),
                    dtype=dtype, device='cuda')
    bias = torch.randn((KERNEL_N), dtype=dtype, device='cuda')
    OUT_H = (IN_H + 2 * padding[0] - dilation[0] * (KERNEL_H - 1) - 1 + stride[0]) // stride[0]
    OUT_W = (IN_W + 2 * padding[1] - dilation[1] * (KERNEL_W - 1) - 1 + stride[1]) // stride[1]

    tflops = lambda ms: 2. * BATCH * OUT_H * OUT_W * IN_C * KERNEL_H * KERNEL_W * KERNEL_N / ms * 1e-9
    if provider == "cublas":
        conv2d_layer = torch.nn.Conv2d(IN_C, KERNEL_N, (KERNEL_H, KERNEL_W),
                                       stride=stride, padding=padding, dilation=dilation,
                                       groups=groups)
        conv2d_layer.weight.data = w.permute((0, 3, 1, 2))
        conv2d_layer.bias.data = bias
        x = x.permute((0, 3, 1, 2))
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: conv2d_layer(x), warmup=warmup, rep=rep)
        return tflops(ms), tflops(max_ms), tflops(min_ms)
    if provider == "triton":
        ms, min_ms, max_ms = \
            triton.testing.do_bench(
                lambda: torchinductor.triton_ops.conv(x, w, bias, stride, padding, dilation,
                                                      False, (0, 0), groups), warmup=warmup, rep=rep)
        return tflops(ms), tflops(max_ms), tflops(min_ms)


bench_op.run(show_plots=True, print_data=True)
