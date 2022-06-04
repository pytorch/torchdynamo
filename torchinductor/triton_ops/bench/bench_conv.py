import torch
import triton

import torchinductor.triton_ops
import model

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
useCudaGraph = False

# conv benchmarks
conv_confs = [
    triton.testing.Benchmark(
        x_names=["BATCH"],
        x_vals=[32],
        line_arg="provider",
        line_vals=["cublas", "triton"],
        line_names=["cuBLAS", "Triton"],
        ylabel="TFLOPS",
        plot_name="resnet50-conv-performance",
        args={"IN_H": IN_H, "IN_W": IN_W, "IN_C": IN_C, "KERNEL_N": KERNEL_N,
              "KERNEL_H": KERNEL_H, "KERNEL_W": KERNEL_W, "stride": stride, "padding": padding},
    ) for IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding in model.resnet50_layers
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
        dtype=torch.float16, warmup=25, rep=75):

    x = torch.randn((BATCH, IN_H, IN_W, IN_C), dtype=dtype, device='cuda')
    w = torch.randn((KERNEL_N, KERNEL_H, KERNEL_W, IN_C // groups),
                    dtype=dtype, device='cuda')
    bias = torch.randn((KERNEL_N), dtype=dtype, device='cuda')
    OUT_H = (IN_H + 2 * padding[0] - dilation[0] * (KERNEL_H - 1) - 1 + stride[0]) // stride[0]
    OUT_W = (IN_W + 2 * padding[1] - dilation[1] * (KERNEL_W - 1) - 1 + stride[1]) // stride[1]

    tflops = lambda ms: 2. * BATCH * OUT_H * OUT_W * IN_C * KERNEL_H * KERNEL_W * KERNEL_N / ms * 1e-9
    if provider == "cublas":
        conv2d_layer = torch.nn.Conv2d(
            IN_C, KERNEL_N, (KERNEL_H, KERNEL_W),
            stride=stride, padding=padding, dilation=dilation,
            groups=groups,
        )
        conv2d_layer.weight.data = w.permute((0, 3, 1, 2))
        conv2d_layer.bias.data = bias
        x = x.permute((0, 3, 1, 2))
        fn = lambda: conv2d_layer(x)
    if provider == "triton":
        # if KERNEL_H == 1 and KERNEL_W == 1:
        #     fn = lambda: torchinductor.triton_ops.conv1x1(
        #         x, w, bias, stride, padding, dilation, False, (0, 0), groups
        #     )
        # else:
        fn = lambda: torchinductor.triton_ops.conv(
            x, w, bias, stride, padding, dilation, False, (0, 0), groups
        )

    # useCudaGraph won't change the TFLOPs,
    # because do_bench() clear L2 cache to hide the latency of CPU launch time
    if useCudaGraph:
        # warmp up for cudagraph
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                tmp = fn()
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            tmp = fn()

        fn = lambda: g.replay()
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


bench_op.run(print_data=True)
