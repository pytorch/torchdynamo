import torch
import torchinductor.triton_ops

from torch.profiler import profile, record_function, ProfilerActivity

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, stride, padding, dilation, groups, dtype = \
    32, 56, 56, 64, 3, 3, 64, (1, 1), (0, 0), (1, 1), 1, torch.float32

def profile_op(
        # provider
        provider,
        # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W,
        KERNEL_N, KERNEL_H, KERNEL_W,
        # parameters of conv
        stride=(1, 1), padding=(0, 0),
        dilation=(1, 1), groups=1,
        dtype=torch.float16, warmup=25, rep=50):


    x = torch.randn((BATCH, IN_H, IN_W, IN_C), dtype=dtype, device='cuda')
    w = torch.randn((KERNEL_N, KERNEL_H, KERNEL_W, IN_C // groups),
                    dtype=dtype, device='cuda')
    bias = torch.randn((KERNEL_N), dtype=dtype, device='cuda')
    OUT_H = (IN_H + 2 * padding[0] - dilation[0] * (KERNEL_H - 1) - 1 + stride[0]) // stride[0]
    OUT_W = (IN_W + 2 * padding[1] - dilation[1] * (KERNEL_W - 1) - 1 + stride[1]) // stride[1]

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
    elif provider == "triton":
        fn = lambda: torchinductor.triton_ops.conv(
            x, w, bias, stride, padding, dilation, False, (0, 0), groups
        )
    else:
        ValueError(f"{provider} not supported")
    # warm up
    for _ in range(warmup):
        fn()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, use_cuda=True) as prof:
        with record_function("model_inference"):
            for _ in range(rep):
                fn()

    print("Profiling ", provider)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


for provider in ["cublas", "triton"]:
    profile_op(
        # provider
        provider,
        # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W,
        KERNEL_N, KERNEL_H, KERNEL_W,
        # parameters of conv
        stride, padding,
        dilation, groups,
        dtype=dtype, warmup=25, rep=50
    )