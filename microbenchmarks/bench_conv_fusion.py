from microbenchmarks import model as model
import torch
import torchdynamo
import torchinductor.config
import triton
from torchdynamo.testing import same
from prettytable import PrettyTable

# torchinductor.config.debug = True
torchinductor.config.triton.convolution = "triton"
torch.manual_seed(0)
useCudaGraph = True

@torchdynamo.optimize("inductor")
def conv_fusion_torchinductor(x, w, bias):
    y =  torch.conv2d(x, w)
    # y += bias
    return torch.relu(y)

def conv_fusion(x, w, bias):
    y =  torch.conv2d(x, w)
    # y += bias
    return torch.relu(y)

@torchdynamo.optimize("inductor")
def conv_torchinductor(x, w, bias):
    return torch.conv2d(x, w)

def conv(x, w, bias):
    return torch.conv2d(x, w)


def cuda_graph(fn, x, w, bias):
    new_x = x.clone()
    new_w = w.clone()
    new_bias = bias.clone()

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

    def fn():
        x.copy_(new_x)
        w.copy_(new_w)
        bias.copy_(new_bias)
        return g.replay()

    return fn



def bench(layer_params, layer_id, p):
    BATCH = 32
    IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = layer_params
    dilation, groups = (1,1), 1
    dtype=torch.float32
    provider = "torchinductor"

    OUT_H = (IN_H + 2 * padding[0] - dilation[0] * (KERNEL_H - 1) - 1 + stride[0]) // stride[0]
    OUT_W = (IN_W + 2 * padding[1] - dilation[1] * (KERNEL_W - 1) - 1 + stride[1]) // stride[1]
    tflops = lambda ms: 2. * BATCH * OUT_H * OUT_W * IN_C * KERNEL_H * KERNEL_W * KERNEL_N / ms * 1e-9

    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device='cuda') #.to(memory_format=torch.channels_last)
    w = torch.randn((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
                    dtype=dtype, device='cuda') #.to(memory_format=torch.channels_last)
    bias = torch.randn((1, KERNEL_N, 1, 1), dtype=dtype, device='cuda') #.to(memory_format=torch.channels_last)

    y = conv_torchinductor(x, w, bias)
    y_correct = conv(x, w, bias)
    # print("y", y[0,:,0,0])
    # print("y_correct", y[0,:,0,0])
    assert(same(y, y_correct, cos_similarity=True))

    fn_conv = lambda: conv(x, w, bias)
    fn_conv_fusion = lambda: conv_fusion(x, w, bias)
    if useCudaGraph:
        fn_conv = cuda_graph(fn_conv, x, w, bias)
        fn_conv_fusion = cuda_graph(fn_conv_fusion, x, w, bias)
        

    fn_conv_torchinductor = lambda: conv_torchinductor(x, w, bias)
    fn_conv_fusion_torchinductor = lambda: conv_fusion_torchinductor(x, w, bias)

    torch_conv_ms, _, _ = triton.testing.do_bench(fn_conv)
    torch_conv_fusion_ms, _, _ = triton.testing.do_bench(fn_conv_fusion)
    triton_conv_ms, _, _ = triton.testing.do_bench(fn_conv_torchinductor)
    triton_conv_fusion_ms, _, _ = triton.testing.do_bench(fn_conv_fusion_torchinductor)

    p.add_row([layer_id, tflops(torch_conv_ms), tflops(triton_conv_ms), tflops(torch_conv_fusion_ms), tflops(triton_conv_fusion_ms)])



p = PrettyTable()
p.field_names = ["layer", "torch conv", "triton conv", "torch conv+relu", "triton conv+relu"]
p.float_format = ".3"
for id, layer in enumerate(model.resnet50_layers):
    bench(layer, id, p)

print(p)