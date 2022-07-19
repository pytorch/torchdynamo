from benchmarks.microbenchmarks import model as model
import torch
import torchdynamo
import torchinductor.config
from torchdynamo.testing import same

# torchinductor.config.debug = True
torchinductor.config.triton.dense_indexing = True
torchinductor.config.triton.convolution = "triton"
torch.manual_seed(0)

class Func(object):
    # conv+bias
    @torchdynamo.optimize("inductor")
    def conv_add_torchinductor(x, w, bias, stride, padding, dilation, groups):
        y =  torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        return y

    # conv+bias
    def conv_add(x, w, bias, stride, padding, dilation, groups):
        y =  torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        return y

    # relu(conv)
    @torchdynamo.optimize("inductor")
    def conv_relu_torchinductor(x, w, bias, stride, padding, dilation, groups):
        y =  torch.conv2d(x, w, None, stride, padding, dilation, groups)
        return torch.relu(y)

    # relu(conv)
    def conv_relu(x, w, bias, stride, padding, dilation, groups):
        y =  torch.conv2d(x, w, None, stride, padding, dilation, groups)
        return torch.relu(y)

    # relu(conv+bias)
    @torchdynamo.optimize("inductor")
    def conv_add_relu_torchinductor(x, w, bias, stride, padding, dilation, groups):
        y =  torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        return torch.relu(y)

    # relu(conv+bias)
    def conv_add_relu(x, w, bias, stride, padding, dilation, groups):
        y =  torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        return torch.relu(y)

    # bn(conv)
    @torchdynamo.optimize("inductor")
    def conv_bn_torchinductor(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        y = torch.batch_norm(
            y, weight=bias, bias=torch.ones_like(bias),
            running_mean=torch.zeros_like(bias), running_var=torch.zeros_like(bias),
            training=False, momentum=1, eps=1e-5, cudnn_enabled=True,
        )
        return y

    # bn(conv)
    def conv_bn(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        y = torch.batch_norm(
            y, weight=bias, bias=torch.ones_like(bias),
            running_mean=torch.zeros_like(bias), running_var=torch.zeros_like(bias),
            training=False, momentum=1, eps=1e-5, cudnn_enabled=True,
        )
        return y

    # relu(bn(conv))
    @torchdynamo.optimize("inductor")
    def conv_bn_relu_torchinductor(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        y = torch.batch_norm(
            y, weight=bias, bias=None,
            running_mean=torch.zeros_like(bias), running_var=torch.zeros_like(bias),
            training=False, momentum=1, eps=1e-5, cudnn_enabled=True,
        )
        return torch.relu(y)

    # relu(bn(conv))
    def conv_bn_relu(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        y = torch.batch_norm(
            y, weight=bias, bias=None,
            running_mean=torch.zeros_like(bias), running_var=torch.zeros_like(bias),
            training=False, momentum=1, eps=1e-5, cudnn_enabled=True,
        )
        return torch.relu(y)

def test(layer_params, fusion_type="add"):
    BATCH = 32
    IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = layer_params
    dilation, groups = (1,1), 1
    dtype=torch.float32

    torch.manual_seed(0)
    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device='cuda') #.to(memory_format=torch.channels_last)
    w = torch.randn((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
                    dtype=dtype, device='cuda') #.to(memory_format=torch.channels_last)
    # bias = torch.randn((1, KERNEL_N, 1, 1), dtype=dtype, device='cuda') #.to(memory_format=torch.channels_last)
    bias = torch.randn((KERNEL_N), dtype=dtype, device='cuda')

    conv_fusion_torchinductor = getattr(Func, f"conv_{fusion_type}_torchinductor")
    conv_fusion = getattr(Func, f"conv_{fusion_type}")
    y = conv_fusion_torchinductor(x, w, bias, stride, padding, dilation, groups)
    y_correct = conv_fusion(x, w, bias, stride, padding, dilation, groups)
    # print("y", y[0])
    # print("y_correct", y_correct[0])
    assert(same(y, y_correct, cos_similarity=True))

fusion_types = ["add", "relu", "add_relu", "bn", "bn_relu"]
for fusion_type in fusion_types:
    print(f"testing correctness of conv+{fusion_type}")
    for id, layer in enumerate(model.resnet50_layers[:1]):
        test(layer, fusion_type)
    print("passed")
