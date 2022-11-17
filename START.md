# TorchDynamo

> TorchDynamo makes it easy to experiment with different compiler backends to make PyTorch code faster with a single line decorator `torch._dynamo.optimize()`

TorchDynamo supports arbitrary PyTorch code, control flow, mutation and comes with experimental support for dynamic shapes

Let's start with a simple example and make things more complicated step by step. Please note that you're likely to see more significant speedups the newer your GPU is.


```python
from torch._dynamo import optimize
import torch


def fn(x, y):
    a = torch.cos(x).cuda()
    b = torch.sin(y).cuda()
    return a + b

new_fn = optimize("inductor")(fn)
input_tensor = torch.randn(10000).to(device="cuda:0")
a = new_fn()
```

This example won't actually run faster but it's a good educational example that features `torch.cos()` and `torch.sin()` which are examples of pointwise ops as in they operate element by element on a vector. A more famous pointwise op you might actually want to use would be something like `torch.relu()`. Pointwise ops in eager mode are suboptimal because each one would need to need to read a tensor from memory, make some changes and then write back those changes. The single most important optimization that inductor does is fusion. So back to our example we can turn 2 reads and 2 writes into 1 read and 1 write which is crucial especially for newer GPUs where the bottleneck is memory bandwidth (how quickly you can send data to a GPU) instead of compute (how quickly your GPU can crunch floating point operations)

dynamo supports many different backends but inductor specifically works by generating [Triton](https://github.com/openai/triton) kernels and we can inspect them by running `TORCHINDUCTOR_TRACE=1 python trig.py ` with the actual generated kernel being

```python
@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sin(tmp0)
    tmp2 = tl.sin(tmp1)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
```

And you can verify that fusing the two `sins` did actually occur because the two `sin` operations occur within a single Triton kernel and the temporary variables are held in registers with very fast access.

You can read up a lot more on Triton's performance [here](https://openai.com/blog/triton/) but the key is it's in python so you can easily understand it even if you haven't written all that many CUDA kernels.

As a next step let's try a real model like resnet50 from the PyTorch hub.

```python
import torch
import torch._dynamo as dynamo
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
opt_model = dynamo.optimize("inductor")(model)
model(torch.randn(1,3,64,64))
```

And that's not the only available backend, you can run in a REPL `dynamo.list_backends()` to see all the available ones. Try out the `aot_cudagraphs` or `nvfuser` next as inspiration.

Let's do something a bit more interesting now, our community frequently uses pretrained models from [transformers](https://github.com/huggingface/transformers) or [TIMM](https://github.com/rwightman/pytorch-image-models) and one of our design goals is for dynamo and inductor to work out of the box with any model that people would like to author.

So we're going to directly download a pretrained model from the HuggingFace hub and optimize it

```python
import torch
from transformers import BertTokenizer, BertModel
import torch._dynamo as dynamo
# Copy pasted from here https://huggingface.co/bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")
model = dynamo.optimize("inductor")(model) # This is the only line of code that we changed
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:0")
output = model(**encoded_input)
```

If you remove the `to(device="cuda:0")` from the model and encoded_input then triton will generate C++ kernels that will be optimized for running on your CPU. You can inspect both Triton or C++ kernels for BERT, they're obviously more complex than the trigonometry example we had above but you can similarly skim it and understand if you understand PyTorch.

Similarly let's try out a TIMM example

```python
import timm
import torch._dynamo as dynamo
import torch
model = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=2)
opt_model = dynamo.optimize("inductor")(model)
opt_model(torch.randn(64,3,7,7))
```

