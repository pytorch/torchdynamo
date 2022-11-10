# NOTICE: TorchDynamo has moved

We have moved TorchDynamo to 
[pytorch/pytorch](https://github.com/pytorch/pytorch/tree/master/torch/_dynamo)
- `import torchdynamo` is now `import torch._dynamo`
- `import torchinductor` is now `import torch._inductor`

For instructions to port PRs over, or more details on the move see 
[issue 1588](https://github.com/pytorch/torchdynamo/issues/1588).

This repository still contains:
- An alias to the new location
- Issues: we will continue using this project for issue tracking
- Documentation that needs to be ported over/updated


# TorchDynamo

> TorchDynamo makes it easy to experiment with different compiler backends to make PyTorch code faster with a single line decorator `torch._dynamo.optimize()`

TorchDynamo supports arbitrary PyTorch code, control flow, mutation and dynamic shapes.

You can follow our nightly benchmarks [here](https://github.com/pytorch/torchdynamo/issues/681)


TorchDynamo is a Python-level JIT compiler designed to make unmodified
PyTorch programs faster. TorchDynamo hooks into the frame evaluation API
in CPython ([PEP 523]) to dynamically modify Python bytecode right before
it is executed. It rewrites Python bytecode in order to extract sequences
of PyTorch operations into an [FX Graph] which is then just-in-time
compiled with a customizable backend. It creates this FX Graph through
bytecode analysis and is designed to mix Python execution with compiled
backends to get the best of both worlds: usability and performance.

![](./documentation/images/TorchDynamo.png)

For more on TorchDynamo you can read our [posts on PyTorch dev-discuss]
or [watch a deep-dive video].

This repository also hosts *TorchInductor*, which is TorchDynamo backend
able to translate an [FX Graph] into [Triton] for GPUs or [C++/OpenMP]
for CPUs.  We have a [training performance dashboard] comparing the
performance of different training backends.  You can read more in the
[TorchInductor post on PyTorch dev-discuss].

[Triton]: https://github.com/openai/triton
[C++/OpenMP]: https://www.openmp.org/
[posts on PyTorch dev-discuss]: https://dev-discuss.pytorch.org/search?q=TorchDynamo%20order%3Alatest
[watch a deep-dive video]: https://www.youtube.com/watch?v=egZB5Uxki0I
[FX Graph]: https://pytorch.org/docs/stable/fx.html
[PEP 523]: https://peps.python.org/pep-0523/
[training performance dashboard]: https://github.com/pytorch/torchdynamo/issues/681#issuecomment-1233828468
[TorchInductor post on PyTorch dev-discuss]: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747

*TorchDynamo is experimental* and under active development.
You are welcome to try it out and contribute, but should expect to find
bugs and rough edges.

## Requirements and Setup

Python 3.8 is recommended.
Python 3.7 through 3.10 are supported and tested.
Make sure to have a development version of python installed locally as well.

TorchDynamo is included in the nightly binaries of PyTorch, for reference, https://pytorch.org/get-started/locally/

### Install GPU/CUDA version requirements

To use GPU back ends (and in particular Triton), please make sure that the cuda
that you have installed locally matches the PyTorch version you are running.

The following command installs GPU PyTorch+TorchDynamo along with GPU TorchDynamo dependencies (for CUDA 11.7):

`pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117`


### CPU requirements

There are no additional requirements for CPU TorchDynamo. CPU TorchDynamo is included in the nightly versions of PyTorch, which, for reference, can be installed with

`pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

### Install from local source

Build PyTorch from source: https://github.com/pytorch/pytorch#from-source, which has TorchDynamo included.

To install GPU TorchDynamo dependencies, run `make triton` in the PyTorch repo root directory.

### Verify Installation

If you built PyTorch from source, then you can run the following commands (from the PyTorch repo root directory) that run minimal examples to check that TorchDynamo is installed correctly:

```shell
cd tools/dynamo
python verify_dynamo.py
```

If you do not have the PyTorch source locally, you can alternatively copy the script (`tools/dynamo/verify_dynamo.py`) from the PyTorch repo and run it locally.

## Docker installation

We also provide all the required dependencies in the PyTorch nightly binaries which you can download with

`docker pull ghcr.io/pytorch/pytorch-nightly`

And for ad hoc experiments just make sure that your container has access to all your GPUs

`docker run --rm --gpus all -it pytorch-nightly /bin/bash`


## Getting started

Here is a basic example of how to use TorchDynamo. You can decorate a function
or a method using `torch._dynamo.optimize()` and pass in the name of a compiler e.g: inductor and your code will run faster.

```py
@dynamo.optimize("inductor")
def fn(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b
```

It's also easy to define your own compiler backends in pure python [custom backend](./documentation/custom-backend.md)


### Existing Backends

TorchDynamo has a growing list of backends, which can be found in [backends.py](https://github.com/pytorch/pytorch/blob/master/torch/_dynamo/optimizations/backends.py)
or `torchdynamo.list_backends()` each of which with its optional dependencies.

Some of the most commonly used backends are

**Debugging backends**:
* `dynamo.optimize("eager")` - Uses PyTorch to run the extracted GraphModule. This is quite useful in debugging TorchDynamo issues.
* `dynamo.optimize("aot_eager")` - Uses AotAutograd with no compiler, i.e, just using PyTorch eager for the AotAutograd's extracted forward and backward graphs. This is useful for debugging, and unlikely to give speedups.

**Training & inference backends**:
* `dynamo.optimize("inductor")` - Uses TorchInductor backend with AotAutograd and cudagraphs by leveraging codegened Triton kernels  [Read more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
* `dynamo.optimize("nvfuser")` -  nvFuser with TorchScript. [Read more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_nvfuser")` -  nvFuser with AotAutograd. [Read more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_cudagraphs")` - cudagraphs with AotAutograd. [Read more](https://github.com/pytorch/torchdynamo/pull/757)

**Inference-only backend**s:
* `dynamo.optimize("ofi")` -  Uses Torchscript optimize_for_inference.  [Read more](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
* `dynamo.optimize("fx2trt")` -  Uses Nvidia TensorRT for inference optimizations.  [Read more](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
* `dynamo.optimize("onnxrt")` -  Uses ONNXRT for inference on CPU/GPU.  [Read more](https://onnxruntime.ai/)
* `dynamo.optimize("ipex")` -  Uses IPEX for inference on CPU.  [Read more](https://github.com/intel/intel-extension-for-pytorch)

## Why yet another way of optimizing PyTorch code?
* `torch.jit.trace()` is silently wrong if it cannot trace e.g: during control flow
* `torch.jit.script()` requires modifications to user or library code by adding type annotations and removing non PyTorch code
* `torch.fx.symbolic_trace()` either traces correctly or gives a hard error but it's limited to traceable code so still can't handle control flow
* `torch._dynamo` works out of the box and produces partial graphs. It still has the option of producing a single graph with `nopython=True` which are needed for [some situations](./documentation/FAQ.md#do-i-still-need-to-export-whole-graphs) but allows a smoother transition where partial graphs can be optimized without code modification

## Next steps
* [Troubleshooting](./documentation/TROUBLESHOOTING.md)
* [FAQ](./documentation/FAQ.md)
* [Add your own backend](./documentation/custom-backend.md)

## License

TorchDynamo has a BSD-style license, as found in the LICENSE file.
