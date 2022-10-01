# TorchDynamo

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

### Install nightly binaries

TorchDynamo is evolving very quickly and so we only provide binaries based
on a nightly version of PyTorch.
To use GPU back ends (and in particular Triton), please make sure that the cuda
that you have installed locally matches the PyTorch version you are running. For
the command below, you will need CUDA 11.7.

```shell
pip3 install --pre torch==1.13.0.dev20220930+cu117 --extra-index-url https://download.pytorch.org/whl/nightly/cu117
pip install -U "git+https://github.com/openai/triton@998fd5f9afe166247f441999c605dfe624ca9331#subdirectory=python"
pip install -U "git+https://github.com/pytorch/torchdynamo"
```

### Install from local source

You can also install PyTorch, Triton and or Dynamo from source at the same
commits as the ones listed above. The Makefile target `make setup_nightly_gpu`
contain the commands used by our CI to setup dependencies.
Note that only CUDA 11.6+ is officially supported.

Other development requirements can be installed with:
```shell
pip install -r requirements.txt
```

Install TorchDynamo with:
```shell
python setup.py develop
```

## Usage Example

Here is a basic example of how to use TorchDynamo. One can decorate a function
or a method using `torchdynamo.optimize` to enable TorchDynamo optimization.

```py
import torch
import torchdynamo

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

@torchdynamo.optimize(my_compiler)
def fn(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b

fn(torch.randn(10), torch.randn(10))
```

Running the above example produces this output

```
my_compiler() called with FX graph:
opcode         name    target                                                  args        kwargs
-------------  ------  ------------------------------------------------------  ----------  --------
placeholder    x       x                                                       ()          {}
placeholder    y       y                                                       ()          {}
call_function  cos     <built-in method cos of type object at 0x7f1a894649a8>  (x,)        {}
call_function  sin     <built-in method sin of type object at 0x7f1a894649a8>  (y,)        {}
call_function  add     <built-in function add>                                 (cos, sin)  {}
output         output  output                                                  ((add,),)   {}
```

This works for `torch.nn.Module` as well as shown below

```py
import torch
import torchdynamo

class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(torch.cos(x))

mod = MockModule()
optimized_mod = torchdynamo.optimize(my_compiler)(mod)
optimized_mod(torch.randn(10))
```

In the above examples, TorchDynamo uses a custom compiler (also referred to as
backend in the rest of the doc) `my_compiler` that just prints the Fx
GraphModule extracted by TorchDynamo's bytecode analysis, and returns the
`forward` callable. One could write new compilers in a similar fashion.

Let's take a look at one more example with control flow.
```py
from typing import List
import torch
import torchdynamo

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

@torchdynamo.optimize(my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

for _ in range(100):
    toy_example(torch.randn(10), torch.randn(10))
```

Running this example produces the following output:
```
my_compiler() called with FX graph:
opcode         name     target                                                  args              kwargs
-------------  -------  ------------------------------------------------------  ----------------  --------
placeholder    a        a                                                       ()                {}
placeholder    b        b                                                       ()                {}
call_function  abs_1    <built-in method abs of type object at 0x7f8d259298a0>  (a,)              {}
call_function  add      <built-in function add>                                 (abs_1, 1)        {}
call_function  truediv  <built-in function truediv>                             (a, add)          {}
call_method    sum_1    sum                                                     (b,)              {}
call_function  lt       <built-in function lt>                                  (sum_1, 0)        {}
output         output   output                                                  ((truediv, lt),)  {}

my_compiler() called with FX graph:
opcode         name    target                   args         kwargs
-------------  ------  -----------------------  -----------  --------
placeholder    b       b                        ()           {}
placeholder    x       x                        ()           {}
call_function  mul     <built-in function mul>  (b, -1)      {}
call_function  mul_1   <built-in function mul>  (x, mul)     {}
output         output  output                   ((mul_1,),)  {}

my_compiler() called with FX graph:
opcode         name    target                   args       kwargs
-------------  ------  -----------------------  ---------  --------
placeholder    b       b                        ()         {}
placeholder    x       x                        ()         {}
call_function  mul     <built-in function mul>  (x, b)     {}
output         output  output                   ((mul,),)  {}
```

Note that the order of the last two graphs is nondeterministic depending
on which one is encountered first by the just-in-time compiler.

### Existing Backends

TorchDynamo has a growing list of backends, which can be found in [backends.py]
or `torchdynamo.list_backends()`. Note many backends require installing
additional packages. Some of the most commonly used backends are

Debugging backends:
* `torchdynamo.optimize("eager")` - Uses PyTorch to run the extracted GraphModule. This is quite useful in debugging TorchDynamo issues.
* `torchdynamo.optimize("aot_eager")` - Uses AotAutograd with no compiler, i.e, just using PyTorch eager for the AotAutograd's extracted forward and backward graphs. This is useful for debugging, and unlikely to give speedups.

Training & inference backends:
* `torchdynamo.optimize("inductor")` - Uses TorchInductor backend with AotAutograd and cudagraphs.  [Read more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
* `torchdynamo.optimize("nvfuser")` -  nvFuser with TorchScript. [Read more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `torchdynamo.optimize("aot_nvfuser")` -  nvFuser with AotAutograd. [Read more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `torchdynamo.optimize("aot_cudagraphs")` - cudagraphs with AotAutograd. [Read more](https://github.com/pytorch/torchdynamo/pull/757)

Inference-only backends:
* `torchdynamo.optimize("ofi")` -  Uses Torchscript optimize_for_inference.  [Read more](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
* `torchdynamo.optimize("fx2trt")` -  Uses Nvidia TensorRT for inference optimizations.  [Read more](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
* `torchdynamo.optimize("onnxrt")` -  Uses ONNXRT for inference on CPU/GPU.  [Read more](https://onnxruntime.ai/)
* `torchdynamo.optimize("ipex")` -  Uses IPEX for inference on CPU.  [Read more](https://github.com/intel/intel-extension-for-pytorch)

### Training and AotAutograd

Torchdynamo supports training, using AotAutograd to capture backwards:
* only the .forward() graph is captured by torchdynamo's python evalframe frontend
* for each segment of .forward() that torchdynamo captures, it uses AotAutograd to generate a backward graph segment
* each pair of forward, backward graph are (optionally) min-cut partitioned to save the minimal state between forward/backward
* the forward, backward pairs are wrapped in autograd.function modules
* usercode calling .backward() still triggers eager's autograd engine, which runs each 'compiled backward' graph as if it were one op, also running any non-compiled eager ops' .backward() functions

Current limitations:
* optimizer ops are currently not captured at all, and thus not compiled (under investigation to add support)
* DDP and FSDP, which rely on autograd 'hooks' firing between backward ops to schedule communications ops, may be pessimized by having all communication ops scheduled _after_ whole compiled regions of backwards ops (WIP to fix this)

Example
```py
model = ...
optimizer = ...

@torchdynamo.optimize("inductor")
def training_iteration(...):
    return model(...)

for _ in range(100):
    loss = training_iteration(...)
    loss.backward()
    optimizer.step()
```

## Troubleshooting
See [Troubleshooting](./documentation/TROUBLESHOOTING.md).

## Adding Backends

One could replace `my_compiler()` in the examples above with something that generates faster
code, for example one using [optimize_for_inference]:
```py
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    scripted = torch.jit.trace(gm, example_inputs)
    return torch.jit.optimize_for_inference(scripted)
```

TorchDynamo also includes many backends, which can be found in
[backends.py] or `torchdynamo.list_backends()`.  Note many backends
require installing additional packages.  You can combine these backends
together with code like:
```py
from torchdynamo.optimizations import BACKENDS

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    trt_compiled = BACKENDS["tensorrt"](gm, example_inputs)
    if trt_compiled is not None:
        return trt_compiled
    # first backend failed, try something else...

    cudagraphs_compiled = BACKENDS["cudagraphs"](gm, example_inputs)
    if cudagraphs_compiled is not None:
        return cudagraphs_compiled

    return gm.forward
```

[optimize_for_inference]: https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html
[backends.py]: https://github.com/pytorch/torchdynamo/blob/main/torchdynamo/optimizations/backends.py

## Guards

TorchDynamo operates just-in-time and specializes graphs based on dynamic
properties.  For example, the first graph above has the following guards:
```
GUARDS:
 - local 'a' TENSOR_MATCH
 - local 'b' TENSOR_MATCH
 - global 'torch' FUNCTION_MATCH
```

If any of those guards fail, the graph will be recaptured and recompiled.
The interesting guard type there is `TENSOR_MATCH`, which checks the
following torch.Tensor properties:
- Python class of the tensor (tensor subclassing, etc)
- dtype
- device
- requires_grad
- dispatch_key (with thread-local includes/excludes applied)
- ndim
- sizes* (optional)
- strides* (optional)

*For sizes/strides you can disable this specialization by setting:
```py
torchdynamo.config.dynamic_shapes = True
```

The full specialization mode allows the backend compiler to assume
an entirely static graph.  Unfortunately, most backends require this.
Operators which return dynamic shapes will trigger a graph break when
not in dynamic shape mode.

## Run Mode / Quiescence Guarantee

In some cases, you may not want unexpected compiles after a program
has warmed up.  For example, if you are serving production traffic in a
latency critical application.  For this, TorchDynamo provides an alternate
mode where prior compiled graphs are used, but no new ones are generated:
```py
frozen_toy_example = torchdynamo.run(toy_example)
frozen_toy_example(torch.randn(10), torch.randn(10))
```

## Single Whole-Program Graph Mode

In some cases, you may want to ensure there are no graph breaks in your
program to debug performance issues.  You can turn graph breaks into
errors by setting
`nopython=True`:
```py
@torchdynamo.optimize(my_compiler, nopython=True)
def toy_example(a, b):
```

Which will trigger the following error in the example program above:
```py
Traceback (most recent call last):
  ...
torchdynamo.exc.Unsupported: generic_jump TensorVariable()
Processing original code:
  File "example.py", line 7, in toy_example
    if b.sum() < 0:
```

## Developer Setup

As background reading, I'd suggest looking at the
[PyTorch](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md),
[functorch](https://github.com/pytorch/functorch), and
[TorchBench](https://github.com/pytorch/benchmark#installation)
setup docs.  Since these projects work together in different ways.

The following instructions use [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```
conda create --name=torchdynamo python=3.8
conda activate torchdynamo

# install pytorch nightlies
# for CUDA version, replace `cpuonly` with `pytorch-cuda=11.6`
conda install pytorch torchvision torchaudio torchtext cpuonly -c pytorch-nightly
pip install -v "git+https://github.com/pytorch/pytorch.git@`python -c "import torch.version; print(torch.version.git_version)"`#subdirectory=functorch"

git clone https://github.com/pytorch/torchdynamo
cd torchdynamo
pip install -r requirements.txt

# check if everything works
make test
```

If see errors about missing symbols from `guards.so`, that may mean your
C++ compiler is incompatible CUDA and/or with the one used to compile
PyTorch.  You may need to change your compiler version or build PyTorch
from source.

## Tests

Run tests with
```shell
pytest test
```

To debug a specific test (with more debug prints) do:
```shell
pytest -vsk <test name>
```

Test on torchbenchmark models with:
```shell
python benchmarks/torchbench.py
```

## Linting and Automatic Code Formatting

[![Lint](https://github.com/pytorch/torchdynamo/actions/workflows/lint.yml/badge.svg)](https://github.com/pytorch/torchdynamo/actions/workflows/lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Install format/linter deps with `pip install -r requirements.txt`, then:
```shell
make format  # reformat all files (in-place)
make lint    # run the linters
```

## License

TorchDynamo has a BSD-style license, as found in the LICENSE file.
