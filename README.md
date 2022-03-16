# TorchDynamo

TorchDynamo is a Python-level JIT compiler designed to make unmodified
PyTorch programs faster. TorchDynamo hooks into the frame evaluation API
in CPython ([PEP 523]) to dynamically modify Python bytecode right before
it is executed. It rewrites Python bytecode in order to extract sequences
of PyTorch operations into an [FX Graph] which is then just-in-time
compiled with an ensemble of different backends and autotuning. It
creates this FX Graph through bytecode analysis and is designed to mix
Python execution with compiled backends to get the best of both worlds:
usability and performance.

[PEP 523]: https://www.python.org/dev/peps/pep-0523/
[FX Graph]: https://pytorch.org/docs/stable/fx.html

![](TorchDynamo.png)

For more information see progress updates posted on dev-discuss.pytorch.org:
- [Update 1: An Experiment in Dynamic Python Bytecode Transformation](https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361)
- [Update 2: 1.48x Geomean Speedup on TorchBench CPU Inference](https://dev-discuss.pytorch.org/t/torchdynamo-update-1-48x-geomean-speedup-on-torchbench-cpu-inference/397)
- [Update 3: GPU Inference Edition](https://dev-discuss.pytorch.org/t/torchdynamo-update-3-gpu-inference-edition/460)
- [Update 4: LazyTensor & nvFuser Experiments](https://dev-discuss.pytorch.org/t/torchdynamo-update-4-lazytensor-nvfuser-experiments/496)
- [Update 5: Improved Capture & Bigger Graphs](https://dev-discuss.pytorch.org/t/torchdynamo-update-5-improved-capture-bigger-graphs/556)

*TorchDynamo is experimental* and under active development.
You are welcome to try it out and contribute, but should expect to find
bugs and rough edges.

## Requirements and Setup

*Python 3.8* is highly recommended.  Python 3.7 works, but is only
sporadically tested and has lower coverage.  Python 3.9+ does not work,
but should be supportable with minor changes.

*PyTorch*'s main branch contains some fixes that improve TorchDynamo
support, so we recommend building [PyTorch from source] or using PyTorch
nightly builds.

[PyTorch from source]: https://github.com/pytorch/pytorch#from-source

For reproducing the experiments in the posts above, use the TorchBenchmark
[fork found here].  This fork contains a few minor fixes that have not
yet been merged upstream.

[fork found here]: https://github.com/jansel/benchmark

Other development requirements can be installed with:
```shell
pip install -r requirements.txt
```

Install TorchDynamo with:
```shell
python setup.py develop
```


## Usage Example

Here is a basic example of how to use TorchDynamo:
```py
from typing import List
import torch
import torchdynamo

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

with torchdynamo.optimize(my_compiler):
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


## Adding Backends

One could replace `my_compiler()` with something that generates faster
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

If you just want to use an existing backend, you can pass a
string containing the backend name to `torchdynamo.optimize()`.
`torchdynamo.optimize()` can also be used as a decorator on functions,
methods, or nn.Modules().  So a shorter version of using [optimize_for_inference] on `toy_example` would be:

```py
@torchdynamo.optimize("ofi")
def toy_example(a, b):
    ...
```

[optimize_for_inference]: https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html
[backends.py]: https://github.com/jansel/torchdynamo/blob/main/torchdynamo/optimizations/backends.py

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
with torchdynamo.run():
    toy_example(torch.randn(10), torch.randn(10))
```

## Single Whole-Program Graph Mode

In some cases, you may want to ensure there are no graph breaks in your
program to debug performance issues.  You can turn graph breaks into
errors by setting
`nopython=True`:
```py
with torchdynamo.optimize(my_compiler, nopython=True):
    toy_example(torch.randn(10), torch.randn(10))
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

## Deeper Dive

If you want to understand better what TorchDynamo is doing, you can set:
```py
torchdynamo.config.debug = True
```

which triggers useful (but spammy) printouts.

For example, the printouts for the first graph in the `toy_example`
above are:
```
__compiled_fn_0 <eval_with_key>.1
opcode         name     target                                                  args              kwargs
-------------  -------  ------------------------------------------------------  ----------------  --------
placeholder    a        a                                                       ()                {}
placeholder    b        b                                                       ()                {}
call_function  abs_1    <built-in method abs of type object at 0x7f9ca082f8a0>  (a,)              {}
call_function  add      <built-in function add>                                 (abs_1, 1)        {}
call_function  truediv  <built-in function truediv>                             (a, add)          {}
call_method    sum_1    sum                                                     (b,)              {}
call_function  lt       <built-in function lt>                                  (sum_1, 0)        {}
output         output   output                                                  ((truediv, lt),)  {}

ORIGINAL BYTECODE toy_example example.py 9
 10           0 LOAD_FAST                0 (a)
              2 LOAD_GLOBAL              0 (torch)
              4 LOAD_METHOD              1 (abs)
              6 LOAD_FAST                0 (a)
              8 CALL_METHOD              1
             10 LOAD_CONST               1 (1)
             12 BINARY_ADD
             14 BINARY_TRUE_DIVIDE
             16 STORE_FAST               2 (x)

 11          18 LOAD_FAST                1 (b)
             20 LOAD_METHOD              2 (sum)
             22 CALL_METHOD              0
             24 LOAD_CONST               2 (0)
             26 COMPARE_OP               0 (<)
             28 POP_JUMP_IF_FALSE       38

 12          30 LOAD_FAST                1 (b)
             32 LOAD_CONST               3 (-1)
             34 BINARY_MULTIPLY
             36 STORE_FAST               1 (b)

 13     >>   38 LOAD_FAST                2 (x)
             40 LOAD_FAST                1 (b)
             42 BINARY_MULTIPLY
             44 RETURN_VALUE

MODIFIED BYTECODE
  9           0 LOAD_GLOBAL              3 (__compiled_fn_0)
              2 LOAD_FAST                0 (a)
              4 LOAD_FAST                1 (b)
              6 CALL_FUNCTION            2
              8 UNPACK_SEQUENCE          2
             10 STORE_FAST               2 (x)
             12 POP_JUMP_IF_FALSE       24
             14 LOAD_GLOBAL              4 (__resume_at_30_1)
             16 LOAD_FAST                1 (b)
             18 LOAD_FAST                2 (x)
             20 CALL_FUNCTION            2
             22 RETURN_VALUE
        >>   24 LOAD_GLOBAL              5 (__resume_at_38_2)
             26 LOAD_FAST                1 (b)
             28 LOAD_FAST                2 (x)
             30 CALL_FUNCTION            2
             32 RETURN_VALUE

GUARDS:
 - local 'a' TENSOR_MATCH
 - local 'b' TENSOR_MATCH
 - global 'torch' FUNCTION_MATCH
```

At the top you can see the FX graph (which we already shared above).
Next you see the original bytecode of the function, followed by the
modified bytecode generated by TorchDynamo.  Finally, you see the guards
which we covered above.

In the modified bytecode `__compiled_fn_0` is the return value
of `my_compiler()` (the compiled graph). `__resume_at_30_1` and
`__resume_at_38_2` are both generated continuation functions that pick up
execution after a graph break (at bytecode offsets 30 and 38).  Each of
these functions take the form:
```
__resume_at_<offset>:
    ... restore stack state if needed ...
    JUMP_ABSOLUTE <offset> into toy_example
    ... original bytecode of toy_example ...
```

By generating these resume_at function we force the remainder of the
function to be executed in a new Python frame which recursively will
trigger TorchDynamo to re-start its capture once execution reaches that
point for the first time.


## Tests

[![Test Python 3.8](https://github.com/facebookresearch/torchdynamo/actions/workflows/test-py38.yml/badge.svg)](https://github.com/facebookresearch/torchdynamo/actions/workflows/test-py38.yml)

Run tests with
```shell
pytest tests
```

To debug a specific test (with more debug prints) do:
```shell
pytest -vsk <test name>
```

Test on torchbenchmark models with:
```shell
python torchbench.py
```

## Performance Measurement

To reproduce the performance measurements shared in the posts above,
run either `make offline-autotune-cpu` or `make offline-autotune-gpu`.  These targets
will run something like the following:

```shell
# cleanup leftover state
rm -rf subgraphs

# initial run to record all the graphs to ./subgraphs/*
python torchbench.py -dcuda --speedup -n1

# autotune each graph and store which backend is best on disk
python autotune.py

# measure the speedups using the autotuned backend choices
python torchbench.py -dcuda --speedup -n100

# results are in ./speedups.csv
```

The baselines can be run with `make baseline-cpu` or `make baseline-gpu`.
Which both string together a lot of calls to `./torchbench.py` and
generate `*.csv` files.  See `./torchbench.py --help` for more options.


## Linting and Automatic Code Formatting

[![Lint](https://github.com/facebookresearch/torchdynamo/actions/workflows/lint.yml/badge.svg)](https://github.com/facebookresearch/torchdynamo/actions/workflows/lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Install format/linter deps with `pip install -r requirements.txt`, then:
```shell
make format  # reformat all files (in-place)
make lint    # run the linters
```

## License

TorchDynamo has a BSD-style license, as found in the LICENSE file.
