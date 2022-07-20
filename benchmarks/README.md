# Torchdynamo Benchmarks

## What We Benchmark
TorchDynamo provides a benchmark harness that takes care of uniformly benchmarking different models.  It interleaves runs of eager and dynamo to avoid machine noise/variability issues, and reports results based on medians along with P-values. 

The runner integrates with models from TorchBenchmark, HuggingFace and TIMM suites and covers both training and inference.

The infrastructure allows us to specify a loss function. For torchbench models, we use .sum().backward() call in place of the native loss function. For TIMM models, we use a CrossEntropy loss. And HF models contain a loss function inside the model itself, so we don't need any special loss computation handling.

Training benchmarks approximate training by running the model forward, computing loss and then running backward. We entirely skip the optimizer step today.

Inference benchmarks and Training benchmarks measure correctness by comparing dynamo and eager model outputs given fixed inputs and seeds.

## Setup

### Machine
We run benchmarks on AWS machines (p4d.24xlarge) using 8xNVidia A100 40GB cards.  We suggest using Cuda 11.6 for consistency.

### Benchmarks
Make sure to carefully follow the [torchbench installation](https://github.com/pytorch/benchmark#installation) instructions, taking care to build the auxiliary libraries (torchvision, torchtext) from a matching version to your pytorch version.

For HF and TIMM models, the scripts already install the transformers and timm package respectively on the first run. 

## Runbook
There are two ways to run the benchmarks.

First, one could directly call torchbench.py, huggingface.py or timm_models.py with the necessary flags. There are a lot of flags in the benchmarks runner. Some of the examples are as follows.

**Inference Commands**
* TorchScript NVFuser Inference - `python benchmarks/torchbench.py -dcuda --isolate -n100 --speedup-ts`
* TorchInductor CUDA Graphs Inference - `python benchmarks/torchbench.py -dcuda --inductor-settings --float32 -n50 --inductor`

**Training Commands**
* Torchscript (with TorchDynamo capture) NVFuser Training - `python benchmarks/timm_models.py --float32 -dcuda --training --nvfuser --speedup-dynamo-ts --use-eval-mode --isolate`
* AOTAutograd Torchscript NVFuser Training - `python benchmarks/timm_models.py --float32 -dcuda --training --nvfuser --accuracy-aot-ts-mincut --use-eval-mode --isolate`


As evident from the above scripts, there are a lot of flags. In order to support apples-to-apples comparison, we provide another wrapper over the benchmark infra called runner.py. We will update the commands for the latest and most relevant compilers in runner.py. runner.py also has graph utilities to visualize and compare results. Some of the example commands are

* Training compilers on TIMM models - `python benchmarks/runner.py --suites=timm_models --training --dtypes=float32 --output-dir=timm_logs`
* AOTAutograd Training compiler on TIMM models - `python benchmarks/runner.py --suites=timm_models --training --dtypes=float32 --compilers=aot_nvfuser --output-dir=timm_logs`

### Training
* TorchInductor + Triton
* NVFuser

### Inference
TODO (which configs do we advertise?)
