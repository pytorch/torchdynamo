# Torchdynamo Benchmarks

## What We Benchmark
TorchDynamo provides a benchmark harness that takes care of uniformly benchmarking different models.  It interleaves runs of eager and dynamo to avoid machine noise/variability issues, and reports results based on medians along with P-values. 

The runner integrates with models from TorchBenchmark, HuggingFace and TIMM suites and covers both training and inference.

Training benchmarks approximate training by running the model forward, using a .sum().backward() call in place of the native loss function and skips the optimizer entirely.

Inference benchmarks and Training benchmarks measure correctness by comparing dynamo and eager model outputs given fixed inputs and seeds.

## Setup

### Machine
We run benchmarks on AWS machines (p4d.24xlarge) using 8xNVidia A100 40GB cards.  We suggest using Cuda 11.6 for consistency.

### Benchmarks
Make sure to carefully follow the [torchbench installation](https://github.com/pytorch/benchmark#installation) instructions, taking care to build the auxiliary libraries (torchvision, torchtext) from a matching version to your pytorch version.

TODO(HF, TIMM instructions)

## Runbook
There are a lot of flags in the benchmark runner, and it can be confusing to know which settings to use or what machine to run it on.  In order to support apples-to-apples comparison, we have provided the following 'standard' settings.

TODO - update run commands after (https://github.com/pytorch/torchdynamo/pull/585) lands

### Training
* TorchInductor + Triton
* NVFuser

### Inference
TODO (which configs do we advertise?)
