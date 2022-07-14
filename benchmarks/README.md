# Torchdynamo Benchmarks

## What We Benchmark
TorchDynamo provides a benchmark harness that takes care of uniformly benchmarking different models.  It interleaves runs of eager and dynamo to avoid machine noise/variability issues, and reports results based on medians along with P-values. 

The runner integrates with models from TorchBenchmark, HuggingFace and TIMM suites and covers both training and inference.

Training benchmarks approximate training by running the model forward, using a .sum().backward() call in place of the native loss function and skips the optimizer entirely.

Inference benchmarks and Training benchmarks measure correctness by comparing dynamo and eager model outputs given fixed inputs and seeds.

## Runbook
There are a lot of flags in the benchmark runner, and it can be confusing to know which settings to use or what machine to run it on.  In order to support apples-to-apples comparison, we have provided the following 'standard' settings.

We run benchmarks on AWS machines using 8xNVidia A100 64GB cards and (TODO) CPU.  (TODO instance type and OS/cuda versions) 

### Training
* TorchInductor + Triton
* NVFuser


### Inference
TODO (which configs do we advertise?)
