#!/usr/bin/env python
import argparse
import collections
import copy
import csv
import functools
import gc
import io
import itertools
import logging
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import gmean
from scipy.stats import ttest_ind
from torch.utils._pytree import tree_map

import torchdynamo
import torchdynamo.utils
from torchdynamo.optimizations import backends
from torchdynamo.optimizations.inference import fixed_strategy1
from torchdynamo.optimizations.inference import fixed_strategy2
from torchdynamo.optimizations.inference import offline_autotuner
from torchdynamo.optimizations.inference import online_autotuner
from torchdynamo.optimizations.log_args import conv_args_analysis
from torchdynamo.optimizations.python_key import python_key
from torchdynamo.profiler import Profiler
from torchdynamo.profiler import fx_insert_profiling
from torchdynamo.testing import dummy_fx_compile
from torchdynamo.testing import format_speedup
from torchdynamo.testing import same

try:
    from functorch._src.aot_autograd import set_model_name
except ImportError:

    def set_model_name(name):
        pass


log = logging.getLogger(__name__)

# We are primarily interested in TF32
torch.backends.cuda.matmul.allow_tf32 = True

current_name = ""
current_device = ""
current_batch_size = None
output_filename = None


def output_csv(filename, headers, row):
    assert filename
    existed = os.path.exists(filename)
    output = csv.writer(
        io.TextIOWrapper(
            open(filename, "ab", buffering=0),
            "utf-8",
            write_through=True,
        ),
        lineterminator="\n",
    )
    if not existed:
        output.writerow(headers)
    output.writerow([(f"{x:.4f}" if isinstance(x, float) else x) for x in row])


class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def synchronize():
    pass


def print_summary(filename):
    if not (filename and os.path.exists(filename)):
        return
    data = pd.read_csv(filename)
    width = max(map(len, data.columns))
    for col in data.columns:
        try:
            if col in ("dev", "name", "batch_size"):
                continue
            elif col in ("pct_ops", "pct_time"):
                print(col.ljust(width), f"{data[col].mean():.1%}")
            elif col in ("graphs", "graph_calls", "captured_ops", "total_ops"):
                print(col.ljust(width), f"{data[col].mean():.1f}")
            else:
                cdata = data[col].clip(1)
                print(
                    col.ljust(width),
                    f"gmean={gmean(cdata):.2f}x mean={cdata.mean():.2f}x",
                )
        except Exception:
            pass


def timed(model, model_iter_fn, example_inputs, times=1, return_result=False):
    synchronize()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    # Dont collect outputs to correctly measure timing
    for _ in range(times):
        result = model_iter_fn(model, example_inputs, collect_outputs=False)
        synchronize()
    t1 = time.perf_counter()
    return (t1 - t0, result) if return_result else t1 - t0


class Stats:
    totals = collections.defaultdict(collections.Counter)

    @classmethod
    def reset_counters(cls):
        for k, v in torchdynamo.utils.counters.items():
            cls.totals[k].update(v)
        ok = torchdynamo.utils.counters["frames"]["ok"]
        total = torchdynamo.utils.counters["frames"]["total"]
        torchdynamo.utils.counters.clear()
        return ok, total

    @classmethod
    def print_summary(cls):
        for k, v in sorted(cls.totals.items()):
            lines = "\n  ".join(map(str, v.most_common(50)))
            print(f"STATS {k}\n  {lines}")

    @classmethod
    def aot_summary(cls):
        return [cls.totals["aot_autograd"]["total"], cls.totals["aot_autograd"]["ok"]]


def coverage_experiment(args, model_iter_fn, model, example_inputs, start_latency):
    """
    Test operator/model coverage of TorchDynamo and record statistics
    taken from a profiler.  This target is mainly intended to check
    correctness.

    Writes to ./coverage.csv
    """
    profiler = Profiler()
    with profiler.prof, torchdynamo.run():
        model_iter_fn(model, example_inputs)
    coverage_result = profiler.results()
    output_csv(
        output_filename,
        (
            "dev",
            "name",
            "batch_size",
            "graphs",
            "graph_calls",
            "captured_ops",
            "total_ops",
            "pct_ops",
            "pct_time",
            "start_latency",
        ),
        [
            current_device,
            current_name,
            current_batch_size,
        ]
        + coverage_result.tocsv()
        + [
            start_latency,
        ],
    )
    return coverage_result


def speedup_experiment_fx2trt(args, model_iter_fn, model, example_inputs):
    """
    Measure speedups over eager using the trt inference backend. TRT backend is based fx graph
    generated by torchdynamo.
    Writes to ./speedups_fx2trt.csv
    """
    return speedup_experiment(args, model_iter_fn, model, example_inputs)


def recompile_profiler_experiment(args, model_iter_fn, model, example_inputs):
    prof = torchdynamo.utils.CompileProfiler()
    with torchdynamo.optimize(prof, nopython=args.nopython):
        model_iter_fn(model, example_inputs)
    output_csv(
        output_filename, ["model", "profiler report"], [current_name, prof.report()]
    )
    met = prof.get_metrics()
    guard_failures = len(met["guard_failures"])
    return [guard_failures]


def randomize_input(inputs):
    if isinstance(inputs, (list, tuple)):
        return type(inputs)([randomize_input(x) for x in inputs])
    elif isinstance(inputs, torch.Tensor):
        if inputs.dtype in (torch.float32, torch.float64):
            torchdynamo.utils.counters["randomize_input"]["times"] += 1
            return torch.randn_like(inputs)
        elif inputs.dtype == torch.int64:
            # Note: we can not simply tune integer tensors as follows
            #   `return torch.randint_like(inputs, high=inputs.max().item())`
            # This may break some invariants between tensors.
            # E.g. in embedding lookup case, one tensor is the length
            # and another is an indices tensor.
            return inputs
        else:
            raise RuntimeError(
                f"randomize_input need support tensor of type {inputs.dtype}"
            )
    else:
        raise RuntimeError(
            f"randomize_input can not handle input of type {type(inputs)}"
        )


def cold_start_experiment(args, model_iter_fn, model, example_inputs, optimize_ctx):
    compile_iters = 2
    total_iters = compile_iters + 2
    timings = np.zeros((total_iters, 2), np.float64)
    # if we randomize the input, we should also check the result is correct
    should_check_result = should_randomize_input = args.randomize_input
    is_correct = True

    optimized_model_iter_fn = optimize_ctx(model_iter_fn)
    for rep in range(total_iters):
        inputs = (
            randomize_input(copy.deepcopy(example_inputs))
            if should_randomize_input
            else example_inputs
        )

        # interleave the runs to handle frequency scaling and load changes
        timings[rep, 0], expected_output = timed(
            model, model_iter_fn, inputs, return_result=True
        )
        timings[rep, 1], actual_output = timed(
            model, optimized_model_iter_fn, inputs, return_result=True
        )
        if should_check_result:
            is_correct = is_correct and same(expected_output, actual_output)
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    worst = np.max(timings, axis=0)

    def breakeven(dynamo_times, eager_times):
        """
        Solve for the number of iterations it takes dynamo to 'catch up' with eager,
        taking into account the time it spent compiling.  Assumes all compilation
        happens up front and the model is static thereafter, which is definitely not
        true in general but might be across torchbench.

            dc1, dc2 = dynamo compilation iterations (with Prof Exec)
            d, e = dynamo, eager warmed up iteration
            B = num iters to break even
            dc1 + dc2 + (B-2)d = B*e
            B = (dc1 + dc2 - 2d) / (e - d)
        """
        dc1, dc2, d = dynamo_times[0], dynamo_times[1], np.median(dynamo_times[2:])
        e = np.median(eager_times)
        if d < e:
            return (dc1 + dc2 + 2 * d) / (e - d)
        else:
            # if optimized dynamo is not faster than eager we'll compute
            # a nonsense negative number
            return 0

    speedup = worst[0] / worst[1]
    eager_times, dynamo_times = timings[:, 0], timings[:, 1]
    output_csv(
        output_filename,
        ("dev", "name", "batch_size", "cold-start speedup", "breakeven iters"),
        [
            current_device,
            current_name,
            current_batch_size,
            float(speedup),
            breakeven(dynamo_times, eager_times),
        ],
    )

    def format_speedup(
        speedup, pvalue, breakeven_iters, is_correct=True, pvalue_threshold=0.1
    ):
        if not is_correct:
            return "ERROR"
        if pvalue > pvalue_threshold:
            return f"{speedup:.3f}x breakeven={breakeven_iters:.2f} iters SAME"
        return f"{speedup:.3f}x breakeven={breakeven_iters:.2f} iters p={pvalue:.2f}"

    return format_speedup(
        speedup, pvalue, breakeven(dynamo_times, eager_times), is_correct=is_correct
    )


def speedup_experiment(args, model_iter_fn, model, example_inputs):
    """
    Measure speedups over eager using the autotuning inference backend.  To use this:
        1) First run once to record graphs that need autotuning
        2) Next run ./autotune.py to select the right backend for each recorded graph
        3) Finally, run this target again to measure speedups

    Writes to ./speedups.csv
    """
    if args.dynamic_shapes:
        return speedup_experiment_ds(args, model_iter_fn, model, example_inputs)

    timings = np.zeros((args.repeat, 2), np.float64)
    # if we randomize the input, we should also check the result is correct
    should_check_result = should_randomize_input = args.randomize_input
    is_correct = True

    import contextlib

    @contextlib.contextmanager
    def maybe_profile(*args, **kwargs):
        if kwargs.pop("enabled", True):
            with torch.profiler.profile(*args, **kwargs) as p:
                yield p
        else:
            yield

    with maybe_profile(enabled=args.export_profiler_trace) as p:
        for rep in range(args.repeat):
            inputs = (
                randomize_input(copy.deepcopy(example_inputs))
                if should_randomize_input
                else example_inputs
            )

            # interleave the runs to handle frequency scaling and load changes
            timings[rep, 0], expected_output = timed(
                model, model_iter_fn, inputs, return_result=True
            )
            with torchdynamo.run():
                timings[rep, 1], actual_output = timed(
                    model, model_iter_fn, inputs, return_result=True
                )
            if should_check_result:
                is_correct = is_correct and same(expected_output, actual_output)
    if args.export_profiler_trace:
        name = args.profiler_trace_name + "_" + model.name + ".json"
        name = os.path.join(torchdynamo.config.base_dir, name)
        p.export_chrome_trace(name)
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    if args.dump_raw_metrics:
        np.save(
            f"{output_filename[:-4]}-raw_timings-{current_name}-{current_device}.npy",
            timings,
        )
    output_csv(
        output_filename,
        ("dev", "name", "batch_size", "speedup"),
        [current_device, current_name, current_batch_size, float(speedup)],
    )
    return format_speedup(speedup, pvalue, is_correct=is_correct)


def speedup_experiment_ds(args, model_iter_fn, model, example_inputs):
    """
    Run dynamic shapes benchmarks.

    Requires dynamic shape compatible models, which provide a list of example inputs.

    Warms up using the first input example and then iterates the inputs,
    measuring (and expecting minimal) variance between the runtime for different examples.

    """
    timings = np.zeros((args.repeat, len(example_inputs), 2), np.float64)

    if args.repeat > 5:
        print(
            f"\ndynamic shapes experiments are slow, consider setting --repeat less than {args.repeat}\n"
        )

    nwarmup = 4
    for rep in range(args.repeat):
        # Start each rep fresh, e.g. only warmup on example 0
        torchdynamo.reset()
        optimized_model_iter_fn = optimize_ctx(model_iter_fn)
        for _ in range(nwarmup):
            optimized_model_iter_fn(model, example_inputs[0])

        for input_idx, inputs in enumerate(example_inputs):
            # interleave the runs to handle frequency scaling and load changes
            timings[rep, input_idx, 0] = timed(
                model, model_iter_fn, inputs, return_result=False
            )
            # different from regular speedup_experiment, we _DO_ want to allow recompilation
            timings[rep, input_idx, 1] = timed(
                model, optimized_model_iter_fn, inputs, return_result=False
            )
    medians = np.median(timings, axis=0)
    speedups = list(medians[:, 0] / medians[:, 1])
    speedups_mean = np.mean(speedups)
    speedups_median = np.median(speedups)
    speedups_var = np.var(speedups)

    # TODO this x[0] is not going to work in general but bert only has 1 input
    shapes = [x[0].shape for x in example_inputs]
    shape_keys = sorted(set(shapes))
    shape_speedups = {
        shape: list(
            map(
                lambda it: it[1],
                filter(lambda it: it[0] == shape, zip(shapes, speedups)),
            )
        )
        for shape in shape_keys
    }
    output_str = (
        f"mean: {speedups_mean:.3f}, median: {speedups_median:.3f}, var: {speedups_var:.3f}"
        + "\nSpeedups by shape: "
        + "\n".join(
            [
                f"{shape}: "
                + ", ".join([f"{speedup: .3g}" for speedup in shape_speedups[shape]])
                for shape in shape_keys
            ]
        )
    )
    output_csv(
        output_filename,
        ("dev", "name", "batch_size", "speedup mean", "speedup median", "speedup var"),
        [
            current_device,
            current_name,
            current_batch_size,
            speedups_mean,
            speedups_median,
            speedups_var,
        ],
    )
    return output_str


def overhead_experiment(*args, model_iter_fn):
    """
    Measure overheads of TorchDynamo by running with no backend (only
    eager+FX), and reporting speedup/slowdown over eager.

    Writes to ./overheads.csv
    """
    return speedup_experiment(*args, model_iter_fn)


def print_fx(gm, example_inputs):
    print(gm.graph)
    return gm


def print_aten_ops(gm, example_inputs):
    from functorch.compile import aot_module

    def trace_printer(gm, _):
        print(gm.graph)
        return gm

    return aot_module(gm, fw_compiler=trace_printer, bw_compiler=trace_printer)


def baselines(models, model_iter_fn, example_inputs, args):
    """
    Common measurement code across all baseline experiments.
    """
    models = list(models)
    for idx, (name, model) in enumerate(models):
        if idx == 0:
            result0 = model_iter_fn(model, example_inputs)
        elif model is not None:
            try:
                result = model_iter_fn(model, example_inputs)
                if same(result0, result):
                    continue
                print(name, "is INCORRECT")
            except Exception:
                log.exception("error checking %s", name)
            models[idx] = (name, None)
    timings = np.zeros((args.repeat, len(models)), np.float64)
    timings.fill(1.0e10)
    for rep in range(args.repeat):
        for idx, (name, model) in enumerate(models):
            if model is not None:
                try:
                    timings[rep, idx] = timed(model, model_iter_fn, example_inputs)
                except Exception:
                    pass
    pvalue = [
        ttest_ind(timings[:, 0], timings[:, i]).pvalue
        for i in range(1, timings.shape[1])
    ]
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1:]
    for idx, (name, model) in enumerate(models[1:]):
        if model is None:
            speedup[idx] = 0.0
    result = " ".join(
        [
            format_speedup(s, p, m is not None)
            for s, p, m in zip(speedup, pvalue, [m for n, m in models[1:]])
        ]
    )
    output_csv(
        output_filename,
        ("dev", "name", "batch_size") + tuple(n for n, m in models[1:]),
        [current_device, current_name, current_batch_size]
        + [f"{x:.4f}" for x in speedup],
    )
    return result


def try_script(model, example_inputs):
    try:
        return torch.jit.script(model)
    except Exception:
        return None


def speedup_experiment_ts(args, model_iter_fn, model, example_inputs):
    """
    Measure baseline performance (without using TorchDynamo) of TorchScript and optimize_for_inference.

    Writes to ./baseline_ts.csv
    """
    if args.training:
        return baselines(
            [
                ("eager", model),
                ("ts", try_script(model, example_inputs)),
            ],
            model_iter_fn,
            example_inputs,
            args,
        )

    return baselines(
        [
            ("eager", model),
            ("ts", try_script(model, example_inputs)),
            (
                "ofi",
                backends.ofi(try_script(model, example_inputs), example_inputs),
            ),
            # ("nnc", backends.nnc(try_script(model, example_inputs), example_inputs)),
            # ("nvfuser", backends.nvfuser(try_script(model, example_inputs), example_inputs)),
        ],
        model_iter_fn,
        example_inputs,
        args,
    )


def speedup_experiment_sr(args, model_iter_fn, model, example_inputs):
    """
    Measure baseline performance (without using TorchDynamo) of static runtime.

    Writes to ./baseline_sr.csv
    """

    if current_name not in ("opacus_cifar10", "timm_nfnet", "hf_T5"):
        sr = backends.static_runtime(try_script(model, example_inputs), example_inputs)
    else:
        # segfaults on these models
        sr = None
    return baselines(
        [
            ("eager", model),
            (
                "sr",
                sr,
            ),
        ],
        model_iter_fn,
        example_inputs,
        args,
    )


def speedup_experiment_onnx(args, model_iter_fn, model, example_inputs):
    """
    Measure baseline performance (without using TorchDynamo) of ONNXRT and TensorFlow.

    Writes to ./baseline_onnx.csv
    """
    if current_device == "cpu":
        m_onnxrt = backends.onnxrt_cpu(
            try_script(model, example_inputs), example_inputs
        )
    else:
        m_onnxrt = backends.onnxrt_cuda(
            try_script(model, example_inputs), example_inputs
        )

    if current_name != "timm_resnest":
        m_onnx2tf = backends.onnx2tf(try_script(model, example_inputs), example_inputs)
    else:
        # this one takes 8+ hours to finish
        m_onnx2tf = None

    return baselines(
        [
            ("eager", model),
            ("onnxrt", m_onnxrt),
            ("onnx2tf", m_onnx2tf),
        ],
        model_iter_fn,
        example_inputs,
        args,
    )


def speedup_experiment_trt(args, model_iter_fn, model, example_inputs):
    """
    Measure baseline performance (without using TorchDynamo) of TensorRT.

    Writes to ./baseline_trt.csv
    """
    m_onnx2trt = backends.onnx2tensorrt(
        try_script(model, example_inputs), example_inputs
    )

    m_torch2trt = backends.torch2trt(model, example_inputs)

    if current_name != "opacus_cifar10":
        m_fx2trt = backends.fx2trt(model, example_inputs)
    else:
        # fx2trt infinite loops on one model
        m_fx2trt = None

    return baselines(
        [
            ("eager", model),
            ("onnx2trt", m_onnx2trt),
            ("torch2trt", m_torch2trt),
            ("fx2trt", m_fx2trt),
        ],
        model_iter_fn,
        example_inputs,
        args,
    )


def read_batch_size_from_file(args, filename, model_name):
    batch_size = None
    if os.path.exists("benchmarks"):
        filename = os.path.join("benchmarks", filename)
    assert os.path.exists(filename), filename
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [i.split(",") for i in lines if len(i.strip()) > 0]
        for val in lines:
            cur_name, b = val
            if model_name == cur_name:
                batch_size = int(b)
    if batch_size is None:
        raise RuntimeError(
            f"Batch size could not be found for {model_name} in {args.batch_size_file}"
        )
    print(f"batch size: {batch_size}")
    return batch_size


def compilation_profiling_experiment(
    model_iter_fn, model, example_inputs, backend="pytorch"
):
    # Get the context
    if backend == "pytorch":
        ctx = NullContext()
    else:
        ctx = torchdynamo.optimize(backend)

    def get_peak_memory():
        return torch.cuda.max_memory_allocated() / 10**9

    try:
        # Reset and warmup
        torchdynamo.reset()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        with ctx:
            model_iter_fn(model, example_inputs)
            model_iter_fn(model, example_inputs)
            model_iter_fn(model, example_inputs)
        t1 = time.perf_counter()
        compilation_latency = t1 - t0

        # Measure memory
        torch.cuda.reset_peak_memory_stats()
        with ctx:
            model_iter_fn(model, example_inputs)
        peak_memory = get_peak_memory()
    except Exception:
        compilation_latency = 0
        peak_memory = 0
    return compilation_latency, peak_memory


def null_experiment(args, model_iter_fn, model, example_inputs):
    """
    A no-op experiment useful for making sure TorchBenchark alone works properly.
    """

    return []


def cast_to_fp16(model, inputs):
    # cast model and inputs to fp16
    model = model.half()

    inputs = tree_map(
        lambda x: x.to(torch.float16)
        if getattr(x, "dtype", None) == torch.float32
        or getattr(x, "dtype", None) == torch.float64
        else x,
        inputs,
    )

    # Disable this part temporarily. Further evaluation needed
    # TRT does not support int64. Some model does need it like Super_SloMo
    # if current_name != "Super_SloMo" and current_name != "fastNLP_Bert":
    #     inputs = tuple(
    #         tree_map(
    #             lambda x: x.to(torch.int32)
    #             if getattr(x, "dtype", None) == torch.int64
    #             else x,
    #             inputs,
    #         )
    #     )
    return model, inputs


def cast_to_fp32(model, inputs):
    # cast model and inputs to fp16
    model = model.to(torch.float32)

    inputs = tree_map(
        lambda x: x.to(torch.float32)
        if getattr(x, "dtype", None) == torch.float16
        or getattr(x, "dtype", None) == torch.float64
        else x,
        inputs,
    )

    return model, inputs


class DummyGradScaler:
    def scale(self, loss):
        return loss


class BenchmarkRunner:
    def __init__(self):
        self.use_amp = False
        self.grad_scaler = DummyGradScaler()
        self.autocast = NullContext
        self._args = None

    def setup_amp(self):
        if self.args.amp and self.args.training:
            assert self.args.devices == ["cuda"], "AMP is supported only for CUDA"
            self.grad_scaler = torch.cuda.amp.GradScaler()
            self.autocast = torch.cuda.amp.autocast
            # TODO - Debug whats going wrong with the numerics
            self.args.cosine = True

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        self._args = args

    @property
    def skip_models(self):
        return set()

    @property
    def slow_models(self):
        return set()

    @property
    def very_slow_models(self):
        return set()

    @property
    def non_deterministic_models(self):
        return set()

    @property
    def skip_not_suitable_for_training_models(self):
        return set()

    @property
    def failing_python_key_models(self):
        return set()

    @property
    def failing_torchinductor_models(self):
        return set()

    @property
    def failing_fx2trt_models(self):
        return set()

    @property
    def failing_dynamic_shape_models(self):
        return set()

    @property
    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        raise NotImplementedError()

    def resolve_precision(self):
        use_amp = False
        model_dtype = torch.float32
        data_dtype = torch.float32
        if self._args.amp:
            use_amp = True
        elif self._args.float16:
            model_dtype = torch.float16
            data_dtype = torch.float16
        # elif self._args.bfloat16:
        #     model_dtype = torch.bfloat16
        #     data_dtype = torch.bfloat16
        return use_amp, model_dtype, data_dtype

    def decay_batch_exp(self, batch_size, factor=0.5, divisor=2):
        out_batch_size = batch_size * factor
        if out_batch_size > divisor:
            out_batch_size = (out_batch_size + 1) // divisor * divisor
        else:
            out_batch_size = batch_size - 1
        return max(0, int(out_batch_size))

    def profile_compilation(self, device, model_name, model_iter_fn, backend):
        """
        Profiles compilation characteristics, e.g., compilation latency and memory.
        """

        try:
            batch_size = None
            if self.args.batch_size_file:
                batch_size = read_batch_size_from_file(
                    self.args, self.args.batch_size_file, model_name
                )
            elif self.args.batch_size:
                batch_size = self.args.batch_size
            device, name, model, example_inputs, batch_size = self.load_model(
                device,
                model_name,
                self._args.training,
                self._args.use_eval_mode,
                batch_size,
            )
        except NotImplementedError:
            logging.warn(f"{model_name} failed to load")

        assert (
            device == "cuda"
        ), "The memory measurement is currently specific to CUDA devices"
        experiment = functools.partial(
            compilation_profiling_experiment,
            model_iter_fn=model_iter_fn,
            model=model,
            example_inputs=example_inputs,
        )
        time, memory = experiment(backend=backend)

        output_csv(
            output_filename,
            ("dev", "name", "batch_size", "time", "memory"),
            [device, model_name, batch_size, time, memory],
        )

    def batch_size_finder(
        self, device, model_name, model_iter_fn, initial_batch_size=128
    ):
        batch_size = initial_batch_size
        while batch_size >= 1:
            torch.cuda.empty_cache()
            try:
                device, name, model, example_inputs, _ = self.load_model(
                    device,
                    model_name,
                    self._args.training,
                    self._args.use_eval_mode,
                    batch_size,
                )
                model_iter_fn(model, example_inputs)
                return batch_size
            except RuntimeError as e:
                error_str = str(e)
                if "channels_last" in error_str:
                    break
            batch_size = self.decay_batch_exp(batch_size)
        return 1

    def run_one_model(
        self,
        name,
        model,
        is_training,
        model_iter_fn,
        example_inputs,
        optimize_ctx,
        accuracy_ctx,
        experiment,
        skip_accuracy_check=False,
        dynamic_shapes=False,
    ):
        tolerance, cos_similarity = self.get_tolerance_and_cosine_flag(
            is_training, current_device, name
        )
        experiment_kwargs = dict()
        with self.pick_grad(name, is_training):
            mode = "train" if is_training else "eval"
            sys.stdout.write(f"{current_device:4} {mode:5} {current_name:34} ")
            sys.stdout.flush()
            for submod in itertools.chain([model], model.modules()):
                assert not torchdynamo.utils.is_jit_model(submod)

            if dynamic_shapes:
                # skip correctness check for ds benchmark, becuase example_inputs are not
                # compatible with the code below, and the same benchmarks can be run in
                # non-dynamic shapes mode for correctness checks
                torch.manual_seed(1337)
                torchdynamo.reset()
                results = []
                results.append(experiment(model, example_inputs))
                print(" ".join(map(str, results)))
                return 0

            torch.manual_seed(1337)
            correct_result = model_iter_fn(
                copy.deepcopy(model), torchdynamo.utils.clone_inputs(example_inputs)
            )

            torch.manual_seed(1337)
            if current_name not in self.non_deterministic_models:
                correct_rerun_result = model_iter_fn(
                    copy.deepcopy(model), torchdynamo.utils.clone_inputs(example_inputs)
                )
                if not same(correct_result, correct_rerun_result):
                    print("INCORRECT - Variation in Eager runs itself")
                    if not skip_accuracy_check:
                        return sys.exit(-1)

            t0 = time.perf_counter()
            torch.manual_seed(1337)
            torchdynamo.reset()
            if experiment.func is cold_start_experiment:
                results = []
                results.append(experiment(model, example_inputs, optimize_ctx))
                print(" ".join(map(str, results)))
                return 0

            try:
                accuracy_model_iter_fn = accuracy_ctx(model_iter_fn)
                new_result = accuracy_model_iter_fn(model, example_inputs)
            except Exception:
                logging.exception("unhandled error")
                print("ERROR")
                return sys.exit(-1)
            if current_name in self.non_deterministic_models:
                # This model has non-deterministic output so we cant
                # check correctness.
                # TODO(jansel): submit upstream fix for this
                pass
            elif not same(correct_result, new_result, cos_similarity, tolerance):
                print("INCORRECT")
                if not skip_accuracy_check:
                    return sys.exit(-1)
            ok, total = Stats.reset_counters()
            results = []
            if optimize_ctx != accuracy_ctx:
                torchdynamo.reset()
            # run with torchdynamo few times to populate the cache
            optimized_model_iter_fn = optimize_ctx(model_iter_fn)
            for _ in range(3):
                optimized_model_iter_fn(model, example_inputs)
            _, frames_second_pass = Stats.reset_counters()  # should be 0

            if frames_second_pass > 0:
                optimized_model_iter_fn(model, example_inputs)
                _, frames_third_pass = Stats.reset_counters()  # should be 0
            else:
                frames_third_pass = 0

            if output_filename and "coverage" in output_filename:
                t1 = time.perf_counter()
                results.append(
                    f"{ok:3}/{total:3} +{frames_third_pass} frames {t1-t0:3.0f}s"
                )

            if experiment.func is coverage_experiment:
                experiment_kwargs["start_latency"] = t1 - t0

            if not hasattr(model, name):
                model.name = name
            results.append(experiment(model, example_inputs, **experiment_kwargs))
            print(" ".join(map(str, results)))


def help(fn):
    return fn.__doc__


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude", "-x", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument("--devices", "-d", action="append", help="cpu or cuda")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    parser.add_argument(
        "--randomize-input",
        action="store_true",
        help="Whether to randomize the input values. Dimensions will be kept the same.",
    )
    parser.add_argument(
        "--threads", "-t", type=int, help="number of threads to use for eager"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="enable verbose debug printouts"
    )
    parser.add_argument(
        "--nopython", action="store_true", help="Turn graph breaks into errors"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="run models that are in the global SKIP list",
    )
    parser.add_argument(
        "--nvfuser", action="store_true", help="enable nvfuser globally"
    )
    parser.add_argument(
        "--prims-nvfuser", action="store_true", help="user prims + nvfuser backend"
    )
    parser.add_argument(
        "--isolate", action="store_true", help="run each model in its own process"
    )
    parser.add_argument(
        "--dump-raw-metrics",
        action="store_true",
        help="dump raw timing metrics from speedup experiment",
    )

    parser.add_argument(
        "--channels-last",
        action="store_true",
        default=False,
        help="use channels last format",
    )
    parser.add_argument("--batch_size", type=int, help="batch size for benchmarking")
    parser.add_argument(
        "--batch_size_file", type=str, help="String to load batch size from"
    )
    parser.add_argument("--cosine", action="store_true", help="use cosine similarity")
    parser.add_argument(
        "--fast", "-f", action="store_true", help="skip slow benchmarks"
    )
    parser.add_argument("--only", help="used by --isolate to run just one model")
    parser.add_argument(
        "--minimum-call-count", type=int, help="filter out graphs with too few ops"
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Performs training",
    )
    parser.add_argument(
        "--dynamic_shapes",
        action="store_true",
        help="Runs a dynamic shapes version of the benchmark, if available.",
    )
    parser.add_argument(
        "--use-eval-mode",
        action="store_true",
        help="sets model.eval() to reduce randomness",
    )
    parser.add_argument(
        "--skip-accuracy-check",
        action="store_true",
        help="keeps running even when accuracy fails",
    )
    parser.add_argument(
        "--generate-aot-autograd-stats",
        action="store_true",
        help="Generates AOT Autograd stats like how mnay graphs are sent to AOT",
    )
    parser.add_argument(
        "--inductor-settings",
        action="store_true",
        help="Use same settings as --inductor for baseline comparisons",
    )
    parser.add_argument(
        "--raise-on-assertion-error",
        action="store_true",
        help="Fail a benchmark if torchdynamo triggers an internal assertion",
    )
    parser.add_argument(
        "--raise-on-backend-error",
        action="store_true",
        help="Fail a benchmark if backend throws an exception",
    )
    parser.add_argument(
        "--output",
        help="Overides the output filename",
    )
    parser.add_argument(
        "--export-profiler-trace",
        action="store_true",
        help="exports trace of kineto profiler",
    )
    parser.add_argument("--profiler_trace_name", help="Overwrites exported trace name")

    group_prec = parser.add_mutually_exclusive_group()
    group_prec.add_argument("--float16", action="store_true", help="cast model to fp16")
    group_prec.add_argument("--float32", action="store_true", help="cast model to fp32")
    group_prec.add_argument(
        "--amp", action="store_true", help="use automatic mixed precision"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--coverage", action="store_true", help="(default) " + help(coverage_experiment)
    )
    group.add_argument(
        "--online-autotune", action="store_true", help=help(speedup_experiment)
    )
    group.add_argument(
        "--offline-autotune", action="store_true", help=help(speedup_experiment)
    )
    group.add_argument(
        "--speedup-fixed1",
        action="store_true",
        help="speedup using experimental fixed_strategy backend",
    )
    group.add_argument(
        "--speedup-fixed2",
        action="store_true",
        help="speedup using experimental fixed_strategy backend",
    )
    group.add_argument(
        "--speedup-ltc",
        action="store_true",
        help="speedup using the ltc backend",
    )
    group.add_argument(
        "--speedup-ltc-trivial",
        action="store_true",
        help="speedup using the ltc backend without reusing compiled graph",
    )
    group.add_argument(
        "--cold-start", action="store_true", help=help(cold_start_experiment)
    )
    group.add_argument(
        "--overhead", action="store_true", help=help(overhead_experiment)
    )
    group.add_argument(
        "--speedup-ts", action="store_true", help=help(speedup_experiment_ts)
    )
    group.add_argument(
        "--speedup-sr", action="store_true", help=help(speedup_experiment_sr)
    )
    group.add_argument(
        "--speedup-onnx", action="store_true", help=help(speedup_experiment_onnx)
    )
    group.add_argument(
        "--speedup-trt", action="store_true", help=help(speedup_experiment_trt)
    )
    group.add_argument(
        "--speedup-dynamo-ts",
        action="store_true",
        help="TorchDynamo frontend with torchscript backend",
    )
    group.add_argument("--python-key", action="store_true")
    group.add_argument(
        "--speedup-fx2trt", action="store_true", help=help(speedup_experiment_fx2trt)
    )
    group.add_argument(
        "--speedup-fx2trt-fp16",
        action="store_true",
        help=help(speedup_experiment_fx2trt),
    )
    group.add_argument(
        "--accuracy-aot-nop",
        action="store_true",
        help="Accuracy testing and speedup for AOT vs Eager",
    )
    group.add_argument(
        "--accuracy-aot-ts",
        action="store_true",
        help="Accuracy testing and speedup for AOT with Torchscript(NNC/NVFuser) vs Eager",
    )
    group.add_argument(
        "--accuracy-aot-ts-mincut",
        action="store_true",
        help="Accuracy testing and speedup for AOT with Torchscript(NNC/NVFuser) with mincut vs Eager",
    )
    group.add_argument(
        "--print-fx",
        action="store_true",
        help="Print fx traces captured from model",
    )
    group.add_argument(
        "--print-aten-ops",
        action="store_true",
        help="Print traces of aten ops captured by AOT autograd",
    )
    group.add_argument(
        "--accuracy-ts",
        action="store_true",
        help="Accuracy testing and speedup using Torchscript (NNC/NVFuser) vs eager",
    )
    group.add_argument(
        "--inductor",
        action="store_true",
        help="Measure speedup with TorchInductor",
    )
    group.add_argument(
        "--inductor-dynamic",
        action="store_true",
        help="Measure speedup with TorchInductor",
    )
    group.add_argument(
        "--backend",
        choices=torchdynamo.list_backends(),
        help="measure speedup with a given backend",
    )
    group.add_argument("--nothing", action="store_true", help=help(null_experiment))
    group.add_argument(
        "--nops",
        action="store_true",
        help="Test that bytecode rewriting works properly.",
    )
    group.add_argument(
        "--log-conv-args",
        action="store_true",
        help="Dump convolution input/weight/bias's shape/stride/dtype and other options to json",
    )
    group.add_argument(
        "--recompile_profiler",
        action="store_true",
        help="Run the dynamo recompilation profiler on each model.",
    )
    group.add_argument(
        "--find-batch-sizes",
        action="store_true",
        help="finds the largest batch size that could fit on GPUs",
    )
    group.add_argument(
        "--profile-backend",
        type=str,
        help="reports the peak memory and compilation latency for a backend",
    )
    args = parser.parse_args()
    return args


def main(runner, original_dir=None):
    args = parse_args()

    # Pass the parsed args object to benchmark runner object
    runner.args = args

    # defaults
    args.devices = args.devices or ["cpu"]
    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]

    if args.devices != ["cpu"] and torch.cuda.is_available():
        global synchronize
        synchronize = torch.cuda.synchronize

    if (
        args.devices == ["cuda"]
        and torch.cuda.get_device_properties(0).total_memory < 25 * 2**30
    ):
        # OOM errors on an RTX 3090 with 24gb RAM
        runner.skip_models.update(
            {
                # torchbench
                "hf_Longformer",
                "timm_nfnet",
                "timm_efficientdet",
                # timm
                "beit_base_patch16_224",
                "cait_m36_384",
                "convmixer_768_32",
                "deit_base_distilled_patch16_224",
                "dm_nfnet_f0",
                "dpn107",
                "dm_nfnet_f0",
            }
        )
        if args.training:
            runner.skip_models.add("hf_T5")

    if torchdynamo.config.dynamic_shapes:
        # TODO(jansel): fix bugs in these
        runner.skip_models.update(runner.failing_dynamic_shape_models)

    if args.nvfuser:
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
    else:
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        if torch.cuda.is_available():
            torch._C._jit_set_nvfuser_enabled(False)

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.verbose:
        torchdynamo.config.debug = True

    torchdynamo.config.raise_on_assertion_error = args.raise_on_assertion_error
    torchdynamo.config.raise_on_backend_error = args.raise_on_backend_error

    if args.training:
        model_iter_fn = runner.forward_and_backward_pass
        runner.skip_models.update(runner.skip_not_suitable_for_training_models)
    else:
        model_iter_fn = runner.forward_pass

    if args.fast:
        runner.skip_models.update(runner.slow_models)

    if args.devices == ["cpu"]:
        runner.skip_models.update(runner.very_slow_models)

    if args.inductor or args.inductor_dynamic or args.inductor_settings:
        runner.skip_models.update(runner.failing_torchinductor_models)
        args.isolate = True
        args.cosine = True
        if args.float16:
            # TODO(jansel): check if correctness issue is real
            runner.skip_models.add("yolov3")
        if args.training:
            # dropout,etc makes results not match
            args.skip_accuracy_check = True

    if args.float16:
        # these give `INCORRECT - Variation in Eager runs itself` sometimes
        runner.non_deterministic_models.update(
            {
                "demucs",
                "pyhpc_equation_of_state",
                "timm_efficientdet",
                "pyhpc_isoneutral_mixing",
                "pyhpc_turbulent_kinetic_energy",
                "shufflenet_v2_x1_0",
            }
        )

    if args.no_skip:
        runner.skip_models.clear()

    accuracy_ctx = None
    experiment = null_experiment
    global current_name, current_device, current_batch_size, output_filename, optimize_ctx
    optimize_ctx = NullContext()

    if args.overhead:
        optimize_ctx = torchdynamo.optimize(dummy_fx_compile, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "overheads.csv"
    elif args.cold_start:
        optimize_ctx = torchdynamo.optimize("aot_nvfuser", nopython=args.nopython)
        experiment = cold_start_experiment
        assert args.nvfuser, "TODO - Add another aot string for mem fusion with NNC"
        backend_str = "nvfuser" if args.nvfuser else "nnc"
        output_filename = f"cold_start_{backend_str}.csv"
        args.isolate = True
        # TODO(whc) should we move this to a more general part of the script?
        torch.backends.cuda.matmul.allow_tf32 = True
    elif args.inductor or args.inductor_dynamic:
        import torchinductor.config

        torchinductor.config.debug = args.verbose
        if args.threads:
            torchinductor.config.cpp.threads = args.threads

        if args.inductor_dynamic:
            torchinductor.config.triton.cudagraphs = False
            torchinductor.config.dynamic_shapes = True
        else:
            torchinductor.config.dynamic_shapes = False
            if args.export_profiler_trace:
                print("Profiling requested, setting cudagraphs to False")
                torchinductor.config.triton.cudagraphs = False

        optimize_ctx = torchdynamo.optimize("inductor", nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "inductor.csv"
    elif args.online_autotune:
        optimize_ctx = torchdynamo.optimize(online_autotuner, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "speedups.csv"
        args.isolate = True
    elif args.offline_autotune:
        optimize_ctx = torchdynamo.optimize(offline_autotuner, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "speedups.csv"
        args.isolate = True
    elif args.python_key:
        optimize_ctx = torchdynamo.optimize(python_key, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "pythonkey.csv"
        if not args.no_skip:
            runner.skip_models.update(runner.failing_python_key_models)
    elif args.speedup_ltc:
        optimize_ctx = torchdynamo.optimize(
            backends.ltc_reuse_graph, nopython=args.nopython
        )
        experiment = speedup_experiment
        output_filename = "speedups_ltc.csv"
        args.isolate = True
    elif args.speedup_ltc_trivial:
        optimize_ctx = torchdynamo.optimize(
            backends.ltc_trivial, nopython=args.nopython
        )
        experiment = speedup_experiment
        output_filename = "speedups_ltc_trivial.csv"
        args.isolate = True
    elif args.speedup_fixed1:
        optimize_ctx = torchdynamo.optimize(fixed_strategy1, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "speedups_fixed1.csv"
        args.isolate = True
    elif args.speedup_fixed2:
        optimize_ctx = torchdynamo.optimize(fixed_strategy2, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "speedups_fixed2.csv"
        args.isolate = True
    elif args.speedup_ts:
        experiment = speedup_experiment_ts
        output_filename = "baseline_ts.csv"
    elif args.speedup_sr:
        experiment = speedup_experiment_sr
        output_filename = "baseline_sr.csv"
    elif args.speedup_onnx:
        experiment = speedup_experiment_onnx
        output_filename = "baseline_onnx.csv"
    elif args.speedup_trt:
        experiment = speedup_experiment_trt
        output_filename = "baseline_trt.csv"
    elif args.speedup_dynamo_ts:
        optimize_ctx = torchdynamo.optimize(backends.ts, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "speedup_dynamo_ts.csv"
    elif args.speedup_fx2trt:
        optimize_ctx = torchdynamo.optimize(
            backends.fx2trt_compiler, nopython=args.nopython
        )
        experiment = speedup_experiment_fx2trt
        output_filename = "speedups_fx2trt.csv"
        runner.skip_models.update(runner.failing_fx2trt_models)
        args.float32 = True
        args.float16 = False
        args.cosine = True
    elif args.speedup_fx2trt_fp16:
        optimize_ctx = torchdynamo.optimize(
            backends.fx2trt_compiler_fp16, nopython=args.nopython
        )
        experiment = speedup_experiment_fx2trt
        output_filename = "speedups_fx2trt_fp16.csv"
        args.float32 = False
        args.float16 = True
        args.cosine = True
    elif args.accuracy_aot_nop:
        optimize_ctx = torchdynamo.optimize("aot_nop", nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "accuracy_aot_nop.csv"
    elif args.accuracy_aot_ts:
        optimize_ctx = torchdynamo.optimize("aot_ts", nopython=args.nopython)
        experiment = speedup_experiment
        backend_str = "nvfuser" if args.nvfuser else "nnc"
        output_filename = f"accuracy_aot_{backend_str}.csv"
    elif args.accuracy_aot_ts_mincut:
        optimize_ctx = torchdynamo.optimize("aot_nvfuser", nopython=args.nopython)
        accuracy_ctx = torchdynamo.optimize(
            "aot_nvfuser_nodecomps", nopython=args.nopython
        )
        experiment = speedup_experiment
        assert args.nvfuser, "TODO - Add another aot string for mem fusion with NNC"
        backend_str = "nvfuser" if args.nvfuser else "nnc"
        output_filename = f"accuracy_aot_{backend_str}_mincut.csv"
    elif args.prims_nvfuser:
        optimize_ctx = torchdynamo.optimize("prims_nvfuser", nopython=args.nopython)
        experiment = speedup_experiment
        backend_str = "prims_nvfuser"
        output_filename = f"accuracy_aot_{backend_str}.csv"
    elif args.print_fx:
        optimize_ctx = torchdynamo.optimize(
            print_fx,
            nopython=args.nopython,
        )
    elif args.print_aten_ops:
        optimize_ctx = torchdynamo.optimize(
            print_aten_ops,
            nopython=args.nopython,
        )
    elif args.accuracy_ts:
        optimize_ctx = torchdynamo.optimize(fixed_strategy1, nopython=args.nopython)
        experiment = speedup_experiment
        backend_str = "nvfuser" if args.nvfuser else "nnc"
        output_filename = f"accuracy_{backend_str}.csv"
    elif args.nothing:
        pass
    elif args.nops:
        optimize_ctx = torchdynamo.eval_frame._optimize_catch_errors(
            torchdynamo.testing.debug_insert_nops, nopython=args.nopython
        )
    elif args.backend:
        optimize_ctx = torchdynamo.optimize(args.backend, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = f"speedup_{args.backend}.csv"
        args.isolate = True
    elif args.log_conv_args:
        optimize_ctx = torchdynamo.optimize(conv_args_analysis, nopython=args.nopython)
        output_filename = "log_conv_args.csv"
    elif args.recompile_profiler:
        output_filename = "recompile_profiler_log.csv"
        experiment = recompile_profiler_experiment
    else:
        optimize_ctx = torchdynamo.optimize(fx_insert_profiling, nopython=args.nopython)
        experiment = coverage_experiment
        output_filename = "coverage.csv"

    if accuracy_ctx is None:
        accuracy_ctx = optimize_ctx

    runner.setup_amp()

    if args.output:
        output_filename = args.output

    if output_filename:
        output_filename = os.path.join(torchdynamo.config.base_dir, output_filename)

    if args.minimum_call_count:
        torchdynamo.config.minimum_call_count = args.minimum_call_count

    if args.find_batch_sizes:
        args.isolate = True

    if args.find_batch_sizes and args.only:
        assert args.isolate
        for device in args.devices:
            batch_size = runner.batch_size_finder(device, args.only, model_iter_fn)
            print(args.only, batch_size)
            output_csv(output_filename, [], [args.only, batch_size])
        return

    if args.profile_backend and args.only:
        assert args.isolate, "Use --isolate or --only to enable isolation"
        if output_filename is None:
            output_filename = "backends_profile.csv"
        for device in args.devices:
            runner.profile_compilation(
                device, args.only, model_iter_fn, args.profile_backend
            )
        return

    if args.export_profiler_trace:
        if args.profiler_trace_name is None:
            if args.backend:
                args.profiler_trace_name = args.backend
            elif args.inductor or args.inductor_dynamic:
                args.profiler_trace_name = "inductor"
            else:
                args.profiler_trace_name = "profile"
        else:
            args.profiler_trace_name = args.profiler_trace_name

    if args.batch_size_file:
        if not (args.only or args.isolate):
            raise RuntimeError("--batch-size-file requires --only or --isolate")

    experiment = functools.partial(experiment, args, model_iter_fn)

    if args.only:
        model_name = args.only
        for device in args.devices:
            batch_size = args.batch_size
            if args.batch_size_file:
                batch_size = read_batch_size_from_file(
                    args, args.batch_size_file, model_name
                )
            try:
                device, name, model, example_inputs, batch_size = runner.load_model(
                    device,
                    model_name,
                    args.training,
                    args.use_eval_mode,
                    batch_size,
                    args.dynamic_shapes,
                )
            except NotImplementedError:
                logging.warn(f"{args.only} failed to load")
                continue  # bad benchmark implementation

            current_name = name
            current_device = device
            current_batch_size = batch_size
            set_model_name(name)

            if args.float32:
                model, example_inputs = cast_to_fp32(model, example_inputs)
            elif args.float16:
                model, example_inputs = cast_to_fp16(model, example_inputs)

            runner.run_one_model(
                name,
                model,
                args.training,
                model_iter_fn,
                example_inputs,
                optimize_ctx,
                accuracy_ctx,
                experiment,
                args.skip_accuracy_check,
                args.dynamic_shapes,
            )
        if args.generate_aot_autograd_stats:
            stats_file = output_filename.split(".csv")[0] + "_stats.csv"
            output_csv(
                stats_file,
                ("dev", "name", "batch_size", "total_aot_graphs", "ok_aot_graphs"),
                [
                    current_device,
                    current_name,
                    current_batch_size,
                    *Stats.aot_summary(),
                ],
            )
    elif args.isolate:
        if output_filename and os.path.exists(output_filename):
            os.unlink(output_filename)
        if original_dir:
            os.chdir(original_dir)
        for name in runner.iter_model_names(args):
            current_name = name
            placeholder_batch_size = 0
            try:
                subprocess.check_call([sys.executable] + sys.argv + [f"--only={name}"])
            except subprocess.SubprocessError:
                print("ERROR")
                for device in args.devices:
                    output_csv(
                        output_filename, [], [device, name, placeholder_batch_size, 0.0]
                    )
        print_summary(output_filename)
    else:
        if output_filename and os.path.exists(output_filename):
            os.unlink(output_filename)
        for device, name, model, example_inputs, batch_size in runner.iter_models(args):
            current_name = name
            current_device = device
            current_batch_size = batch_size
            torchdynamo.reset()
            gc.collect()

            if args.float32:
                model, example_inputs = cast_to_fp32(model, example_inputs)
            elif args.float16:
                model, example_inputs = cast_to_fp16(model, example_inputs)
            runner.run_one_model(
                name,
                model,
                args.training,
                model_iter_fn,
                example_inputs,
                optimize_ctx,
                accuracy_ctx,
                experiment,
                args.skip_accuracy_check,
                args.dynamic_shapes,
            )

        Stats.print_summary()
        print_summary(output_filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
