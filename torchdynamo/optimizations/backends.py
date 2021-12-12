import functools
import io
import logging
import os
import subprocess
import tempfile

import torch

log = logging.getLogger(__name__)


def catch_errors(fn):
    @functools.wraps(fn)
    def inner(model, *args, **kwargs):
        if model is None:
            return None
        try:
            return fn(model, *args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception(f"{fn.__name__} error")
            return None

    return inner


def clone_inputs(example_inputs):
    res = list(example_inputs)
    for i in range(len(res)):
        if isinstance(res[i], torch.Tensor):
            res[i] = res[i].clone().detach()
    return res


def is_jit_model(model0):
    return isinstance(
        model0,
        (
            torch.jit._trace.TopLevelTracedModule,
            torch.jit._script.RecursiveScriptModule,
            torch.jit.ScriptFunction,
        ),
    )


def torchscript(model0, example_inputs, verbose=True):
    if is_jit_model(model0):
        return model0
    try:
        model1 = torch.jit.trace(model0, example_inputs)
    except Exception:
        if verbose:
            log.exception("jit trace error")
        try:
            model1 = torch.jit.script(model0)
        except Exception:
            model1 = None
    return model1


@catch_errors
def optimize_for_inference(scripted, example_inputs):
    scripted = torchscript(scripted, example_inputs)
    res = torch.jit.optimize_for_inference(scripted)
    res(*example_inputs)  # shake out any errors
    return res


@catch_errors
def static_runtime(scripted, example_inputs):
    scripted = torchscript(scripted, example_inputs)
    if hasattr(scripted, "_c"):
        static_module = torch._C._jit_to_static_module(scripted._c)
    else:
        static_module = torch._C._jit_to_static_module(scripted.graph)

    def _call(*args):
        res = static_module(args, {})
        # inference mode tensors can cause issues
        # res = [torch.from_numpy(x.numpy()) for x in res]
        res = [x.clone() for x in res]
        return res

    _call(*example_inputs)  # shake out any errors
    return _call


@catch_errors
def onnxrt(scripted, example_inputs, filename=None):
    scripted = torchscript(scripted, example_inputs)
    with tempfile.NamedTemporaryFile() as tmp:
        if filename is None:
            filename = tmp.name
        try:
            torch.onnx.export(
                scripted,
                example_inputs,
                filename,
                input_names=[f"i{i}" for i in range(len(example_inputs))],
                do_constant_folding=True,
                opset_version=14,
            )
        except IndexError:
            # work around bug in constant folding pass
            torch.onnx.export(
                scripted,
                example_inputs,
                filename,
                input_names=[f"i{i}" for i in range(len(example_inputs))],
                do_constant_folding=False,
                opset_version=14,
            )

        return onnxrt_wrapper(filename, example_inputs)


def onnxrt_wrapper(filename, example_inputs):
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(filename)

    def to_numpy(x):
        try:
            return x.numpy()
        except RuntimeError:
            return x.detach().numpy()

    def _call(*args):
        res = ort_session.run(
            None, {f"i{i}": to_numpy(arg) for i, arg in enumerate(args)}
        )
        res = [torch.from_numpy(x) for x in res]
        return res

    # shake out any errors
    _call(*example_inputs)

    return _call


@catch_errors
def taso(example_inputs, onnx_filename, taso_filename):
    subprocess.check_call(
        [
            os.path.expanduser("~/conda/envs/taso/bin/python"),
            "-c",
            "import taso,onnx; onnx.save(taso.export_onnx(taso.optimize("
            f"taso.load_onnx('{onnx_filename}'))), '{taso_filename}')",
        ]
    )
    return onnxrt_wrapper(taso_filename, example_inputs)


@catch_errors
def ipex(scripted, example_inputs):
    scripted = torchscript(scripted, example_inputs)
    import intel_extension_for_pytorch

    return intel_extension_for_pytorch.optimize(scripted)


@catch_errors
def fx2trt(model, inputs):
    from torch.fx.experimental.fx2trt.fx2trt import InputTensorSpec
    from torch.fx.experimental.fx2trt.fx2trt import TRTInterpreter
    from torch.fx.experimental.fx2trt.fx2trt import TRTModule
    import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer

    logging.getLogger("torch.fx.experimental.fx_acc.acc_tracer").setLevel(logging.ERROR)

    model = acc_tracer.trace(model, inputs)
    input_specs = InputTensorSpec.from_tensors(inputs)
    interp = TRTInterpreter(model, input_specs, explicit_precision=True)
    result = interp.run(fp16_mode=False, max_batch_size=len(inputs[0]))
    trt_mod = TRTModule(result.engine, result.input_names, result.output_names)
    outputs = model(*inputs)
    if isinstance(outputs, (tuple, list)) and len(outputs) == 1:
        return lambda *args: (trt_mod(*args),)
    return trt_mod


@catch_errors
def torch2trt(model, inputs):
    from torch2trt import torch2trt

    trt_mod = torch2trt(
        model, inputs, max_batch_size=len(inputs[0]), strict_type_constraints=True
    )
    outputs = model(*inputs)
    if isinstance(outputs, (tuple, list)) and len(outputs) == 1:
        return lambda *args: (trt_mod(*args),)
    return trt_mod


@catch_errors
def onnx2trt(model, inputs):
    import tensorrt as trt
    from torch.fx.experimental.fx2trt.fx2trt import TRTModule

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    assert isinstance(inputs, (list, tuple))
    inputs = tuple(inputs)
    outputs = model(*inputs)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    input_names = [f"i{x}" for x in range(len(inputs))]
    output_names = [f"o{x}" for x in range(len(outputs))]
    f = io.BytesIO()
    torch.onnx.export(
        torchscript(model, inputs),
        inputs,
        f,
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
    )
    f.seek(0)
    onnx_bytes = f.read()
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    success = parser.parse(onnx_bytes)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    assert success

    config.max_workspace_size = 1 << 25
    builder.max_batch_size = len(inputs[0])

    config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    engine = builder.build_engine(network, config)
    assert engine

    trt_mod = TRTModule(engine, input_names, output_names)
    outputs = model(*inputs)
    if isinstance(outputs, (tuple, list)) and len(outputs) == 1:
        return lambda *args: (trt_mod(*args),)
    return trt_mod


@catch_errors
def cudagraphs(model, inputs):
    assert isinstance(inputs, (list, tuple))
    static_inputs = [torch.randn_like(x) for x in inputs]

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)

    def run(*new_inputs):
        assert len(static_inputs) == len(new_inputs)
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)
        graph.replay()
        return [x.clone() for x in static_outputs]

    if isinstance(static_outputs, torch.Tensor):
        static_outputs = (static_outputs,)
        return lambda *args: run(*args)[0]
    else:
        return run


def tvm_compile(jit_mod, example_inputs, log_file=None, **kwargs):
    if jit_mod is None:
        return None
    try:
        return tvm_compile_inner(jit_mod, example_inputs, log_file, **kwargs)
    except Exception as e:
        if log_file and os.path.exists(log_file):
            os.unlink(log_file)
        if isinstance(e, KeyboardInterrupt):
            raise
        log.exception("tvm error")
        return None


@functools.lru_cache(1)
def llvm_target():
    if "avx512" in open("/proc/cpuinfo").read():
        return "llvm -mcpu=skylake-avx512"
    return "llvm -mcpu=core-avx2"


def tvm_compile_inner(jit_mod, example_inputs, log_file, trials=20000):
    # based on functorch version in eager_compile.py
    import tvm
    from tvm import relay, auto_scheduler
    from tvm.contrib import graph_executor

    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
    dev = tvm.cpu(0)
    target = tvm.target.Target(llvm_target())
    if log_file is not None:
        if not os.path.exists(log_file):
            tasks, task_weights = auto_scheduler.extract_tasks(
                mod["main"], params, target
            )
            for task in tasks:
                print(task.compute_dag)
            else:
                print("No tasks")
            if len(tasks) != 0:
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                if not os.path.exists(log_file):
                    assert trials > 0
                    tune_option = auto_scheduler.TuningOptions(
                        num_measure_trials=max(trials, 64 * len(tasks)),
                        # num_measure_trials=10000,  # change this to 20000 to achieve the best performance
                        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                        early_stopping=1000,
                        # verbose=2,
                    )
                    try:
                        tuner.tune(tune_option)
                    except Exception:
                        if os.path.exists(log_file):
                            os.unlink(log_file)
                        raise

        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)
    else:
        # no autotuning (for debugging)
        with tvm.transform.PassContext(opt_level=10):
            lib = relay.build(mod, target=target, params=params)

    m = graph_executor.GraphModule(lib["default"](dev))

    def exec_tvm(*args):
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:
                m.set_input(
                    f"inp_{idx}",
                    tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg.contiguous())),
                )
        m.run()
        outs = [
            torch.utils.dlpack.from_dlpack(m.get_output(i).to_dlpack())
            for i in range(m.get_num_outputs())
        ]
        return outs

    # shake out any errors
    exec_tvm(*example_inputs)

    return exec_tvm
