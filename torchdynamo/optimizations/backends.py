import functools
import logging
import os
import subprocess
import tempfile

import torch

log = logging.getLogger(__name__)


def clone_inputs(example_inputs):
    res = list(example_inputs)
    for i in range(len(res)):
        if isinstance(res[i], torch.Tensor):
            res[i] = res[i].clone().detach()
    return res


def torchscript(model0, example_inputs, filename=None):
    try:
        model1 = torch.jit.trace(model0, example_inputs)
    except Exception:
        log.exception("jit trace error")
        try:
            model1 = torch.jit.script(model0)
        except Exception:
            model1 = None
    return model1


def optimize_for_inference(scripted, example_inputs, filename=None):
    if scripted is None:
        return None
    try:
        res = torch.jit.optimize_for_inference(scripted)

        # shake out any errors
        res(*example_inputs)

        return res
    except KeyboardInterrupt:
        raise
    except Exception:
        log.exception("optimize_for_inference error")
        return None


def static_runtime(scripted, example_inputs, filename=None):
    if scripted is None:
        return None
    try:
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

        # shake out any errors
        _call(*example_inputs)

        return _call
    except Exception:
        log.exception("Error from static_runtime")
        return None


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


def onnxrt(scripted, example_inputs, filename=None):
    if scripted is None:
        return None
    try:
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
    except KeyboardInterrupt:
        raise
    except Exception:
        log.exception("ONNX error")
        return None


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


def taso(example_inputs, onnx_filename, taso_filename):
    if not os.path.exists(onnx_filename):
        return None
    try:
        subprocess.check_call(
            [
                os.path.expanduser("~/conda/envs/taso/bin/python"),
                "-c",
                "import taso,onnx; onnx.save(taso.export_onnx(taso.optimize("
                f"taso.load_onnx('{onnx_filename}'))), '{taso_filename}')",
            ]
        )
        return onnxrt_wrapper(taso_filename, example_inputs)
    except KeyboardInterrupt:
        raise
    except Exception:
        log.exception("TASO error")
        return None


def ipex(scripted, example_inputs, filename=None):
    if scripted is None:
        return None
    try:
        import intel_extension_for_pytorch

        return intel_extension_for_pytorch.optimize(scripted)
    except KeyboardInterrupt:
        raise
    except Exception:
        log.exception("IPEX error")
        return None
