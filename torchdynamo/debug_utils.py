import copy
import functools
import getpass
import logging
import os
import shutil
import subprocess
import textwrap
import uuid
from collections import Counter

import torch
import torch.fx as fx

import torchdynamo
from torchdynamo import config
from torchdynamo.utils import clone_inputs

log = logging.getLogger(__name__)


def minifier_dir():
    return f"/tmp/minifier_{getpass.getuser()}"


class NNModuleToString:
    safe_reprs = [
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.Softmax,
        torch.nn.ReLU,
        torch.nn.MaxPool2d,
        torch.nn.Embedding,
    ]

    @staticmethod
    def can_convert_to_string(gm):
        cant_convert = set()
        for _, module in gm.named_children():
            if type(module) not in NNModuleToString.safe_reprs:
                cant_convert.add(module)

        if len(cant_convert) > 0:
            log.warning(
                f"Was not able to save the following children modules as reprs {cant_convert}"
            )
            return False
        return True

    @staticmethod
    def convert(gm):
        from torch.nn.modules.module import _addindent

        tab = " " * 4

        model_str = textwrap.dedent(
            """
            from torch.nn import *
            class Repro(torch.nn.Module):
                def __init__(self):
                    super().__init__()
            """
        )

        for module_name, module in gm.named_children():
            module_str = f"{module.__repr__()}"
            model_str += f"{tab*2}self.{module_name} = {module_str}\n"

        for buffer_name, buffer in gm._buffers.items():
            if buffer is None:
                continue
            tensor_str = f"torch.randn({list(buffer.shape)}, dtype={buffer.dtype})"
            model_str += f"{tab*2}self.register_buffer('{buffer_name}', {tensor_str})\n"

        for param_name, param in gm._parameters.items():
            if param is None:
                continue
            tensor_str = f"torch.nn.Parameter(torch.randn({list(param.shape)}, dtype={param.dtype}))"
            model_str += f"{tab*2}self.{param_name} = {tensor_str}\n"

        attrs = dir(gm)
        for attr in attrs:
            if "_tensor_constant" in attr:
                val = getattr(gm, attr)
                model_str += f"    {attr} = {val!r}\n"

        model_str += f"{_addindent(gm.code, 4)}\n"
        return model_str


@functools.lru_cache(None)  # subprocess is expensive
def _cuda_system_info_comment():
    if not torch.cuda.is_available():
        return "# torch.cuda.is_available()==False, no GPU info collected\n"

    model_str = "# CUDA Info: \n"
    cuda_version_out = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE)
    cuda_version_lines = cuda_version_out.stdout.decode().split("\n")
    cuda_version_out = "".join(
        [f"# {s} \n" for s in cuda_version_lines if s not in [""]]
    )
    model_str += f"{cuda_version_out}\n"
    gpu_names = subprocess.run(
        ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
        stdout=subprocess.PIPE,
    )
    gpu_names = gpu_names.stdout.decode().split("\n")
    gpu_names = [name for name in gpu_names if name not in ("", "name")]
    gpu_names = Counter(gpu_names)

    model_str += "# GPU Hardware Info: \n"
    for name, count in gpu_names.items():
        model_str += f"# {name} : {count} \n"
    model_str += "\n"
    return model_str


def generate_compiler_repro_string(gm, args):
    model_str = textwrap.dedent(
        """
        import torch
        from torch import tensor, device
        import torch.fx as fx
        from torchdynamo.testing import rand_strided
        from math import inf
        from torch.fx.experimental.proxy_tensor import make_fx

        """
    )
    model_str += f"# torch version: {torch.version.__version__}\n"
    model_str += f"# torch cuda version: {torch.version.cuda}\n"
    model_str += f"# torch git version: {torch.version.git_version}\n\n\n"
    model_str += _cuda_system_info_comment()

    model_str += NNModuleToString.convert(gm)

    model_str += f"args = {[(tuple(arg.shape), tuple(arg.stride()), arg.dtype, arg.device.type) for arg in args]!r}\n"
    model_str += "args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]\n"
    model_str += "mod = make_fx(Repro())(*args)\n"
    return model_str


INDUCTOR_IMPORT = """
from torchinductor.compile_fx import compile_fx_inner
"""

NVFUSER_IMPORT = """
from torch.fx.passes.backends.nvfuser import NvFuserBackend
nvfuser = NvFuserBackend()
"""

COMPILER_REPRO_OPTIONS = {
    "inductor": (INDUCTOR_IMPORT, "compile_fx_inner", "inductor_fails"),
    "nvfuser": (NVFUSER_IMPORT, "nvfuser", "nvfuser_fails"),
}


def dump_compiler_graph_state(gm, args, compiler_name):
    subdir = f"{minifier_dir()}/checkpoints"
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{len(gm.graph.nodes)}.py")
    print(f"Writing checkpoint with {len(gm.graph.nodes)} nodes to {file_name}")
    with open(file_name, "w") as fd:
        save_graph_repro(fd, gm, args, compiler_name)
    repro_path = os.path.join(torchdynamo.config.base_dir, "repro.py")
    shutil.copyfile(file_name, repro_path)


def save_graph_repro(fd, gm, args, compiler_name):
    fd.write(generate_compiler_repro_string(gm, args))
    fd.write(COMPILER_REPRO_OPTIONS[compiler_name][0])
    fd.write(
        textwrap.dedent(
            f"""
            compiled = {COMPILER_REPRO_OPTIONS[compiler_name][1]}(mod, args)
            compiled(*args)
            """
        )
    )


def isolate_fails(fx_g, args, compiler_name: str, env=None):
    if env is None:
        env = {}
    subdir = f"{minifier_dir()}/isolate"
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{str(uuid.uuid4())[:5]}.py")
    with open(file_name, "w") as fd:
        fd.write(generate_compiler_repro_string(fx_g, args))
        fail_fn = COMPILER_REPRO_OPTIONS[compiler_name][2]
        fd.write(
            textwrap.dedent(
                f"""
                from torchdynamo.debug_utils import {fail_fn}
                """
            )
        )
        fd.write(
            textwrap.dedent(
                f"""
                if {fail_fn}(mod, args):
                    exit(1)
                else:
                    exit(0)
                """
            )
        )
    new_env = os.environ.copy()
    new_env = {**new_env, **env}
    p = subprocess.Popen(
        ["python", file_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=new_env,
    )
    out, err = p.communicate()
    if p.returncode != 0:
        print(textwrap.indent(out.decode("utf-8"), prefix=">>  "))
        print(textwrap.indent(err.decode("utf-8"), prefix=">>  "))
        return True
    return False


def inductor_fails(fx_g, args, check_str=None):
    from torchinductor import config
    from torchinductor.compile_fx import compile_fx_inner

    config.triton.autotune = False

    try:
        result = fx_g(*args)
        assert isinstance(result, (tuple, list))
        assert not any([isinstance(x, (tuple, list)) for x in result])
    except Exception:
        return False

    try:
        compile_mod = compile_fx_inner(fx_g, args)
        compile_mod = compile_mod(*args)
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False


def nvfuser_fails(fx_g, args, check_str=None):
    from torch.fx.passes.backends.nvfuser import NvFuserBackend

    nvfuser = NvFuserBackend()

    try:
        compile_mod = nvfuser(fx_g, args)
        compile_mod = compile_mod(*args)
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False


def dump_to_minify(gm, args, compiler_name: str):
    favored_device = 1 if torch.cuda.device_count() >= 2 else 0
    with open(
        os.path.join(torchdynamo.config.base_dir, "minifier_launcher.py"), "w"
    ) as fd:
        fd.write(generate_compiler_repro_string(gm, args))
        fd.write("\n")
        fd.write(
            textwrap.dedent(
                f"""
                from functools import partial
                from torchdynamo.debug_utils import (
                    isolate_fails,
                    dump_compiler_graph_state,
                )
                from functorch.compile import minifier

                env_variables = {{"CUDA_VISIBLE_DEVICES": "{favored_device}"}}

                minifier(
                    mod,
                    args,
                    module_fails=partial(isolate_fails, env=env_variables, compiler_name="{compiler_name}"),
                    dump_state=partial(dump_compiler_graph_state, compiler_name="{compiler_name}"),
                )
                """
            )
        )
    print("wrote out to minifier_launcher.py")


def wrap_compiler_debug(compiler, compiler_name: str):
    """
    Minifier for Fx Graph modules after Aot Autograd has finished. We wrap both
    forward and backward call separately with the backend compiler - like
    inductor or nvfuser. Intercepting after Aot Autograd presents neat
    abstration, where all the params are lifted as graph inputs, making it easy
    to save the graph as a string.
    """

    @functools.wraps(compiler)
    def debug_wrapper(gm, example_inputs, **kwargs):
        orig_graph = copy.deepcopy(gm.graph)
        assert config.repro_after in ("dynamo", "aot", None)
        if config.repro_after == "aot":
            try:
                compiled_fn = compiler(gm, example_inputs, **kwargs)
                compiled_fn(*example_inputs)
            except Exception as e:
                if config.repro_level == 1:
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph), example_inputs, compiler_name
                    )
                elif config.repro_level == 2:
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph), example_inputs, compiler_name
                    )
                raise e
        else:
            compiled_fn = compiler(gm, example_inputs, **kwargs)

        return compiled_fn

    return debug_wrapper


def run_fwd_maybe_bwd(gm, args):
    """
    Runs a forward and possibly backward iteration for a given mod and args.
    """
    from torchdynamo.testing import collect_results
    from torchdynamo.testing import reduce_to_scalar_loss
    from torchdynamo.testing import requires_bwd_pass

    gm = copy.deepcopy(gm)
    args = clone_inputs(args)
    gm.zero_grad(True)
    out = gm(*args)
    if requires_bwd_pass(out):
        loss = reduce_to_scalar_loss(out)
        loss.backward()
        return collect_results(gm, out, loss, [])
    else:
        return out


def same_two_models(gm, opt_gm, example_inputs):
    """
    Check two models have same accuracy.
    """
    from torchdynamo.utils import same

    ref = run_fwd_maybe_bwd(gm, example_inputs)

    fp64_model, fp64_examples = cast_to_fp64(
        copy.deepcopy(gm), clone_inputs(example_inputs)
    )
    fp64_ref = run_fwd_maybe_bwd(fp64_model, fp64_examples)

    res = run_fwd_maybe_bwd(opt_gm, example_inputs)

    passing = same(ref, res, fp64_ref, tol=0.001)
    return passing


def cast_to(dtype, model, inputs):
    from torch.utils._pytree import tree_map

    # cast model and inputs to fp16
    model = model.to(dtype)

    inputs = tree_map(
        lambda x: x.to(dtype)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x,
        inputs,
    )
    return model, inputs


def cast_to_fp64(model, inputs):
    return cast_to(torch.float64, model, inputs)


def generate_dynamo_fx_repro_string(model_str, args, compiler_name):
    """
    Generate a repro string for backend-agnostic minified version.
    """

    imports = textwrap.dedent(
        """
        import torch
        import torchdynamo
        from torch import tensor, device
        import torch.fx as fx
        from torchdynamo.testing import rand_strided
        from math import inf
        from torchdynamo.debug_utils import run_fwd_maybe_bwd

        """
    )

    prep_inputs = textwrap.dedent(
        f"""
        args = {[(tuple(a.shape), tuple(a.stride()), a.dtype, a.device.type, a.requires_grad) for a in args]}
        args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]

        """
    )

    setup_module = textwrap.dedent(
        f"""
        mod = Repro().cuda()
        opt_mod = torchdynamo.optimize("{compiler_name}")(mod)

        """
    )

    run_module = textwrap.dedent(
        f"""
        with torch.cuda.amp.autocast(enabled={torch.is_autocast_enabled()}):
            ref = run_fwd_maybe_bwd(mod, args)
            res = run_fwd_maybe_bwd(opt_mod, args)
        """
    )

    return imports + model_str + setup_module + prep_inputs + run_module


def dump_backend_repro_as_file(gm, args, compiler_name):
    """
    Saves the repro to a repro.py file
    """
    subdir = f"{minifier_dir()}/checkpoints"
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{len(gm.graph.nodes)}.py")
    print(f"Writing checkpoint with {len(gm.graph.nodes)} nodes to {file_name}")

    model_str = NNModuleToString.convert(gm)
    with open(file_name, "w") as fd:
        fd.write(generate_dynamo_fx_repro_string(model_str, args, compiler_name))
    print(f"Writing checkpoint with {len(gm.graph.nodes)} locally to repro.py")
    repro_path = os.path.join(torchdynamo.config.base_dir, "repro.py")
    shutil.copyfile(file_name, repro_path)


def dump_backend_repro_as_tarfile(gm, args, compiler_name):
    """
    Saves the repro in repro.tar.gz, as opposed to a file. This is used for
    cases, where we can't convert a Fx GraphModule to a string, and therefore
    fallback to to_folder for serialization. We accompany this with a repro.py
    script that imports the saved module, sets it up and runs the model to repro
    the error.
    """
    import tarfile

    subdir = f"{minifier_dir()}/checkpoints"
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)

    tmp_dir = os.path.join(subdir, f"{len(gm.graph.nodes)}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    file_name = os.path.join(tmp_dir, "repro.py")
    gm_dir = os.path.join(tmp_dir, "module")
    if not os.path.exists(gm_dir):
        os.makedirs(gm_dir, exist_ok=True)
    for node in gm.graph.nodes:
        new_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, torch.device):
                v = v.type
            new_kwargs[k] = v
        node.kwargs = new_kwargs
    gm.recompile()

    print(f"Writing checkpoint with {len(gm.graph.nodes)} nodes to {file_name}")
    with open(file_name, "w") as fd:
        # TODO - Add the readable version of to_folder when available
        gm.to_folder(gm_dir, "Repro")
        fd.write(
            generate_dynamo_fx_repro_string(
                "from module import Repro", args, compiler_name
            )
        )

    local_dir = os.path.join(torchdynamo.config.base_dir, "repro")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    shutil.copytree(tmp_dir, local_dir)
    local_tar_file = os.path.join(torchdynamo.config.base_dir, "repro.tar.gz")
    print(f"Writing checkpoint with {len(gm.graph.nodes)} locally to {local_tar_file}")
    with tarfile.open(local_tar_file, "w:gz") as tar:
        tar.add(local_dir, arcname=os.path.basename(local_dir))


def dump_backend_state(gm, args, compiler_name):
    """
    Dumps the dynamo graph to repro the issue.
    1) It tries to convert Fx GraphModule to a string. If we can, it writes to a
    repro.py file.
    2) If we can't convert Fx GraphModule to a string, we use to_folder to save
    the module and save a tar file.
    """
    if NNModuleToString.can_convert_to_string(gm):
        return dump_backend_repro_as_file(gm, args, compiler_name)
    return dump_backend_repro_as_tarfile(gm, args, compiler_name)


def backend_fails(gm, example_inputs, compiler_fn, orig_failure):
    """
    Minifier uses this function to identify if the minified graph module fails
    with the same error.

    One caveat is that minifier can potentially go into a wrong direction when
    the resulting graph module fails for a different reason. To avoid this, we
    save the string for the original exception and check similarity between new
    and old exception. They can be somewhat different in some cases, when the
    exception string depends on the failing node information. So, we have a
    loose similarity metric to guide the minifier path.
    """
    from difflib import SequenceMatcher

    try:
        compiled_gm = compiler_fn(gm, example_inputs)
        run_fwd_maybe_bwd(compiled_gm, clone_inputs(example_inputs))
        return False
    except Exception as e:
        new_failure = str(e)
        if SequenceMatcher(None, orig_failure, new_failure).ratio() > 0.5:
            return True
        return False


def wrap_backend_debug(compiler_fn, compiler_name: str):
    """
    A minifier decorator that wraps the TorchDynamo produced Fx graph modules.
    As opposed to wrap_compiler_debug, this wrapper intercepts at the
    TorchDynamo produced Fx Graph Module. This makes it backend-agnostic to some
    level, e.g., it is useful for minifying issues related to Aot Autograd
    tracing.  If an error is found, we minify and save the minified repro in
    repro.tar.gz.
    """
    from functorch.compile import minifier

    @functools.wraps(compiler_fn)
    def debug_wrapper(gm, example_inputs, **kwargs):
        assert config.repro_after in ("dynamo", "aot", None)
        if config.repro_after == "dynamo":
            # Ensure that we fail when backend fails
            config.raise_on_backend_error = True
            try:
                compiled_gm = compiler_fn(gm, example_inputs, **kwargs)
                run_fwd_maybe_bwd(compiled_gm, clone_inputs(example_inputs))
            except Exception as exc:
                orig_failure = str(exc)
                log.warning(
                    f"Compiled Fx GraphModule failed with {orig_failure}. Starting minifier."
                )
                dump_state_fn = functools.partial(
                    dump_backend_state, compiler_name=compiler_name
                )
                dump_state_fn(
                    fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs
                )
                if config.repro_level > 1:
                    # As opposed to using dump_to_minify, like we do in
                    # wrap_compiler_debug, we directly run minifier here. This
                    # is because we can't serialize compiler_fn here.

                    # The minified version uses
                    # torchdynamo.optimize(compiler_str) to repro the error.

                    # Directly running the minifier could be bad if something
                    # goes bad while minification is running. We will have to
                    # investigate how to add isolation.
                    fails_fn = functools.partial(
                        backend_fails,
                        compiler_fn=compiler_fn,
                        orig_failure=orig_failure,
                    )
                    minifier(
                        gm,
                        example_inputs,
                        module_fails=fails_fn,
                        dump_state=dump_state_fn,
                    )
                    raise exc
        else:
            compiled_gm = compiler_fn(gm, example_inputs, **kwargs)

        return compiled_gm

    debug_wrapper._torchdynamo_orig_callable = compiler_fn

    return debug_wrapper
