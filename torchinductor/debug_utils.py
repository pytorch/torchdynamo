import os
import subprocess
import textwrap
import uuid

import torchdynamo
from torchinductor.codecache import cache_dir


def generate_repro_string(gm, args):
    model_str = textwrap.dedent(
        """
        import torch
        from torch import tensor, device
        import torch.fx as fx
        from torchdynamo.testing import rand_strided
        from math import inf
        from torchinductor.compile_fx import compile_fx_inner
        from torch.fx.experimental.proxy_tensor import make_fx

        """
    )
    model_str += "class Repro(torch.nn.Module):\n"
    attrs = dir(gm)
    for attr in attrs:
        if "_tensor_constant" in attr:
            val = getattr(gm, attr)
            model_str += f"    {attr} = {val!r}\n"
    model_str += textwrap.indent(gm.code, "    ")
    model_str += "\n"

    model_str += f"args = {[(tuple(arg.shape), tuple(arg.stride()), arg.dtype, arg.device.type) for arg in args]!r}\n"
    model_str += "args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]\n"
    model_str += "mod = make_fx(Repro())(*args)\n"
    return model_str


def dump_to_repro(gm, args):
    with open(os.path.join(torchdynamo.config.base_dir, "repro.py"), "w") as fd:
        fd.write(generate_repro_string(gm, args))
        fd.write(
            textwrap.dedent(
                """
                expected = Repro().forward(*args)
                with torchdynamo.optimize("inductor", nopython=True):
                    actual = Repro().forward(*args)
                assert same(actual, expected)
                """
            )
        )
        print("wrote repro.py")


def dump_state_inductor(gm, args):
    subdir = f"{cache_dir()}/minimizer_checkpoints"
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{len(gm.graph.nodes)}.py")
    print(f"Writing checkpoint with {len(gm.graph.nodes)} nodes to {file_name}")
    with open(file_name, "w") as fd:
        fd.write(generate_repro_string(gm, args))
        fd.write(
            textwrap.dedent(
                """
                compiled = compile_fx_inner(mod, args)
                compiled(*args)
                """
            )
        )


def isolate_inductor_fails(fx_g, args, env=None):
    if env is None:
        env = {}
    subdir = f"{cache_dir()}/minimizer"
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{str(uuid.uuid4())[:5]}.py")
    with open(file_name, "w") as fd:
        fd.write(generate_repro_string(fx_g, args))
        fd.write(
            textwrap.dedent(
                """
                from torchinductor.debug_utils import inductor_fails
                """
            )
        )
        fd.write(
            textwrap.dedent(
                """
                if inductor_fails(mod, args):
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

    config.autotune = False

    try:
        compile_mod = compile_fx_inner(fx_g, args)
        compile_mod = compile_mod(*args)
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False


def dump_to_minify(gm, args):
    with open(
        os.path.join(torchdynamo.config.base_dir, "minimizer_repro.py"), "w"
    ) as fd:
        fd.write(generate_repro_string(gm, args))
        fd.write("\n")
        fd.write(
            textwrap.dedent(
                """
                from functools import partial
                from torchinductor.debug_utils import (
                    inductor_fails,
                    isolate_inductor_fails,
                    dump_state_inductor,
                )
                from functorch.compile import minifier

                env_variables = {"CUDA_VISIBLE_DEVICES": "1"}

                minifier(
                    mod,
                    args,
                    module_fails=partial(isolate_inductor_fails, env=env_variables),
                    dump_state=dump_state_inductor,
                )
                """
            )
        )
    print("wrote out to minimizer_repro.py")
