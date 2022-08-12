import inspect
import os
import subprocess
import textwrap
import uuid
from functools import wraps

import torch

import torchdynamo
from torchinductor import config
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


def isolate_checker(f):
    @wraps(f)
    def isolated_f(fx_g, args):
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
                    isolate_checker = lambda f: f
                    """
                )
            )
            fd.write(inspect.getsource(f))
            fd.write(f"graph_fails = {f.__name__}")
            fd.write(
                textwrap.dedent(
                    """
                    if graph_fails(mod, args):
                        exit(1)
                    else:
                        exit(0)
                    """
                )
            )
        p = subprocess.Popen(
            ["python", file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = p.communicate()
        if p.returncode != 0:
            print(textwrap.indent(out.decode("utf-8"), prefix=">>  "))
            print(textwrap.indent(err.decode("utf-8"), prefix=">>  "))
            return True
        return False

    return isolated_f


def inductor_fails(fx_g, args, check_str=None):
    from torchinductor.compile_fx import compile_fx_inner

    try:
        compile_mod = compile_fx_inner(fx_g, args)
        compile_mod = compile_mod(*args)
    except Exception as e:
        if check_str is not None and check_str not in str(e):
            return False
        print(e)
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
                import torchdynamo
                from torchinductor.debug_utils import inductor_fails

                from functorch.compile import minifier

                minifier(mod, args, inductor_fails)
                """
            )
        )
    print("wrote out to minimizer_repro.py")
