import base64
import functools
import getpass
import hashlib
import os
import random
import re
import subprocess
import sysconfig
import types
from ctypes import cdll

from torch.utils import cpp_extension

from . import config
from . import exc


def cache_dir():
    return f"/tmp/torchinductor_{getpass.getuser()}"


def code_hash(code):
    return (
        "c"
        + base64.b32encode(hashlib.sha256(code.encode("utf-8")).digest())[:51]
        .decode("utf-8")
        .lower()
    )


def write(source_code, ext, extra=""):
    basename = code_hash(source_code + extra)
    subdir = os.path.join(cache_dir(), basename[1:3])
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"{basename}.{ext}")
    if not os.path.exists(path):
        # use a random temp file for thread safety
        tmp_path = f"{path}.{random.randint(0, 2**31)}"
        with open(tmp_path, "w") as fd:
            fd.write(source_code)
        os.rename(tmp_path, path)
    return basename, path


@functools.lru_cache(None)
def cpp_compiler():
    if isinstance(config.cpp.cxx, str):
        return config.cpp.cxx
    for cxx in config.cpp.cxx:
        try:
            subprocess.check_output([cxx, "--version"])
            return cxx
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    raise exc.InvalidCxxCompiler()


def is_gcc():
    return re.search(r"(gcc|g\+\+)", cpp_compiler())


def cpp_compile_command(input, output, include_pytorch=False):
    if include_pytorch:
        ipaths = cpp_extension.include_paths() + [sysconfig.get_path("include")]
        lpaths = cpp_extension.library_paths() + [sysconfig.get_config_var("LIBDIR")]
        libs = ["c10", "torch", "torch_cpu", "torch_python", "omp5"]
    else:
        ipaths = []
        lpaths = []
        libs = ["omp5"]
    ipaths = " ".join(["-I" + p for p in ipaths])
    lpaths = " ".join(["-L" + p for p in lpaths])
    libs = " ".join(["-l" + p for p in libs])
    return re.sub(
        r"[ \n]+",
        " ",
        f"""
            {cpp_compiler()} -shared -fPIC -Wall -std=c++14 -Wno-unused-variable
            {ipaths} {lpaths} {libs}
            -march=native -O3 -ffast-math -fopenmp
            -o{output} {input}
        """,
    ).strip()


class CppCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code):
        key, input_path = write(source_code, "cpp", extra=cpp_compile_command("i", "o"))
        if key not in cls.cache:
            output_path = input_path[:-3] + "so"
            if not os.path.exists(output_path):
                cmd = cpp_compile_command(input=input_path, output=output_path).split(
                    " "
                )
                subprocess.check_call(cmd)
            cls.cache[key] = cdll.LoadLibrary(output_path)
            cls.cache[key].key = key
        return cls.cache[key]


class PyCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code):
        key, path = write(source_code, "py")
        if key not in cls.cache:
            with open(path) as f:
                code = compile(f.read(), path, "exec")
                mod = types.ModuleType(f"{__name__}.{key}")
                exec(code, mod.__dict__, mod.__dict__)
                cls.cache[key] = mod
                cls.cache[key].key = key
        return cls.cache[key]


@functools.lru_cache(None)
def patch_triton_hackery():
    """
    The following is a copy and paste of triton.code_gen.Kernel.__call__,
    with a bunch of stuff moved to a closure so it is only called once.

    This makes tiny kernels run ~1.2x faster.
    """
    import torch
    from triton.code_gen import Kernel
    from triton.code_gen import _triton

    # query device index and cuda stream
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    cc = torch.cuda.get_device_capability(device)
    cc = str(cc[0]) + "-" + str(cc[1])
    stream = torch.cuda.current_stream(device).cuda_stream

    def faster_triton_kernel_call(
        self, *wargs, grid, num_warps=4, num_stages=2, **kwargs
    ):
        # handle arguments passed by name
        kwargs = {
            self.fn.arg_names.index(name): value for name, value in kwargs.items()
        }
        wargs = list(wargs)
        for i, pos in enumerate(sorted(kwargs)):
            wargs.insert(pos + i, kwargs[pos])

        if len(wargs) != len(self.fn.arg_names):
            raise TypeError(
                f"Function takes {len(self.fn.arg_names)} positional arguments but {len(wargs)} were given"
            )

        # handle annotations
        for pos, _type in self.fn.annotations.items():
            wargs[pos] = _type(wargs[pos])

        # check that tensors are on GPU.
        # for arg in wargs:
        #     if hasattr(arg, 'data_ptr'):
        #         assert arg.is_cuda, "All tensors must be on GPU!"

        return _triton.runtime.launch(
            wargs,
            self.fn.do_not_specialize,
            self.fn.cache_key + cc,
            self.fn.arg_names,
            device,
            stream,
            self.fn.bin_cache,
            num_warps,
            num_stages,
            self.add_to_cache,
            grid,
        )

    Kernel.__call__ = faster_triton_kernel_call


@functools.lru_cache(None)
def patch_triton_dir():
    os.environ["TRITON_CACHE_DIR"] = os.environ.get(
        "TRITON_CACHE_DIR", os.path.join(cache_dir(), "triton")
    )


class TritonCodeCache:
    @classmethod
    def load(cls, source_code):
        patch_triton_dir()
        # this breaks cudagraphs, but speeds up small inputs:
        # patch_triton_hackery()
        return PyCodeCache.load(source_code)


def pointwise_heuristics():
    """args to @triton.heuristics()"""
    from triton import next_power_of_2

    def need_mask(args):
        return True
        return (args["numel"] % args["BLOCK_SIZE"]) > 0

    def block_size(args):
        return min(1024, next_power_of_2(args["numel"]))
        # return 1024

    return {
        "BLOCK_SIZE": block_size,
        "NEED_MASK": need_mask,
    }


def reduction_heuristics():
    """args to @triton.heuristics()"""
    from triton import next_power_of_2

    def need_mask(args):
        return True
        return (args["numel"] % args["BLOCK_SIZE"]) > 0 or (
            args["reduction_numel"] % args["REDUCTION_SIZE"]
        ) > 0

    def reduction_size(args):
        return next_power_of_2(args["reduction_numel"])

    def block_size(args):
        return max(next_power_of_2(1024 // args["REDUCTION_SIZE"]), 1)

    # def num_warps(args):
    #     n = args["numel"] * args["reduction_numel"]
    #     if n <= 65536:
    #         return 2
    #     return 8

    return {
        # "num_warps": num_warps,
        "REDUCTION_SIZE": reduction_size,
        "BLOCK_SIZE": block_size,
        "NEED_MASK": need_mask,
    }


def grid(numel):
    """Helper function to compute triton grids for pointwise ops"""

    def grid_fn(meta):
        bs = meta["BLOCK_SIZE"]
        # inlined triton.cdiv
        return ((numel + bs - 1) // bs,)

    return grid_fn
