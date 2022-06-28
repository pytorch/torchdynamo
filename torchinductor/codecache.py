import base64
import functools
import getpass
import hashlib
import logging
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

log = logging.getLogger(__name__)


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


def cpp_compiler():
    if isinstance(config.cpp.cxx, (list, tuple)):
        search = tuple(config.cpp.cxx)
    else:
        search = (config.cpp.cxx,)
    return cpp_compiler_search(search)


@functools.lru_cache(1)
def cpp_compiler_search(search):
    for cxx in search:
        try:
            if cxx is None:
                return install_gcc_via_conda()
            else:
                subprocess.check_output([cxx, "--version"])
                return cxx
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    raise exc.InvalidCxxCompiler()


def install_gcc_via_conda():
    """On older systems, this is a quick way to get a modern compiler"""
    prefix = os.path.join(cache_dir(), "gcc")
    cxx_path = os.path.join(prefix, "bin", "g++")
    if not os.path.exists(cxx_path):
        log.info("Downloading GCC via conda")
        subprocess.check_call(
            [
                "conda",
                "create",
                f"--prefix={prefix}",
                "--channel=conda-forge",
                "-y",
                "python=3.8",
                "gxx",
            ]
        )
        assert os.path.exists(cxx_path)
    return cxx_path


def is_gcc():
    return re.search(r"(gcc|g\+\+)", cpp_compiler())


def cpp_compile_command(input, output, include_pytorch=False):
    if include_pytorch:
        ipaths = cpp_extension.include_paths() + [sysconfig.get_path("include")]
        lpaths = cpp_extension.library_paths() + [sysconfig.get_config_var("LIBDIR")]
        libs = ["c10", "torch", "torch_cpu", "torch_python", "gomp"]
    else:
        ipaths = []
        lpaths = []
        libs = ["gomp"]
    ipaths = " ".join(["-I" + p for p in ipaths])
    lpaths = " ".join(["-L" + p for p in lpaths])
    libs = " ".join(["-l" + p for p in libs])
    return re.sub(
        r"[ \n]+",
        " ",
        f"""
            {cpp_compiler()} -shared -fPIC -Wall -std=c++20 -Wno-unused-variable
            {ipaths} {lpaths} {libs}
            -march=native -O3 -ffast-math -fno-finite-math-only -fopenmp
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
                try:
                    subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    raise exc.CppCompileError(cmd, e.output)

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
        if config.debug:
            print(f"PyCodeCache {path}")
        return cls.cache[key]


@functools.lru_cache(None)
def patch_triton_hackery():
    """
    The following is a copy and paste of triton.code_gen.Kernel.__call__,
    with a bunch of stuff moved to a closure, so it is only called once.

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
        if config.triton.hackery and not config.triton.cudagraphs:
            # this breaks cudagraphs, but speeds up small inputs:
            patch_triton_hackery()
        return PyCodeCache.load(source_code)
