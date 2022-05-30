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
            {cpp_compiler()} -shared -fPIC -Wall -std=c++14 -Wno-unused-variable
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
            print("PyCodeCache", path)
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


def block_size_fn(maximum, hint, key):
    from triton import next_power_of_2

    if next_power_of_2(hint) >= maximum:
        return lambda args: maximum

    def block_size(args):
        return min(maximum, next_power_of_2(args[key]))

    return block_size


def pointwise_heuristics(size_hints):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    from triton import heuristics

    # TODO(jansel): try tuning these, current just a guess
    if len(size_hints) == 1:
        return heuristics(
            {
                "XBLOCK": block_size_fn(1024, size_hints[0], "xnumel"),
            }
        )
    if len(size_hints) == 2:
        return heuristics(
            {
                "XBLOCK": block_size_fn(32, size_hints[0], "xnumel"),
                "YBLOCK": block_size_fn(32, size_hints[1], "ynumel"),
            }
        )
    if len(size_hints) == 3:
        return heuristics(
            {
                "XBLOCK": block_size_fn(16, size_hints[0], "xnumel"),
                "YBLOCK": block_size_fn(16, size_hints[1], "ynumel"),
                "ZBLOCK": block_size_fn(16, size_hints[2], "znumel"),
            }
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def reduction_heuristics(size_hints):
    """args to @triton.heuristics()"""
    from triton import heuristics
    from triton import next_power_of_2

    def reduction_size(args):
        return next_power_of_2(args["rnumel"])

    if len(size_hints) == 2:
        return heuristics(
            {
                "RBLOCK": reduction_size,
                "XBLOCK": block_size_fn(
                    next_power_of_2(1024 // size_hints[-1] or 1),
                    size_hints[0],
                    "xnumel",
                ),
            }
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def cdiv(numel, bs):
    return (numel + bs - 1) // bs


def grid(xnumel, ynumel=None, znumel=None):
    """Helper function to compute triton grids"""

    def grid_fn(meta):
        result = [cdiv(xnumel, meta["XBLOCK"])]
        if ynumel:
            result.append(cdiv(ynumel, meta["YBLOCK"]))
            if znumel:
                result.append(cdiv(znumel, meta["ZBLOCK"]))
        return result

    return grid_fn
