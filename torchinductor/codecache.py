import base64
import functools
import getpass
import hashlib
import logging
import os
import re
import subprocess
import sysconfig
import tempfile
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
        # use a temp file for thread safety
        fd, tmp_path = tempfile.mkstemp(dir=subdir)
        with os.fdopen(fd, "w") as f:
            f.write(source_code)
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
                "--quiet",
                "-y",
                "python=3.8",
                "gxx",
            ],
            stdout=subprocess.PIPE,
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
        # Note - this is effectively a header only inclusion. Usage of some header files may result in
        # symbol not found, if those header files require a library.
        # For those cases, include the lpath and libs command as we do for pytorch above.
        # This approach allows us to only pay for what we use.
        ipaths = cpp_extension.include_paths() + [sysconfig.get_path("include")]
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
                mod.__file__ = path
                exec(code, mod.__dict__, mod.__dict__)
                cls.cache[key] = mod
                cls.cache[key].key = key
        return cls.cache[key]


@functools.lru_cache(None)
def patch_triton_dir():
    os.environ["TRITON_CACHE_DIR"] = os.environ.get(
        "TRITON_CACHE_DIR", os.path.join(cache_dir(), "triton")
    )


class TritonCodeCache:
    @classmethod
    def load(cls, source_code):
        patch_triton_dir()
        return PyCodeCache.load(source_code)
