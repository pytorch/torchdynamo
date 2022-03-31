import base64
import getpass
import hashlib
import os
import random
import subprocess
import types
from ctypes import cdll

CPP_COMPILE_CMD = (
    "g++ -shared -fPIC -Wall -std=c++14 "
    "-march=native -O3 -ffast-math "
    "-o{output} {input}"
)


def cache_dir():
    return f"/tmp/{getpass.getuser()}_torchinductor_cache"


def code_hash(code):
    return (
        "c"
        + base64.b32encode(hashlib.sha256(code.encode("utf-8")).digest())[:51]
        .decode("utf-8")
        .lower()
    )


def write(source_code, ext):
    basename = code_hash(source_code)
    subdir = os.path.join(cache_dir(), basename[1:3])
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"{basename}.{ext}")
    if not os.path.exists(path):
        # use a a random temp file for thread safety
        tmp_path = f"{path}.{random.randint(0, 2**31)}"
        with open(tmp_path, "w") as fd:
            fd.write(source_code)
        os.rename(tmp_path, path)
    return basename, path


class CppCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code):
        key, input_path = write(source_code, "cpp")
        if key not in cls.cache:
            output_path = input_path[:-3] + "so"
            if not os.path.exists(output_path):
                cmd = CPP_COMPILE_CMD.format(
                    input=input_path, output=output_path
                ).split(" ")
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
