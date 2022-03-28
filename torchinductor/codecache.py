import base64
import functools
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


@functools.lru_cache(None)
def cache_dir():
    # TODO(jansel): should likely change this to ~/.cache/blah
    path = f"/tmp/{getpass.getuser()}_torchinductor_cache"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def code_hash(code):
    return "c" + base64.urlsafe_b64encode(
        hashlib.sha256(code.encode("utf-8")).digest()
    )[:39].decode("utf-8").replace("-", "_")


def write(source_code, ext):
    basename = code_hash(source_code)
    path = os.path.join(cache_dir(), f"{basename}.{ext}")
    if not os.path.exists(path):
        # use a a random temp file for thread safety
        tmp_path = f"{path}.{random.randint(0, 2**31)}"
        with open(tmp_path, "w") as fd:
            fd.write(source_code)
        os.rename(tmp_path, path)
    return basename, path


class CppCodeCache:
    cache = dict()

    @classmethod
    def load(cls, source_code):
        key, input_path = write(source_code, "cpp")
        output_path = input_path[:-3] + "so"
        if key not in cls.cache:
            if not os.path.exists(output_path):
                cmd = CPP_COMPILE_CMD.format(
                    input=input_path, output=output_path
                ).split(" ")
                print(cmd)
                subprocess.check_call(cmd)
            cls.cache[key] = cdll.LoadLibrary(output_path)
        return cls.cache[key]


class PyCodeCache:
    cache = dict()

    @classmethod
    def load(cls, source_code):
        key, path = write(source_code, "py")
        if key not in cls.cache:
            with open(path) as f:
                code = compile(f.read(), path, "exec")
                mod = types.ModuleType(
                    f"{__name__}.{key}"
                )
                exec(code, mod.__dict__, mod.__dict__)
                mod.key = key
                cls.cache[key] = mod
        return cls.cache[key]
