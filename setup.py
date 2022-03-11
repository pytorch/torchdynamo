#!/usr/bin/env python
import sys

from setuptools import Extension
from setuptools import setup
from torch.utils.cpp_extension import CppExtension

assert (3, 7) <= sys.version_info < (3, 9), "requires python 3.7 or 3.8"

setup(
    name="torchdynamo",
    version="0.1",
    author="Jason Ansel",
    author_email="jansel@fb.com",
    packages=["torchdynamo"],
    ext_modules=[
        Extension(
            "torchdynamo._eval_frame",
            ["torchdynamo/_eval_frame.c"],
            extra_compile_args=["-Werror"],
        ),
        CppExtension(
            name="torchdynamo._guards",
            sources=["torchdynamo/_guards.cpp"],
        ),
    ],
)
