#!/usr/bin/env python

from setuptools import Extension
from setuptools import setup
from torch.utils.cpp_extension import CppExtension

setup(
    name="torchdynamo",
    version="0.1",
    author="Jason Ansel",
    author_email="jansel@jansel.net",
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
