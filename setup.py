#!/usr/bin/env python

from setuptools import Extension
from setuptools import setup

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
        )
    ],
)
