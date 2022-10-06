#!/usr/bin/env python
import sys

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CppExtension

long_description = """
TorchDynamo is a Python-level JIT compiler designed to make unmodified
PyTorch programs faster. TorchDynamo hooks into the frame evaluation API
in CPython (PEP 523) to dynamically modify Python bytecode right before
it is executed. It rewrites Python bytecode in order to extract sequences
of PyTorch operations into an FX Graph which is then just-in-time
compiled with an ensemble of different backends and autotuning.
"""


package_include = [
    "torchdynamo",
    "torchdynamo.*",
    "torchinductor",
    "torchinductor.*",
]

if "develop" in sys.argv:
    package_include += [
        "benchmarks",
        "benchmarks.microbenchmarks",
        "benchmarks.microbenchmarks.*",
    ]

install_requires = [
    "torch>=1.12.0",
    "numpy",
    "tabulate",
    "pyyaml",
    "dill",
    "jinja2",
    "networkx",
    "sympy",
]

setup(
    name="torchdynamo",
    version="1.13.0.dev0",
    url="https://github.com/pytorch/torchdynamo",
    description="A Python-level JIT compiler designed to make unmodified PyTorch programs faster.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jason Ansel",
    author_email="jansel@fb.com",
    license="BSD-3",
    keywords="pytorch machine learning compilers",
    python_requires=">=3.7, <3.11",
    install_requires=install_requires,
    packages=find_packages(include=package_include),
    package_data={
        "torchinductor.codegen": ["*.h", "*.j2"],
    },
    zip_safe=False,
    ext_modules=[
        Extension(
            "torchdynamo._eval_frame",
            ["torchdynamo/_eval_frame.c"],
            extra_compile_args=["-Wall"],
        ),
        CppExtension(
            name="torchdynamo._guards",
            sources=["torchdynamo/_guards.cpp"],
            extra_compile_args=["-std=c++14"],
        ),
    ],
)
