#!/usr/bin/env python

from setuptools import setup, Extension

setup(name='torchdynamo',
      version='0.1',
      author="Jason Ansel",
      author_email="jansel@jansel.net",
      packages=["torchdynamo"],
      ext_modules=[Extension('torchdynamo.eval_frame', ['torchdynamo/eval_frame.c'])])
