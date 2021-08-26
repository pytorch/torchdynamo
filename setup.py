#!/usr/bin/env python

from distutils.core import setup, Extension

setup(name='ptdynamo',
      version='0.1',
      author="Jason Ansel",
      author_email="jansel@jansel.net",
      packages=["ptdynamo"],
      ext_modules=[Extension('ptdynamo.eval_frame', ['ptdynamo/eval_frame.c'])])
