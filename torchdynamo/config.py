from os.path import abspath
from os.path import dirname

import torch

# print out lots of stuff
debug = False

# need this many ops to create an FX graph
minimum_call_count = 1

# max graphs to create per frame
max_blocks = 64

# turn on/off DCE pass
dead_code_elimination = True

# disable (for a function) when cache reaches this size
cache_size_limit = 128

# Assume these functions return constants
constant_functions = {
    torch.jit.is_scripting: True,
    torch.jit.is_tracing: False,
}

# root folder of the project
base_dir = dirname(dirname(abspath(__file__)))

# Also need to remove SPECIALIZE_SHAPES_AND_STRIDES from _guards.cpp
dynamic_shapes = False
