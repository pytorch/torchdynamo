import os
from os.path import abspath
from os.path import dirname

import torch

# print out lots of stuff
debug = False

# an unreasonable amount of debug printouts
trace = False

# need this many ops to create an FX graph
minimum_call_count = 1

# turn on/off DCE pass
dead_code_elimination = True

# disable (for a function) when cache reaches this size
cache_size_limit = 64

# Assume these functions return constants
constant_functions = {
    torch.jit.is_scripting: False,
    torch.jit.is_tracing: False,
    torch._C._get_tracing_state: None,
}

# root folder of the project
base_dir = dirname(dirname(abspath(__file__)))

# don't specialize on shapes and strides and put shape ops in graph
dynamic_shapes = os.environ.get("TORCHDYNAMO_DYNAMIC_SHAPES") == "1"

# Set this to False to assume nn.Modules() contents are immutable (similar assumption as freezing)
guard_nn_modules = False

# Run the FX graph as it is created to get better type information
dynamic_propagation = True

# run FX normalization passes in optimizer
normalize_ir = True
