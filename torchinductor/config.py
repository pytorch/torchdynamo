# add some debug printouts
debug = False

# dead code elimination
dce = False

# assume there will be no backwards
forward_only = True

# set to False to burn in all shapes
dynamic_shapes = True

# assuming weight matrices are fixed size
static_weight_shapes = True


# config specific to codegen/cpp.pp
class cpp:
    threads = -1  # set to cpu_count()
    simdlen = None
    min_chunk_size = 4096
    cxx = ("g++-10", "g++")
    # cxx = "clang++-12"


# config specific to codegen/triton.py
class triton:
    pass
