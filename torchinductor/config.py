# add some debug printouts
debug = False

# dead code elimination
dce = False

# assume there will be no backwards
forward_only = True

# assume input tensors are dynamic
dynamic_shapes = True

# assume weight tensors are fixed size
static_weight_shapes = False

# enable some approximation algorithms
approximations = False

# put correctness assertions in generated code
size_asserts = True

# enable loop reordering based on input orders
pick_loop_orders = True

# generate inplace computations
inplace_buffers = False


# config specific to codegen/cpp.pp
class cpp:
    threads = -1  # set to cpu_count()
    simdlen = None
    min_chunk_size = 4096
    cxx = ("g++-10", "g++")
    # cxx = "clang++-12"


# config specific to codegen/triton.py
class triton:
    cudagraphs = True
    hackery = False
