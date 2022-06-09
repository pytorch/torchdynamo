# add some debug printouts
debug = False

# dead code elimination
dce = False

# assume there will be no backwards
forward_only = True

# assume input tensors are dynamic
dynamic_shapes = True

# assume weight tensors are fixed size
static_weight_shapes = True

# enable some approximation algorithms
approximations = False

# put correctness assertions in generated code
size_asserts = True

# enable loop reordering based on input orders
pick_loop_orders = True

# generate inplace computations
inplace_buffers = False

# codegen benchmark harness
benchmark_harness = True


# config specific to codegen/cpp.pp
class cpp:
    threads = -1  # set to cpu_count()
    simdlen = None
    min_chunk_size = 4096
    cxx = ("g++-10", "g++")
    # cxx = "clang++-12"


# config specific to codegen/triton.py
class triton:

    # Use cudagraphs on output code
    cudagraphs = True

    # Monkey patching to lower overheads
    hackery = False

    # use triton conv as backend
    use_conv = False

    # Always load full blocks (rather than broadcasting inside the block)
    dense_indexing = False

    # limit tiling dimensions
    max_tiles = 2

    # put each kernel in its own file
    many_files = False

    # use triton.autotune?
    autotune = True

    # enable codegen to use Triton's mm
    use_mm = False
