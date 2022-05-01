debug = False

dce = False

forward_only = True

dynamic_shapes = True


class cpp:
    threads = -1  # set to cpu_count()
    simdlen = None
    min_chunk_size = 4096
    cxx = ("g++-10", "g++")
    # cxx = "clang++-12"


class triton:
    pass
