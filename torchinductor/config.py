debug = False

dce = False

forward_only = True


class cpp:
    threads = -1  # set to cpu_count()
    simdlen = None
    min_chunk_size = 4096
    cxx = "g++-10"
    # cxx = "clang++-10"


class triton:
    pass
