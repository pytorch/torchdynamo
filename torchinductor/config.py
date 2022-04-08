debug = False


# TODO(jansel): these constants are total guesses without tuning
class cpp:
    threads = -1  # set to cpu_count()
    simdlen = None
    # min_chunk_size = 512
    min_chunk_size = 1


class triton:
    pass
