import torch
import ptdynamo
import os

SKIP_DIRS = [os.path.dirname(torch.__file__) + "/",
             os.path.dirname(ptdynamo.__file__) + "/"]


def check(filename):
    return any(filename.startswith(d) for d in SKIP_DIRS)

