import subprocess
import os
from torch.testing._internal.common_methods_invocations import op_db


if __name__ == "__main__":
    i = 0
    while i < len(op_db):
        start = i
        end = i + 20
        os.environ["PYTORCH_TEST_RANGE_START"] = f"{start}"
        os.environ["PYTORCH_TEST_RANGE_END"] = f"{end}"
        subprocess.run(["pytest", "test/test_torchinductor_opinfo.py"])
        i = end + 1
