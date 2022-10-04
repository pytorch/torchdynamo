import subprocess
import sys
import warnings

MIN_PYTHON_VERSION = (3, 7)
MIN_CUDA_VERSION = (11, 6)

def check_python():
    if sys.version_info < MIN_PYTHON_VERSION:
        raise RuntimeError(
            f"Python version not supported: {sys.version_info} "
            f"- minimum requirement: {MIN_PYTHON_VERSION}\n"
        )

def check_pip_deps():
    # Checks for correct pip dependencies, according to setup.py.
    # Also checks for correct torch version.
    proc = subprocess.run(["pip", "check"], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "`pip` dependencies broken:\n" + 
            proc.stdout.decode("utf-8")
        )
    try:
        import torchdynamo
    except ImportError:
        raise RuntimeError("`torchdynamo` not installed\n")

def check_cuda():
    import torch
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available for torch\n")
        return
    if torch.torch_version.TorchVersion(torch.version.cuda) < MIN_CUDA_VERSION:
        raise RuntimeError(
            f"CUDA version not supported: {torch.version.cuda} "
            f"- minimum requirement: {MIN_CUDA_VERSION}\n"
        )
    # check if torch cuda version matches system cuda version?
    # running pytorch code with gpu tensors should fail if there is a mismatch?

# add gpu tests
def check_dynamo(backend):
    import torch
    import torchdynamo

    torchdynamo.reset()
    @torchdynamo.optimize(backend, nopython=True)
    def fn(x):
        return x + x

    x = torch.randn(10, 10)
    x.requires_grad = True
    y = fn(x)
    torch.testing.assert_close(y, x + x)

def check_eager():
    try:
        check_dynamo("eager")
    except Exception:
        sys.stderr.write("eager sanity check failed\n")
        raise


# gpu needed?
def check_inductor():
    try:
        check_dynamo("inductor")
    except Exception:
        sys.stderr.write("inductor sanity check failed\n")
        sys.stderr.write(
            "Please check if you installed the correct version of `triton`.\n"
        )
        raise

# need to check for functorch for sufficiently old torch (stable 1.12)

def main():
    check_python()
    check_pip_deps()
    check_cuda()
    check_eager()
    check_inductor()
    print("All required checks passed")

if __name__ == '__main__':
    main()