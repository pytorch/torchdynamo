import os
import re
import subprocess
import sys
import traceback

try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging
Version = packaging.version.Version

MIN_CUDA_VERSION = Version("11.6")
MIN_PYTHON_VERSION = (3, 7)
# NOTE: dev version required, e.g. Version('1.13.0.dev123') < Version('1.13.0')
MIN_TORCH_VERSION = Version("1.13.0.dev0")


def check_python():
    if sys.version_info < MIN_PYTHON_VERSION:
        raise RuntimeError(
            f"Python version not supported: {sys.version_info} "
            f"- minimum requirement: {MIN_PYTHON_VERSION}"
        )
    return sys.version_info


# Checks for correct pip dependencies, according to `install_requires` in setup.py.
def check_pip_deps():
    proc = subprocess.run(["pip", "check"], capture_output=True)
    if proc.returncode != 0:
        dynamo_broken_deps = []
        for line in proc.stdout.decode("utf-8").strip().split("\n"):
            if line.startswith("torchdynamo"):
                dynamo_broken_deps.append(line)
        if len(dynamo_broken_deps) > 0:
            raise RuntimeError(
                "torchdynamo requirements not met:\n"
                + "\n".join(map(str, dynamo_broken_deps))
            )
    proc = subprocess.run(["pip", "show", "torchdynamo"], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("`torchdynamo` is not installed")


# Checks for correct torch version.
# Using check_pip_deps does not work if the minimum required torch version
# is not present in PyPI.
def check_torch():
    import torch

    if torch.__version__ < MIN_TORCH_VERSION:
        raise RuntimeError(
            f"torch version not supported: {torch.__version__} "
            f"- minimum requirement: {MIN_TORCH_VERSION}"
        )

    return torch.__version__


def check_cuda():
    import torch

    if not torch.cuda.is_available():
        return None

    torch_cuda_ver = Version(torch.version.cuda)
    if torch_cuda_ver < MIN_CUDA_VERSION:
        raise RuntimeError(
            f"CUDA version not supported: {torch_cuda_ver} "
            f"- minimum requirement: {MIN_CUDA_VERSION}"
        )

    # check if torch cuda version matches system cuda version
    cuda_home = os.getenv("CUDA_HOME")
    if not cuda_home:
        raise RuntimeError("$CUDA_HOME not found")

    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    nvcc_version_str = (
        subprocess.check_output([nvcc, "--version"]).strip().decode("utf-8")
    )
    cuda_version = re.search(r"release (\d+[.]\d+)", nvcc_version_str)
    if cuda_version is None:
        raise RuntimeError("CUDA version not found in `nvcc --version` output")

    cuda_ver = Version(cuda_version.group(1))
    if cuda_ver != torch_cuda_ver:
        raise RuntimeError(
            f"CUDA version mismatch, torch version: {torch_cuda_ver}, env version: {cuda_ver}"
        )

    return cuda_ver


def check_dynamo(backend, device, err_msg):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print(f"CUDA not available -- skipping CUDA check on {backend} backend\n")
        return

    try:
        import torchdynamo

        torchdynamo.reset()

        @torchdynamo.optimize(backend, nopython=True)
        def fn(x):
            return x + x

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        mod = Module()
        opt_mod = torchdynamo.optimize(backend, nopython=True)(mod)

        for f in (fn, opt_mod):
            x = torch.randn(10, 10).to(device)
            x.requires_grad = True
            y = f(x)
            torch.testing.assert_close(y, x + x)
            z = y.sum()
            z.backward()
            torch.testing.assert_close(x.grad, 2 * torch.ones_like(x))
    except Exception:
        sys.stderr.write(traceback.format_exc() + "\n" + err_msg + "\n\n")
        sys.exit(1)


_SANITY_CHECK_ARGS = (
    ("eager", "cpu", "CPU eager sanity check failed"),
    ("eager", "cuda", "CUDA eager sanity check failed"),
    ("aot_eager", "cpu", "CPU aot_eager sanity check failed"),
    ("aot_eager", "cuda", "CUDA aot_eager sanity check failed"),
    ("inductor", "cpu", "CPU inductor sanity check failed"),
    (
        "inductor",
        "cuda",
        "CUDA inductor sanity check failed\n"
        + "NOTE: Please check that you installed the correct hash/version of `triton`",
    ),
)


def main():
    python_ver = check_python()
    check_pip_deps()
    torch_ver = check_torch()
    cuda_ver = check_cuda()
    print(
        f"Python version: {python_ver.major}.{python_ver.minor}.{python_ver.micro}\n"
        f"PyTorch version: {torch_ver}\n"
        f"CUDA version: {cuda_ver}"
    )
    for args in _SANITY_CHECK_ARGS:
        check_dynamo(*args)
    print("All required checks passed")


if __name__ == "__main__":
    main()
