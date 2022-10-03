import sys
import warnings

MIN_PYTHON_VERSION = (3, 7)
MIN_TORCH_VERSION = (1, 12)
MIN_CUDA_VERSION = (11, 6)

def check_python():
    if sys.version_info < MIN_PYTHON_VERSION:
        warnings.warn(
            f"Python version not supported: {sys.version_info} "
            f"- minimum requirement: {MIN_PYTHON_VERSION}"
        )

def check_pip_deps():
    pass

def check_torch():
    try:
        import torch
    except ImportError:
        warnings.warn("`torch` not installed")
        return False
    
    return True

def check_cuda():
    import torch
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available for torch")
        return False
    cuda_version = tuple(map(int, torch.cuda.version.split('.')[:2]))
    if cuda_version < MIN_CUDA_VERSION:
        warnings.warn(
            f"CUDA version not supported: {cuda_version} "
            f"- minimum requirement: {MIN_CUDA_VERSION}"
        )
        return False
    return True

def check_triton():
    pass

def check_dynamo():
    pass

def check_inductor():
    pass


def main():
    for fn in (
        check_python,
        check_torch,
        check_cuda,
        check_triton,
        check_dynamo,
        check_inductor,
    ):
        if not fn():
            print("Check failed")
            return
    print("All checks passed")

if __name__ == '__main__':
    main()