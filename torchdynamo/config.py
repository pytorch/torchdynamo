import logging
import os
import sys
from os.path import abspath
from os.path import dirname
from types import ModuleType

import torch

try:
    import torch._prims
    import torch._refs

    HAS_REFS_PRIMS = True
except ImportError:
    HAS_REFS_PRIMS = False


class AccessLimitingConfig(ModuleType):
    # log level (levels print what it says + all levels listed below it)
    # DEBUG print full traces <-- lowest level + print tracing of every instruction
    # INFO print compiled functions + graphs
    # WARN print warnings (including graph breaks)
    # ERROR print exceptions (and what user code was being processed when it occurred)
    log_level = logging.WARNING
    # Verbose will print full stack traces on warnings and errors
    verbose = False

    # verify the correctness of optimized backend
    verify_correctness = False

    # need this many ops to create an FX graph
    minimum_call_count = 1

    # turn on/off DCE pass
    dead_code_elimination = True

    # disable (for a function) when cache reaches this size
    cache_size_limit = 64

    # specializing int/float by default
    specialize_int_float = True

    # Assume these functions return constants
    constant_functions = {
        torch.jit.is_scripting: False,
        torch.jit.is_tracing: False,
        torch._C._get_tracing_state: None,
        torch.fx._symbolic_trace.is_fx_tracing: False,
        torch.onnx.is_in_onnx_export: False,
    }

    # root folder of the project
    base_dir = dirname(dirname(abspath(__file__)))

    # don't specialize on shapes and strides and put shape ops in graph
    dynamic_shapes = os.environ.get("TORCHDYNAMO_DYNAMIC_SHAPES") == "1"

    # Set this to False to assume nn.Modules() contents are immutable (similar assumption as freezing)
    guard_nn_modules = False

    # Run the FX graph as it is created to get better type information
    dynamic_propagation = True

    # Run the FX graph with FakeTensors
    fake_tensor_propagation = True

    # run FX normalization passes in optimizer
    normalize_ir = True

    # If a tensor subclass type is in this set, torchdynamo will inline the
    # __torch_function__ logic of the subclass.
    traceable_tensor_subclasses = set()

    # Raise torchdynamo internal assertions
    raise_on_assertion_error = False

    # Propagate backend exceptions up to torchdynamo.optimize
    raise_on_backend_error = True

    # If a PyTorch module is in this allowlist, torchdynamo will be allowed
    # to inline objects from it or its children.
    skipfiles_inline_module_allowlist = {torch.nn, torch.distributions}
    if HAS_REFS_PRIMS:
        skipfiles_inline_module_allowlist |= {
            torch._refs,
            torch._prims,
            torch._decomp,
        }

    # If a string representing a PyTorch module is in this ignorelist,
    # the `allowed_functions.is_allowed` function will not consider it
    # when creating a list of PyTorch functions that will appear in
    # FX IR.
    allowed_functions_module_string_ignorelist = {
        "torch.distributions",
        "torch.testing",
        "torch._refs",
        "torch._prims",
        "torch._decomp",
    }

    # Compiler compilation debug info
    # 0: Nothing printed out when compilation fails
    # 1: Dump the graph out to repro.py if compilation fails
    # 2: Dumps the graph out to minify_repro.py with a minifier if compilation fails
    # 3: Always dumps the last graph ran out to minify_repro.py, useful for segfaults/irrecoverable errors
    repro_level = int(os.environ.get("COMPILER_REPRO_LEVEL", 0))

    # Not all backends support scalars. Some calls on torch.Tensor (like .item()) return a scalar type.
    # When this flag is set to False, we introduce a graph break instead of capturing.
    capture_scalar_outputs = False

    def __setattr__(self, name, value):
        if sys.version_info > (3, 8):
            assert hasattr(
                self, name
            ), f"Trying to set {name} - this value does not exist in torchdynamo.config"
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if sys.version_info > (3, 8):
            assert hasattr(
                self, name
            ), f"Trying to del {name} - this value does not exist in torchdynamo.config"
        object.__delattr__(self, name)


sys.modules[__name__] = AccessLimitingConfig("config")
