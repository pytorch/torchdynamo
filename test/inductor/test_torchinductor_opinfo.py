# flake8: noqa

import atexit
import os
from collections import defaultdict
from enum import Enum
from functools import partial
from unittest.mock import patch

import torch
from torch.testing._internal.common_device_type import OpDTypes
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_device_type import onlyNativeDeviceTypes
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import dtype_abbrs
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_utils import skipCUDAMemoryLeakCheckIf
from torch.testing._internal.common_utils import suppress_warnings

import torchdynamo

from .test_torchinductor import check_model
from .test_torchinductor import check_model_cuda

bf16 = torch.bfloat16  # not tested
f64 = torch.float64
f32 = torch.float32
f16 = torch.float16
i8 = torch.int8  # not tested
i16 = torch.int16  # not tested
i32 = torch.int32
i64 = torch.int64
b8 = torch.bool
u8 = torch.uint8  # not tested

_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=[f16, f32, f64, i32, i64, b8]
)

# Success forces pass; failure forces fail; skip unconditionally skips testing
TestExpect = Enum("TestExpect", ("SUCCESS", "XFAILURE", "SKIP"))

COLLECT_EXPECT = os.getenv("PYTORCH_COLLECT_EXPECT", "0") == "1"
FAIL_ON_SUCCESS = os.getenv("PYTORCH_FAIL_ON_SUCCESS", "1") == "1"
ALL_SAMPLES = os.getenv("PYTORCH_ALL_SAMPLES", "0") == "1"
START = os.getenv("PYTORCH_TEST_RANGE_START", None)
END = os.getenv("PYTORCH_TEST_RANGE_END", None)

if START is not None or END is not None:
    assert END is not None
    assert START is not None
    START = int(START)
    END = int(END)
    assert START < END
else:
    START = 0
    END = len(op_db)

seen_succeeded = defaultdict(dict)
seen_failed = defaultdict(dict)
failed_reasons = defaultdict(set)


def print_seen():
    expected_failures = defaultdict(list)

    def fmt_dtypes(dtypes):
        r = ", ".join(sorted(dtype_abbrs[d] for d in dtypes))
        return "{" + r + "}"

    def process(device_type):
        for op, failed_dtypes in seen_failed[device_type].items():
            succeeded_dtypes = seen_succeeded.get(op, set())
            expected_failures_dtypes = failed_dtypes - succeeded_dtypes

            reasons = ""
            if failed_reasons[op]:
                reasons = "  # " + ", ".join(sorted(failed_reasons[op]))
            if expected_failures_dtypes:
                expected_failures[device_type].append(
                    f'   "{op}": {fmt_dtypes(expected_failures_dtypes)},{reasons}'
                )

        expected_failures[device_type].sort()
        nl = "\n"
        print(
            f"""
inductor_expected_failures_single_sample[\"{device_type}\"] = {{
{nl.join(expected_failures[device_type])}
}}
"""
        )

    process("cpu")
    process("cuda")


if COLLECT_EXPECT:
    atexit.register(print_seen)

inductor_skips = defaultdict(dict)

# fmt: off

inductor_skips["cpu"] = {
    "linalg.ldl_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "linalg.lu_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "reciprocal": {b8, i32, i64},  # segfault
    "lu_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "lu_unpack": {b8, f16, f32, f64, i32, i64},  # segfault
    "__rdiv__": {b8, f16, f32, f64, i32, i64},  # flaky
}

inductor_skips["cuda"] = {
    # flaky
    "__rdiv__": {b8, f16, f32, f64, i32, i64},
    "masked.prod": {f16, f32, f64},
    "linalg.vander": {f32, f64},
    "sparse.sampled_addmm": {f32, f64},
    "broadcast_tensors": {f16, f32, f64},
    # Call parameter type does not match function signature!
    "masked.logsumexp": {f64},
    "erf": {f64},
    "logsumexp": {f64},
    "lu_unpack": {f32, f64},  # RuntimeError: CUDA error
    "nn.functional.binary_cross_entropy_with_logits": {f64},
    "nn.functional.gelu": {f64},
    "nn.functional.glu": {f64},
    "nn.functional.poisson_nll_loss": {f64},
    "nn.functional.selu": {f64},
    "nn.functional.silu": {f64},
    "nn.functional.tanhshrink": {f16, f64},
    "nn.functional.conv_transpose3d": {f16, f64},
    "nn.functional._scaled_dot_product_attention": {f64},
    "nn.functional.softmin.with_dtype": {b8, f16, f32, f64, i32, i64},
    "nn.functional.pixel_shuffle": {b8, f16, f32, f64, i32, i64},
    "nn.functional.pixel_unshuffle": {b8, f16, f32, f64, i32, i64},
    "nn.functional.triplet_margin_loss": {f16},
    "special.ndtr": {f64},
    "log_softmax.dtype": {b8, f16, f32, f64, i32, i64},  # segfault
    # Jiterator kernel is not expected to work with inductor
    "jiterator_2inputs_2outputs": {b8, f16, f32, f64, i32, i64},
    "jiterator_4inputs_with_extra_args": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary_return_by_ref": {b8, f16, f32, f64, i32, i64},
    "jiterator_unary": {b8, f16, f32, f64, i32, i64},
    # failed since moving PyTorch pin https://github.com/pytorch/torchdynamo/pull/1453
    "dsplit": {f16, f32, f64},
    "hsplit": {f16, f32, f64},
}


inductor_expected_failures_single_sample = defaultdict(dict)

inductor_expected_failures_single_sample["cpu"] = {
    "H": {b8, f16, f32, f64, i32, i64}, # Dynamo produces no graph - Ran graph without calling compile_fx
    "T": {b8, f16, f32, f64, i32, i64}, # Dynamo produces no graph - Ran graph without calling compile_fx
    "mH": {b8, f16, f32, f64, i32, i64}, # Dynamo produces no graph - Ran graph without calling compile_fx
    "mT": {b8, f16, f32, f64, i32, i64}, # Dynamo produces no graph - Ran graph without calling compile_fx
    "__getitem__": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: call_function UserDefinedObjectVariable(wrapper_descriptor)
    "addr": {f16}, # Correctness - AssertionError: Tensor-likes are not close!
    "allclose": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator: aten.allclose.default
    "angle": {f16, f32, f64}, # Unclear
    "argwhere": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.nonzero.default
    "bernoulli": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "bincount": {i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.bincount.default
    "chalf": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "cholesky": {f32, f64}, # Inductor Exception - torchdynamo.exc.TorchRuntimeError
    "combinations": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.masked_select.default
    "complex": {f16, f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "constant_pad_nd": {f16, f32, f64}, # Correctness - AssertionError: Tensor-likes are not close!
    "copysign": {f16}, # Inductor Exception - C++ compile error
    "corrcoef": {f32, f64, i32, i64}, # torchdynamo.exc.Unsupported: data dependent operator: aten.equal.default
    "cov": {f32, f64, i32, i64}, # torchdynamo.exc.Unsupported: data dependent operator: aten.equal.default
    "equal": {b8, f16, f32, f64, i32, i64}, # torchdynamo.exc.Unsupported: data dependent operator: aten.equal.default
    "erf": {b8, f64}, # Capability - RuntimeError: Negation, the `-` operator, on a bool tensor is not supported.
    "fft.fft": {f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.fft2": {b8, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.fftn": {b8, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.hfft": {b8, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128 
    "fft.hfft2": {b8, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.hfftn": {b8, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ifft": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ifft2": {b8, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ifftn": {b8, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ihfft": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ihfft2": {f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ihfftn": {f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.irfft": {b8, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.irfft2": {b8, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.irfftn": {b8, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.rfft": {f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.rfft2": {f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "index_add": {f16}, # Inductor Exception - C++ compile error
    "fft.rfftn": {f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "index_put": {f16, f32, f64}, # Fake Tensor Access - torch._subclasses.fake_tensor.DynamicOutputShapeException: aten.index.Tensor
    "index_reduce": {f16, f32, f64}, # Fake Tensor Bug - RuntimeError: It appears that you're trying to get value out of a tracing tensor
    "istft": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator: aten.equal.default
    "linalg.cholesky": {f32, f64}, # Correctness - AssertionError: Tensor-likes are not close!
    "linalg.cholesky_ex": {f32, f64}, # Correctness - AssertionError: Tensor-likes are not close!
    "linalg.eig": {f32, f64}, # Fake Tensor Bug - RuntimeError: It appears that you're trying to get value out of a tracing tensor
    "linalg.eigh": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function args: TensorVariable()
    "linalg.eigvals": {f32, f64},  # Complex Type - torch.complex32 / 64 / 128
    "linalg.eigvalsh": {f32, f64},  # Graph Break - torchdynamo.exc.Unsupported: call_function args: TensorVariable()
    "linalg.ldl_factor": {f32, f64}, # Inductor Exception - torchinductor.exc.LoweringException:
    "linalg.lstsq": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.linalg_lstsq.default
    "linalg.lstsq.grad_oriented": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.linalg_lstsq.default
    "linalg.matrix_rank": {f32, f64}, # Inductor Exception - torchinductor.exc.LoweringException:
    "linalg.matrix_rank.hermitian": {f32, f64}, # Inductor Exception - torchinductor.exc.LoweringException:
    "linalg.svd": {f32, f64}, # Inductor Exception - ASsertionError
    "logdet": {f32, f64}, # Inductor Exception - C++ compile error
    "masked.norm": {f16}, # Inductor Exception - C++ compile error
    "masked_fill": {f16}, # Inductor Exception - C++ compile error
    "masked_scatter": {f16, f32, f64}, # Fake Tensor Access - torch._subclasses.fake_tensor.DynamicOutputShapeException: aten.index.Tensor
    "masked_select": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.masked_select.default
    "max.reduction_no_dim": {f16}, # Inductor Exception - fp16 user-defined reduction
    "max.reduction_with_dim": {b8, f16}, # Inductor Exception - fp16 user-defined reduction
    "min.reduction_no_dim": {f16}, # Inductor Exception - fp16 user-defined reduction
    "min.reduction_with_dim": {b8, f16}, 
    "multinomial": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "nan_to_num": {f16}, # Inductor Exception - C++ compile error
    "nanquantile": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator: aten.equal.default
    "nn.functional.avg_pool1d": {i64}, # Correctness -  AssertionError: Tensor-likes are not equal!
    "nn.functional.avg_pool2d": {i64},  # Correctness -  AssertionError: Tensor-likes are not equal!
    "nn.functional.adaptive_avg_pool2d": {f16}, # Unclear
    "nn.functional.ctc_loss": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten._ctc_loss.Tensor
    "nn.functional.gaussian_nll_loss": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator
    "nn.functional.gelu": {f64},  # Correctness -  AssertionError: Tensor-likes are not equal!
    "nn.functional.huber_loss": {f16, f32, f64}, # Unclear
    "nn.functional.local_response_norm": {i64},  # Correctness -  AssertionError: Tensor-likes are not equal!
    "nn.functional.one_hot": {i64}, # Unclear
    "nn.functional.pairwise_distance": {f16},  # Inductor Exception - C++ compile error
    "nn.functional.rrelu": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "nn.functional.triplet_margin_with_distance_loss": {f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "nonzero": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shapes: nonzero
    "normal": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "normal.number_mean": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "pca_lowrank": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "pinverse": {f32, f64},  # Inductor Exception - torchinductor.exc.LoweringException: AssertionError
    "polar": {f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "quantile": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator: aten.equal.default
    "rand_like": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "randint_like": {f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "randn_like": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "repeat_interleave": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.repeat_interleave.Tensor
    "scatter_add": {f16}, # Inductor Exception - C++ compile error
    "scatter_reduce.prod": {f16, f32, f64},
    "scatter_reduce.sum": {f16},  # Inductor Exception - C++ compile error
    "segment_reduce.lengths": {f16, f32, f64}, # Inductor Exception - torchinductor.exc.LoweringException: AssertionError
    "segment_reduce.offsets": {f16, f32, f64}, # Inductor Exception - torchinductor.exc.LoweringException: AssertionError
    "sgn": {f16, f32, f64}, # Inductor Exception - NotImplementedError (Potentially due to complex)
    "sparse.sampled_addmm": {f32, f64}, # Unclear
    "stft": {f32, f64}, # Inductor Exception - NotImplementedError (Potentially due to complex)
    "svd": {f32, f64}, # Inductor Exception - torchinductor.exc.LoweringException: AssertionError
    "svd_lowrank": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "tensor_split": {b8, f16, f32, f64, i32, i64}, # Inductor Exception - torchinductor.exc.LoweringException: AssertionError
    "to": {b8, f16, f32, f64, i32, i64}, # Inductor Exception - torchinductor.exc.LoweringException: AssertionError
    "to_sparse": {f32, f64}, # Fake Tensor Access - Exception: Invoking operators with non-Fake Tensor inputs in FakeTensorMode 
    "tril": {f16},  # Inductor Exception - C++ compile error
    "triu": {f16},  # Inductor Exception - C++ compile error
    "uniform": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "unique": {b8, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shapes: unique
    "unique_consecutive": {b8, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shapes: unique_consecutive
    "var": {f16}, # Correctness - AssertionError: Scalars Are not close!
    "var_mean": {f16}, # Correctness - AssertionError: Scalars Are not close!
    "view_as_complex": {f16, f32, f64}, # Complex Type - torch.complex32 / 64 / 128
}


inductor_expected_failures_single_sample["cuda"] = {
    "H": {b8, f16, f32, f64, i32, i64}, # Dynamo produces no graph - Ran graph without calling compile_fx
    "T": {b8, f16, f32, f64, i32, i64}, # Dynamo produces no graph - Ran graph without calling compile_fx
    "__getitem__": {b8, f16, f32, f64, i32, i64}, #  Graph Break - torchdynamo.exc.Unsupported: call_function UserDefinedObjectVariable(wrapper_descriptor)
    "allclose": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator: aten.allclose.default
    "angle": {f32, f64}, # Unclear
    "argwhere": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.nonzero.default
    "baddbmm": {f16}, # Correctness - AssertionError: Tensor-likes are not close!
    "bernoulli": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "bincount": {i32, i64}, # Graph Break Dyn shape - torchdynamo.exc.Unsupported: dynamic shape operator: aten.bincount.default
    "chalf": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32
    "cholesky": {f32, f64}, # Correctness - AssertionError: Tensor-likes are not close!
    "combinations": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.masked_select.default
    "complex": {f16, f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "corrcoef": {f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator: aten.equal.default
    "cov": {f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator: aten.equal.default
    "equal": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator: aten.equal.default
    "erf": {b8}, # Capability - RuntimeError: Negation, the `-` operator, on a bool tensor is not supported.
    "fft.fft": {f16, f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.fft2": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.fftn": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.hfft": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.hfft2": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.hfftn": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ifft": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ifft2": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ifftn": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ihfft": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ihfft2": {f16, f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.ihfftn": {f16, f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.irfft": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.irfft2": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.irfftn": {b8, f16, f32, f64, i32, i64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.rfft": {f16, f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.rfft2": {f16, f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "fft.rfftn": {f16, f32, f64}, # Complex Type - torch.complex32 / 64 / 128
    "index_put": {f16, f32, f64}, # Fake Tensor Access - torch._subclasses.fake_tensor.DynamicOutputShapeException: aten.index.Tensor
    "index_reduce": {f16, f32, f64}, # Fake Tensor Bug - RuntimeError: It appears that you're trying to get value out of a tracing tensor wit
    "istft": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator: aten.equal.default
    "linalg.cholesky": {f32, f64}, # Correctness - AssertionError: Tensor-likes are not close!
    "linalg.cholesky_ex": {f32, f64}, # Correctness - AssertionError: Tensor-likes are not close!
    "linalg.eig": {f32, f64}, # Unclear
    "linalg.eigh": {f32, f64}, # Unclear
    "linalg.eigvals": {f32, f64}, # Unclear
    "linalg.eigvalsh": {f32, f64}, # Unclear
    "linalg.ldl_factor": {f32, f64}, # Inductor Exception - torchinductor.exc.LoweringException: AttributeError
    "linalg.lstsq": {f32, f64}, # Unclear
    "linalg.lstsq.grad_oriented": {f32, f64}, # Unclear
    "linalg.matrix_rank": {f32, f64},  # Inductor Exception - torchinductor.exc.LoweringException: AssertionError
    "linalg.matrix_rank.hermitian": {f32, f64}, # Inductor Exception - torchinductor.exc.LoweringException: _LinAlgError:
    "linalg.pinv.hermitian": {f32, f64}, # Inductor Exception - torchinductor.exc.LoweringException: _LinAlgError:
    "linalg.svd": {f32, f64}, # Unclear - AssertionError
    "mH": {b8, f16, f32, f64, i32, i64}, # Dynamo produces no graph - Ran graph without calling compile_fx
    "mT": {b8, f16, f32, f64, i32, i64}, # Dynamo produces no graph - Ran graph without calling compile_fx
    "masked.argmax": {f16, f32, f64, i32}, # Unclear
    "masked.argmin": {f16, f32, f64, i32}, # Unclear
    "masked_scatter": {f16, f32, f64}, # Fake Tensor Access - torch._subclasses.fake_tensor.DynamicOutputShapeException
    "masked_select": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.masked_select.default
    "max.reduction_with_dim": {b8, i32, i64}, # Correctness - Tensor-likes are not Equal! / Bool is dtype mismatch.
    "min.reduction_with_dim": {b8, i32, i64},  # Correctness - Tensor-likes are not Equal! / Bool is dtype mismatch.
    "multinomial": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "nn.functional.adaptive_avg_pool2d": {f16}, # Correctness - Tensor-likes are not close!
    "nn.functional._scaled_dot_product_attention": {f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "nn.functional.conv_transpose3d": {f16}, # Unclear
    "nn.functional.ctc_loss": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten._ctc_loss.Tensor
    "nn.functional.grid_sample": {f16}, # Correctness - Tensor-likes are not close!
    "nn.functional.gaussian_nll_loss": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator
    "nn.functional.huber_loss": {f16, f32, f64}, # Unclear
    "nn.functional.one_hot": {i64}, # Graph Break - torchdynamo.exc.Unsupported: data dependent operator
    "nn.functional.pairwise_distance": {f16, f32, f64},  # Inductor Exception - C++ compile error
    "nn.functional.rrelu": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "nn.functional.triplet_margin_with_distance_loss": {f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "nonzero": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shapes: nonzero
    "normal": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "normal.number_mean": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "pca_lowrank": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "pinverse": {f32, f64},  # Inductor Exception - torchinductor.exc.LoweringException: AssertionError
    "polar": {f32, f64}, # Inductor Exception - NotImplementedError (Potentially due to complex)
    "rand_like": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "randint_like": {f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "randn_like": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "repeat_interleave": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shape operator: aten.repeat_interleave.Tensor
    "round.decimals_3": {f16}, # Correctness - Tensor-likes are not close!
    "scatter_reduce.prod": {f16, f32, f64},
    "segment_reduce.lengths": {f16, f32, f64}, # Inductor Exception - torchdynamo.exc.TorchRuntimeError
    "segment_reduce.offsets": {f16, f32, f64}, # Inductor Exception - torchinductor.exc.LoweringException: AssertionError
    "sgn": {f16, f32, f64}, # Inductor Exception - NotImplementedError
    "stft": {f32, f64}, # Inductor Exception - NotImplementedError (Potentially due to complex)
    "svd": {f32, f64}, # Inductor Exception - AssertionError
    "svd_lowrank": {f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "tensor_split": {b8, f16, f32, f64, i32, i64}, # Inductor Exception - torchdynamo.exc.TorchRuntimeError
    "to": {b8, f16, f32, f64, i32, i64}, # Inductor Exception - AssertionError (TODO)
    "to_sparse": {f16, f32, f64}, # Fake Tensor Access - Exception: Invoking operators with non-Fake Tensor inputs in FakeTensorMode 
    "uniform": {f16, f32, f64}, # Graph Break - torchdynamo.exc.Unsupported: call_function in skip_files 
    "unique": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shapes: unique
    "unique_consecutive": {b8, f16, f32, f64, i32, i64}, # Graph Break - torchdynamo.exc.Unsupported: dynamic shapes: unique_consecutive
    "view_as_complex": {f16, f32, f64}, # Inductor Exception - NotImplementedError (Potentially due to complex)
}

inductor_should_fail_with_exception = defaultdict(dict)

inductor_should_fail_with_exception["cpu"] = {}


inductor_should_fail_with_exception["cuda"] = {
    "__rpow__": {
        i32: "Pow input must be floating point.",
        i64: "Pow input must be floating point.",
    },
    "pow": {
        i32: "Pow input must be floating point.",
        i64: "Pow input must be floating point.",
    }
}


def wrapper_set_seed(op, *args, **kwargs):
    """Wrapper to set seed manually for some functions like dropout
    See: https://github.com/pytorch/pytorch/pull/62315#issuecomment-896143189 for more details.
    """
    torch.manual_seed(42)
    return op(*args, **kwargs)


torch.testing._internal.common_methods_invocations.wrapper_set_seed = wrapper_set_seed

# This file does a global patch to `disable_global_flags()` - which we should not invoke in non testing cases.
torchdynamo.variables.torch.tensor_dunder_fns.append(
    torch.testing._internal.common_utils.disable_functorch
)

# key can be either op_name, or (op_name, deivce_type), or (op_name, device_type, dtype)
inductor_override_kwargs = {
    # the return value of empty is undefined
    "empty": {"assert_equal": False},
    "empty_like": {"assert_equal": False},
    "new_empty": {"assert_equal": False},
    "new_empty_strided": {"assert_equal": False},
    "randn": {"assert_equal": False},
    ("nn.functional.tanhshrink", "cuda", f16): {"atol": 3e-4, "rtol": 0.001},
}

# Always test with all sample for following ops
inductor_all_samples = {
    "softmax.with_dtype",
    "index_add",
    "index_put",
    "index_copy",
    "scatter_reduce.sum",
}

# fmt: on


class TestInductorOpInfo(TestCase):
    check_model = check_model
    check_model_cuda = check_model_cuda

    @onlyNativeDeviceTypes
    @suppress_warnings
    @skipCUDAMemoryLeakCheckIf(
        True
    )  # inductor kernels failing this test intermittently
    @_ops(op_db[START:END])
    @patch("torchdynamo.config.raise_on_unsafe_aot_autograd", True)
    def test_comprehensive(self, device, dtype, op):
        torchdynamo.reset()
        with torch.no_grad():
            torch.cuda.empty_cache()
        op_name = op.name
        if op.variant_test_name:
            op_name += f".{op.variant_test_name}"

        device_type = torch.device(device).type

        assert device_type in ("cuda", "cpu")

        # with open("test_output.txt", "a") as f:
        #     print(f"CONSIDERING OP {op_name} on {device_type} with {dtype} |
        # {inductor_skips[device_type].get(op_name, set())}", flush=True, file=f)
        #     print(f"CONSIDERING OP {op_name} on {device_type} with {dtype} |
        # {inductor_skips[device_type].get(op_name, set())}", flush=True)
        if dtype in inductor_skips[device_type].get(op_name, set()):
            test_expect = TestExpect.SKIP
            # with open("test_output.txt", "a") as f:
            #     print(f"SKIPPING OP {op_name} on {device_type}", flush=True, file=f)
            #     print(f"SKIPPING OP {op_name} on {device_type}", flush=True)
            self.skipTest(f"{op_name} in {dtype} not supported")
        elif dtype in inductor_expected_failures_single_sample[device_type].get(
            op_name, set()
        ):
            test_expect = TestExpect.XFAILURE
        else:
            test_expect = TestExpect.SUCCESS

        additional_kwargs = {}
        if op_name in inductor_override_kwargs:
            additional_kwargs = inductor_override_kwargs[op_name]
        elif (op_name, device_type) in inductor_override_kwargs:
            additional_kwargs = inductor_override_kwargs[(op_name, device_type)]
        elif (op_name, device_type, dtype) in inductor_override_kwargs:
            additional_kwargs = inductor_override_kwargs[(op_name, device_type, dtype)]

        func = op.get_op()

        def fn(*args, **kwargs):
            return func(*args, **kwargs)

        requires_grad = (
            op.supports_autograd
            and dtype in op.supported_backward_dtypes(device_type)
            # TODO: OpInfo really ought to error out for this case, but it's
            # not exercised in test_ops_gradients atm.  The problem is not
            # complex32 per-se (which is supported by data movement only ops)
            # but that when we do backwards we expect other ops like add to work
            and not dtype == torch.complex32
        )
        samples = op.sample_inputs(device, dtype, requires_grad=requires_grad)

        if op_name not in inductor_all_samples and not ALL_SAMPLES:
            if isinstance(samples, (list, tuple)):
                samples = [samples[0]]
            else:
                samples = [next(samples)]

        try:
            for sample_input in samples:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                # UNCOMMENT TO DEBUG SEGFAULTS
                # with open("test_output.txt", "a") as f:
                #     print(f"RUNNING OP {op_name} on {device_type} with {dtype}", flush=True, file=f)
                #     print(f"RUNNING OP {op_name} on {device_type} with {dtype}", flush=True)
                if device_type == "cuda":
                    # opinfo test case have already place the input on the correct device
                    # so we don't need do additional copy by setting copy_to_cuda=False
                    self.check_model_cuda(
                        fn,
                        args,
                        kwargs,
                        check_lowp=False,
                        nopython=True,
                        copy_to_cuda=False,
                        reference_in_float=False,
                        **additional_kwargs,
                    )
                elif device_type == "cpu":
                    self.check_model(
                        fn,
                        args,
                        kwargs,
                        check_lowp=False,
                        nopython=True,
                        **additional_kwargs,
                    )

        except Exception as e:

            if test_expect is TestExpect.XFAILURE:
                return

            seen_failed[device_type].setdefault(op_name, set()).add(dtype)

            if COLLECT_EXPECT:
                return

            known_failure = False
            if dtype in inductor_should_fail_with_exception[device_type].get(
                op_name, set()
            ):
                failure = inductor_should_fail_with_exception[device_type][op_name][
                    dtype
                ]
                if failure in str(e):
                    known_failure = True
            if not known_failure:
                raise e

        # with open("test_output.txt", "a") as f:
        #     print(f"SUCCEEDED OP {op_name} on {device_type} with {dtype}", flush=True, file=f)
        seen_succeeded[device_type].setdefault(op_name, set()).add(dtype)

        if test_expect is TestExpect.XFAILURE and not COLLECT_EXPECT:
            if FAIL_ON_SUCCESS:
                raise RuntimeError(
                    f"unexpected success {op_name}, {dtype}, {device_type}"
                )


instantiate_device_type_tests(TestInductorOpInfo, globals())

if __name__ == "__main__":
    torchdynamo.config.raise_on_assertion_error = True
    run_tests()
