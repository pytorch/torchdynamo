import atexit
import os
from collections import defaultdict
from enum import Enum
from functools import partial

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

inductor_skips["cpu"] = {
    # the return value of empty is undefined
    "empty": {b8, f16, f32, f64, i32, i64},
    "empty_like": {b8, f16, f32, f64, i32, i64},
    "new_empty": {b8, f16, f32, f64, i32, i64},
    "new_empty_strided": {b8, f16, f32, f64, i32, i64},
    "linalg.ldl_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "linalg.lu_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "lu_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "lu_unpack": {b8, f16, f32, f64, i32, i64},  # segfault
    "__rdiv__": {b8, f16, f32, f64, i32, i64},  # flaky
}

inductor_skips["cuda"] = {
    # flaky
    "__rdiv__": {b8, f16, f32, f64, i32, i64},
    "mvlgamma.mvlgamma_p_1": {f16, f32, f64, i32, i64},
    "mvlgamma.mvlgamma_p_3": {f16, f32, f64, i32, i64},
    "mvlgamma.mvlgamma_p_5": {f16, f32, f64, i32, i64},
    "cumprod": {f16, f32, f64},
    "masked.prod": {f16, f32, f64},
    "empty_like": {b8, f16, f32, f64, i32, i64},
    "empty": {b8, f16, f32, f64, i32, i64},
    "reciprocal": {b8},
    "linalg.vander": {f32, f64},
    "sparse.sampled_addmm": {f32, f64},
    "nn.functional.conv_transpose1d": {f16},
    "nn.functional.conv_transpose2d": {f16},
    # Call parameter type does not match function signature!
    "masked.logsumexp": {f64},
    "cos": {b8, f64, i32, i64},
    "erf": {f64, i32, i64},
    "exp": {b8, f64, i32, i64},
    "log": {b8, i32, i64},
    "log1p": {b8, i32, i64},
    "log2": {b8, i32, i64},
    "logsumexp": {f64},
    "lu_unpack": {f32, f64},  # RuntimeError: CUDA error
    "nn.functional.binary_cross_entropy": {f64},
    "nn.functional.binary_cross_entropy_with_logits": {f64},
    "nn.functional.cross_entropy": {f64},
    "nn.functional.elu": {f64},
    "nn.functional.gelu": {f64},
    "nn.functional.glu": {f64},
    "nn.functional.poisson_nll_loss": {f64, i32, i64},
    "nn.functional.selu": {f64},
    "nn.functional.silu": {f64},
    "nn.functional.tanhshrink": {f64},
    "nn.functional.softmin.with_dtype": {b8, f16, f32, f64, i32, i64},
    "nn.functional.pixel_shuffle": {b8, f16, f32, f64, i32, i64},
    "nn.functional.pixel_unshuffle": {b8, f16, f32, f64, i32, i64},
    "nn.functional.softmin": {b8, f16, f32, f64, i32, i64},  # segfault
    "nn.functional.softmax": {b8, f16, f32, f64, i32, i64},  # segfault
    "rsqrt": {b8, i32, i64},
    "sigmoid": {b8, f64, i32, i64},
    "sin": {b8, f64, i32, i64},
    "special.log_ndtr": {f64},
    "special.ndtr": {f64},
    "sqrt": {b8, i32, i64},
    "tanh": {f64},
    "nn.functional.embedding_bag": {b8, f16, f32, f64, i32, i64},  # segfault
    "masked.log_softmax": {b8, f16, f32, f64, i32, i64},  # segfault
    "masked.logaddexp": {b8, f16, f32, f64, i32, i64},  # segfault
    "masked.softmax": {b8, f16, f32, f64, i32, i64},  # segfault
    "masked.softmin": {b8, f16, f32, f64, i32, i64},  # segfault
    "scatter_add": {b8, f16, f32, f64, i32, i64},  # segfault
    "scatter_reduce": {b8, f16, f32, f64, i32, i64},  # segfault
    "scatter_reduce.amax": {f16, f32, f64, i32, i64},  # segfault
    "scatter_reduce.amin": {f16, f32, f64, i32, i64},  # segfault
    "scatter_reduce.mean": {f16, f32, f64, i32, i64},  # segfault
    "scatter_reduce.prod": {f16, f32, f64, i32, i64},  # segfault
    "scatter_reduce.sum": {b8, f16, f32, f64, i32, i64},  # segfault
    "softmax": {b8, f64, i32, i64},  # segfault
    "softmax.with_dtype": {b8, f16, f32, f64, i32, i64},  # segfault
    "nn.functional.kl_div": {b8, f16, f32, f64, i32, i64},  # segfault
    "log_softmax": {f64},  # segfault
    "log_softmax.dtype": {b8, f16, f32, f64, i32, i64},  # segfault
    # Jiterator kernel is not expected to work with inductor
    "jiterator_2inputs_2outputs": {b8, f16, f32, f64, i32, i64},
    "jiterator_4inputs_with_extra_args": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary_return_by_ref": {b8, f16, f32, f64, i32, i64},
    "jiterator_unary": {b8, f16, f32, f64, i32, i64},
}


inductor_expected_failures_single_sample = defaultdict(dict)

inductor_expected_failures_single_sample["cpu"] = {
    "H": {b8, f16, f32, f64, i32, i64},
    "T": {b8, f16, f32, f64, i32, i64},
    "mH": {b8, f16, f32, f64, i32, i64},
    "mT": {b8, f16, f32, f64, i32, i64},
    "__getitem__": {b8, f16, f32, f64, i32, i64},
    "__radd__": {b8, f16, f32, f64, i32, i64},
    "__rand__": {b8, i32, i64},
    "__rmod__": {f16, f32, f64},
    "__rmul__": {b8, f16, f32, f64, i32, i64},
    "__ror__": {b8, i32, i64},
    "__rpow__": {f16, f32, f64, i32, i64},
    "__rsub__": {f16, f32, f64, i32, i64},
    "__rxor__": {b8, i32, i64},
    "addmm": {f32, f64, i32, i64},
    "addr": {f16},
    "allclose": {f16, f32, f64},
    "angle": {f16, f32, f64},
    "argwhere": {b8, f16, f32, f64, i32, i64},
    "bernoulli": {f32, f64},
    "bincount": {i32, i64},
    "chalf": {b8, f16, f32, f64, i32, i64},
    "cholesky": {f32, f64},
    "combinations": {b8, f16, f32, f64, i32, i64},
    "complex": {f16, f32, f64},
    "constant_pad_nd": {f16, f32, f64},
    "copysign": {f16},
    "corrcoef": {f32, f64, i32, i64},
    "cov": {f32, f64, i32, i64},
    "equal": {b8, f16, f32, f64, i32, i64},
    "erf": {b8, f64},
    "fft.fft": {f32, f64},
    "fft.fft2": {b8, f32, f64, i32, i64},
    "fft.fftn": {b8, f32, f64, i32, i64},
    "fft.hfft": {b8, f32, f64, i32, i64},
    "fft.hfft2": {b8, f32, f64, i32, i64},
    "fft.hfftn": {b8, f32, f64, i32, i64},
    "fft.ifft": {b8, f16, f32, f64, i32, i64},
    "fft.ifft2": {b8, f32, f64, i32, i64},
    "fft.ifftn": {b8, f32, f64, i32, i64},
    "fft.ihfft": {b8, f16, f32, f64, i32, i64},
    "fft.ihfft2": {f32, f64},
    "fft.ihfftn": {f32, f64},
    "fft.irfft": {b8, f32, f64, i32, i64},
    "fft.irfft2": {b8, f32, f64, i32, i64},
    "fft.irfftn": {b8, f32, f64, i32, i64},
    "fft.rfft": {f32, f64},
    "fft.rfft2": {f32, f64},
    "fft.rfftn": {f32, f64},
    "index_add": {b8, f16, f32, f64, i32, i64},
    "index_copy": {f16, f32, f64},
    "index_reduce": {f16, f32, f64},
    "inner": {f32, f64, i32, i64},
    "istft": {f32, f64},
    "linalg.cholesky": {f32, f64},
    "linalg.cholesky_ex": {f32, f64},
    "linalg.eig": {f32, f64},
    "linalg.eigh": {f32, f64},
    "linalg.eigvals": {f32, f64},
    "linalg.eigvalsh": {f32, f64},
    "linalg.ldl_factor": {f32, f64},
    "linalg.lstsq": {f32, f64},
    "linalg.lstsq.grad_oriented": {f32, f64},
    "linalg.matrix_rank": {f32, f64},
    "linalg.matrix_rank.hermitian": {f32, f64},
    "linalg.svd": {f32, f64},
    "logdet": {f32, f64},
    "logsumexp": {b8, f32, f64, i32, i64},
    "masked.norm": {f16},
    "masked_fill": {f16},
    "masked_scatter": {f16, f32, f64},
    "masked_select": {b8, f16, f32, f64, i32, i64},
    "max.reduction_no_dim": {f16},
    "max.reduction_with_dim": {b8, f16},
    "min.reduction_no_dim": {f16},
    "min.reduction_with_dim": {b8, f16},
    "multinomial": {f32, f64},
    "mvlgamma.mvlgamma_p_1": {f32, f64},
    "mvlgamma.mvlgamma_p_3": {f32, f64},
    "mvlgamma.mvlgamma_p_5": {f32, f64},
    "nan_to_num": {b8, f16, i32, i64},
    "nanquantile": {f32, f64},
    "nn.functional._scaled_dot_product_attention": {f32, f64},
    "nn.functional.avg_pool1d": {i64},
    "nn.functional.avg_pool2d": {i64},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.dropout": {f32, f64},
    "nn.functional.dropout2d": {f32, f64},
    "nn.functional.dropout3d": {f32, f64},
    "nn.functional.embedding_bag": {f16},
    "nn.functional.feature_alpha_dropout.with_train": {f32, f64},
    "nn.functional.feature_alpha_dropout.without_train": {b8, f16, f32, f64, i32, i64},
    "nn.functional.fractional_max_pool2d": {f32, f64},
    "nn.functional.fractional_max_pool3d": {f32, f64},
    "nn.functional.gaussian_nll_loss": {f32, f64},
    "nn.functional.gelu": {f64},
    "nn.functional.huber_loss": {f16, f32, f64},
    "nn.functional.local_response_norm": {i64},
    "nn.functional.one_hot": {i64},
    "nn.functional.pairwise_distance": {f16},
    "nn.functional.rrelu": {f32, f64},
    "nn.functional.triplet_margin_with_distance_loss": {f32, f64, i32, i64},
    "nonzero": {b8, f16, f32, f64, i32, i64},
    "normal": {f16, f32, f64},
    "normal.number_mean": {f16, f32, f64},
    "pca_lowrank": {f32, f64},
    "pinverse": {f32, f64},
    "polar": {f32, f64},
    "quantile": {f32, f64},
    "rand_like": {f16, f32, f64},
    "randint_like": {f16, f32, f64, i32, i64},
    "randn": {f16, f32, f64},
    "randn_like": {f16, f32, f64},
    "reciprocal": {b8, i32, i64},
    "repeat_interleave": {b8, f16, f32, f64, i32, i64},
    "scatter_add": {f16},
    "scatter_reduce.amax": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.amin": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.mean": {f16, f32, f64, i32, i64},
    "scatter_reduce.prod": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.sum": {f16},
    "segment_reduce.lengths": {f16, f32, f64},
    "segment_reduce.offsets": {f16, f32, f64},
    "sgn": {b8, f16, f32, f64, i32, i64},
    "sign": {i32, i64},
    "sparse.sampled_addmm": {f32, f64},
    "std_mean": {f16, f32, f64},
    "stft": {f32, f64},
    "svd": {f32, f64},
    "svd_lowrank": {f32, f64},
    "tensor_split": {b8, f16, f32, f64, i32, i64},
    "tensordot": {f32, f64, i32, i64},
    "to": {b8, f16, f32, f64, i32, i64},
    "to_sparse": {f32, f64},
    "tril": {f16},
    "triu": {f16},
    "uniform": {f16, f32, f64},
    "unique": {b8, f32, f64, i32, i64},
    "unique_consecutive": {b8, f32, f64, i32, i64},
    "var": {f16},
    "var_mean": {f16},
    "view_as_complex": {f16, f32, f64},
}


inductor_expected_failures_single_sample["cuda"] = {
    "H": {b8, f16, f32, f64, i32, i64},
    "T": {b8, f16, f32, f64, i32, i64},
    "__getitem__": {b8, f16, f32, f64, i32, i64},
    "__radd__": {b8, f16, f32, f64, i32, i64},
    "__rand__": {b8, i32, i64},
    "__rmod__": {f16, f32, f64, i32, i64},
    "__rmul__": {b8, f16, f32, f64, i32, i64},
    "__ror__": {b8, i32, i64},
    "__rpow__": {f16, f32, f64, i32, i64},
    "__rsub__": {f16, f32, f64, i32, i64},
    "__rxor__": {b8, i32, i64},
    "addbmm": {f16},
    "addmm": {f16, f32, f64},
    "addr": {f16},
    "allclose": {f16, f32, f64},
    "angle": {f32, f64},
    "argwhere": {b8, f16, f32, f64, i32, i64},
    "baddbmm": {f16},
    "bernoulli": {f16, f32, f64},
    "bincount": {i32, i64},
    "chalf": {b8, f16, f32, f64, i32, i64},
    "cholesky": {f32, f64},
    "combinations": {b8, f16, f32, f64, i32, i64},
    "complex": {f16, f32, f64},
    "corrcoef": {f16, f32, f64, i32, i64},
    "cov": {f16, f32, f64, i32, i64},
    "cumsum": {f16},
    "equal": {b8, f16, f32, f64, i32, i64},
    "erf": {b8},
    "fft.fft": {f16, f32, f64},
    "fft.fft2": {b8, f16, f32, f64, i32, i64},
    "fft.fftn": {b8, f16, f32, f64, i32, i64},
    "fft.hfft": {b8, f16, f32, f64, i32, i64},
    "fft.hfft2": {b8, f16, f32, f64, i32, i64},
    "fft.hfftn": {b8, f16, f32, f64, i32, i64},
    "fft.ifft": {b8, f16, f32, f64, i32, i64},
    "fft.ifft2": {b8, f16, f32, f64, i32, i64},
    "fft.ifftn": {b8, f16, f32, f64, i32, i64},
    "fft.ihfft": {b8, f16, f32, f64, i32, i64},
    "fft.ihfft2": {f16, f32, f64},
    "fft.ihfftn": {f16, f32, f64},
    "fft.irfft": {b8, f16, f32, f64, i32, i64},
    "fft.irfft2": {b8, f16, f32, f64, i32, i64},
    "fft.irfftn": {b8, f16, f32, f64, i32, i64},
    "fft.rfft": {f16, f32, f64},
    "fft.rfft2": {f16, f32, f64},
    "fft.rfftn": {f16, f32, f64},
    "index_add": {b8, f16, f32, f64, i32, i64},
    "index_copy": {f16, f32, f64},
    "index_reduce": {f16, f32, f64},
    "inner": {f16, f32, f64},
    "isinf": {b8, i32, i64},
    "isnan": {b8, i32, i64},
    "istft": {f32, f64},
    "lgamma": {b8, i32, i64},
    "linalg.cholesky": {f32, f64},
    "linalg.cholesky_ex": {f32, f64},
    "linalg.eig": {f32, f64},
    "linalg.eigh": {f32, f64},
    "linalg.eigvals": {f32, f64},
    "linalg.eigvalsh": {f32, f64},
    "linalg.ldl_factor": {f32, f64},
    "linalg.lstsq": {f32, f64},
    "linalg.lstsq.grad_oriented": {f32, f64},
    "linalg.matrix_rank": {f32, f64},
    "linalg.matrix_rank.hermitian": {f32, f64},
    "linalg.pinv.hermitian": {f32, f64},
    "linalg.svd": {f32, f64},
    "logsumexp": {b8, f16, f32, i32, i64},
    "mH": {b8, f16, f32, f64, i32, i64},
    "mT": {b8, f16, f32, f64, i32, i64},
    "masked.argmax": {f16, f32, f64, i32},
    "masked.argmin": {f16, f32, f64, i32},
    "masked_scatter": {f16, f32, f64},
    "masked_select": {b8, f16, f32, f64, i32, i64},
    "matrix_exp": {f16},
    "max.reduction_no_dim": {b8},
    "max.reduction_with_dim": {b8, i32, i64},
    "min.reduction_no_dim": {b8},
    "min.reduction_with_dim": {b8, i32, i64},
    "multinomial": {f16, f32, f64},
    "nan_to_num": {b8, i32, i64},
    "new_empty": {f16, f32, f64, i32, i64},
    "new_empty_strided": {f16, f32, f64, i32, i64},
    "nn.functional._scaled_dot_product_attention": {f16, f32, f64},
    "nn.functional.conv_transpose3d": {f16},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.dropout": {f16, f32, f64},
    "nn.functional.dropout2d": {f16, f32, f64},
    "nn.functional.dropout3d": {f16, f32, f64},
    "nn.functional.feature_alpha_dropout.with_train": {f16, f32, f64},
    "nn.functional.feature_alpha_dropout.without_train": {b8, f16, f32, f64, i32, i64},
    "nn.functional.fractional_max_pool2d": {f16, f32, f64},
    "nn.functional.fractional_max_pool3d": {f16, f32, f64},
    "nn.functional.gaussian_nll_loss": {f16, f32, f64},
    "nn.functional.group_norm": {f16},
    "nn.functional.huber_loss": {f16, f32, f64},
    "nn.functional.instance_norm": {f16},
    "nn.functional.one_hot": {i64},
    "nn.functional.prelu": {f16},
    "nn.functional.rrelu": {f16, f32, f64},
    "nn.functional.soft_margin_loss": {f16},
    "nn.functional.triplet_margin_with_distance_loss": {f16, f32, f64, i32, i64},
    "nonzero": {b8, f16, f32, f64, i32, i64},
    "normal": {f16, f32, f64},
    "normal.number_mean": {f16, f32, f64},
    "pca_lowrank": {f32, f64},
    "pinverse": {f32, f64},
    "polar": {f32, f64},
    "pow": {i32, i64},
    "rand_like": {f16, f32, f64},
    "randint_like": {f16, f32, f64, i32, i64},
    "randn": {f16, f32, f64},
    "randn_like": {f16, f32, f64},
    "repeat_interleave": {b8, f16, f32, f64, i32, i64},
    "round.decimals_3": {f16},
    "segment_reduce.lengths": {f16, f32, f64},
    "segment_reduce.offsets": {f16, f32, f64},
    "sgn": {b8, f16, f32, f64, i32, i64},
    "sign": {i32, i64},
    "std_mean": {f16, f32, f64},
    "stft": {f32, f64},
    "svd": {f32, f64},
    "svd_lowrank": {f32, f64},
    "tensor_split": {b8, f16, f32, f64, i32, i64},
    "tensordot": {f16, f32, f64},
    "to": {b8, f16, f32, f64, i32, i64},
    "to_sparse": {f16, f32, f64},
    "uniform": {f16, f32, f64},
    "unique": {b8, f16, f32, f64, i32, i64},
    "unique_consecutive": {b8, f16, f32, f64, i32, i64},
    "view_as_complex": {f16, f32, f64},
}


class TestInductorOpInfo(TestCase):
    check_model = check_model
    check_model_cuda = check_model_cuda

    @onlyNativeDeviceTypes
    @suppress_warnings
    @skipCUDAMemoryLeakCheckIf(
        True
    )  # inductor kernels failing this test intermittently
    @_ops(op_db[START:END])
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

        if not ALL_SAMPLES:
            if isinstance(samples, (list, tuple)):
                samples = [samples[0]]
            else:
                samples = [next(samples)]

        for sample_input in samples:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            try:
                # with open("test_output.txt", "a") as f:
                #     print(f"RUNNING OP {op_name} on {device_type} with {dtype}", flush=True, file=f)
                #     print(f"RUNNING OP {op_name} on {device_type} with {dtype}", flush=True)
                if device_type == "cuda":
                    self.check_model_cuda(
                        fn, args, kwargs, check_lowp=False, nopython=True
                    )
                elif device_type == "cpu":
                    self.check_model(fn, args, kwargs, check_lowp=False, nopython=True)

            except Exception as e:

                if test_expect is TestExpect.XFAILURE:
                    return

                seen_failed[device_type].setdefault(op_name, set()).add(dtype)

                if COLLECT_EXPECT:
                    return

                raise e
            else:
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
