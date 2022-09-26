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
inductor_expected_failures[\"{device_type}\"] = {{
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
    "lu_unpack": {f32, f64},  # free(): invalid next size (fast)
    "__rdiv__": {b8, f16, f32, f64, i32, i64},  # flaky
    "mvlgamma.mvlgamma_p_1": {f32, f64, i32, i64},  # flaky
    "mvlgamma.mvlgamma_p_3": {f32, f64, i32, i64},  # flaky
    "mvlgamma.mvlgamma_p_5": {f32, f64, i32, i64},  # flaky
    "cumprod": {f32, f64},  # flaky
    "_masked.prod": {f32, f64},  # flaky
    "_masked.std": {b8, f16, f32, f64, i32, i64},  # segfault
    "histc": {b8, f16, f32, f64, i32, i64},  # segfault
    "empty_like": {b8, f16, f32, f64},  # flaky
    "reciprocal": {b8},  # flaky
    "linalg.ldl_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "linalg.lu_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "linalg.vander": {f32, f64},  # flaky
    "lu_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "reciprocal": {b8, f16, f32, f64, i32, i64},  # segfault
    "sgn": {f16, f32, f64},  # flaky
    "index_add": {b8, f16, f32, f64, i32, i64},  # flaky
    "index_select": {f16, f32, f64},  # flaky,
    "nn.functional.embedding_bag": {b8, f16, f32, f64, i32, i64},  # segfault
}

inductor_skips["cuda"] = {
    # flaky
    "__rdiv__": {b8, f16, f32, f64, i32, i64},
    "mvlgamma.mvlgamma_p_1": {f16, f32, f64, i32, i64},
    "mvlgamma.mvlgamma_p_3": {f16, f32, f64, i32, i64},
    "mvlgamma.mvlgamma_p_5": {f16, f32, f64, i32, i64},
    "cumprod": {f16, f32, f64},
    "_masked.prod": {f16, f32, f64},
    "empty_like": {f16, f32, f64},
    "reciprocal": {b8},
    "linalg.vander": {f32, f64},
    "sparse.sampled_addmm": {f32, f64},
    "nn.functional.conv_transpose1d": {f16},
    "nn.functional.conv_transpose2d": {f16},
    # Call parameter type does not match function signature!
    "_masked.logsumexp": {f64},
    "cos": {b8, f64, i32, i64},
    "erf": {f64, i32, i64},
    "exp": {b8, f64, i32, i64},
    "isclose": {b8, f16, f32, f64, i32, i64},  # LLVM ERROR
    "isfinite": {f16, f32, f64, i32, i64},  # LLVM ERROR
    "log": {b8, i32, i64},
    "log1p": {b8, i32, i64},
    "log2": {b8, i32, i64},
    "logsumexp": {f64},
    "lu_unpack": {f32, f64},  # RuntimeError: CUDA error
    "nan_to_num": {f16, f32, f64, i32, i64},  # LLVM ERROR
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
    "rsqrt": {b8, i32, i64},
    "sigmoid": {b8, f64, i32, i64},
    "sin": {b8, f64, i32, i64},
    "special.log_ndtr": {f64},
    "special.ndtr": {f64},
    "sqrt": {b8, i32, i64},
    "tanh": {f64},
    "nn.functional.embedding_bag": {b8, f16, f32, f64, i32, i64},  # segfault
    "roll": {b8, f16, f32, f64, i32, i64},  # segfault
    "_masked.log_softmax": {b8, f16, f32, f64, i32, i64},  # segfault
    "_masked.logaddexp": {b8, f16, f32, f64, i32, i64},  # segfault
    "scatter_add":  {b8, f16, f32, f64, i32, i64},  # segfault
    "scatter_reduce": {b8, f16, f32, f64, i32, i64},  # segfault
    "scatter_reduce.amax": {f16, f32, f64, i32, i64}, # segfault
    "scatter_reduce.amin": {f16, f32, f64, i32, i64}, # segfault
    "scatter_reduce.mean": {f16, f32, f64, i32, i64}, # segfault
    "scatter_reduce.prod": {f16, f32, f64, i32, i64}, # segfault
    "scatter_reduce.sum": {b8, f16, f32, f64, i32, i64}, # segfault
    "softmax": {b8, f16, f32, f64, i32, i64}, # segfault
    "softmax.with_dtype": {b8, f16, f32, f64, i32, i64}, # segfault
}


inductor_expected_failures = defaultdict(dict)

inductor_expected_failures["cpu"] = {
    "__getitem__": {b8, f16, f32, f64, i32, i64},
    "__radd__": {b8, f16, f32, f64, i32, i64},
    "__rand__": {b8, i32, i64},
    "__rmatmul__": {f32, f64, i32, i64},
    "__rmod__": {f16, f32, f64},
    "__rmul__": {b8, f16, f32, f64, i32, i64},
    "__ror__": {b8, i32, i64},
    "__rpow__": {f16, f32, f64, i32, i64},
    "__rsub__": {f16, f32, f64, i32, i64},
    "__rxor__": {b8, i32, i64},
    "_masked.amax": {f16},
    "_masked.amin": {f16},
    "_masked.argmax": {f16, f32, f64, i32, i64},
    "_masked.argmin": {f16, f32, f64, i32, i64},
    "_masked.log_softmax": {f32, f64},
    "_masked.mean": {f16},
    "_masked.median": {f32, f64},
    "_masked.norm": {f16},
    "_masked.normalize": {f16, f32, f64},
    "_masked.softmax": {f32, f64},
    "_masked.softmin": {f32, f64},
    "_masked.sum": {f16},
    "_masked.var": {f16},
    "add": {f16},
    "addmm": {f32, f64, i32, i64},
    "allclose": {f16, f32, f64},
    "amax": {f16},
    "amin": {f16},
    "aminmax": {b8, f32, f64, i32, i64},
    "angle": {f16, f32, f64},
    "argmax": {f16, f32, f64, i32, i64},
    "argmin": {f16, f32, f64, i32, i64},
    "argwhere": {b8, f16, f32, f64, i32, i64},
    "baddbmm": {i32, i64},
    "bernoulli": {f32, f64},
    "bincount": {i32, i64},
    "bool": {b8, f16, f32, f64, i32, i64},
    "bfloat16": {b8, f16, f32, f64, i32, i64},
    "byte": {b8, f16, f32, f64, i32, i64},
    "cdist": {f32, f64},
    "chalf": {b8, f16, f32, f64, i32, i64},
    "char": {b8, f16, f32, f64, i32, i64},
    "cholesky_inverse": {f32, f64},
    "chunk": {b8},
    "combinations": {b8, f16, f32, f64, i32, i64},
    "complex": {f16, f32, f64},
    "constant_pad_nd": {b8},
    "corrcoef": {f32, f64, i32, i64},
    "cov": {f32, f64, i32, i64},
    "cumulative_trapezoid": {f32, f64, i32, i64},
    "dist": {f16},
    "double": {b8, f16, f32, f64, i32, i64},
    "empty_like": {i64},
    "equal": {b8, f16, f32, f64, i32, i64},
    "erf": {b8},
    "fft.fft": {f32, f64},
    "fft.fft2": {b8, f32, f64, i32, i64},
    "fft.fftn": {b8, f32, f64, i32, i64},
    "fft.hfft": {b8, f32, f64, i32, i64},
    "fft.hfft2": {b8, f32, f64, i32, i64},
    "fft.hfftn": {b8, f32, f64, i32, i64},
    "fft.ifft": {f32, f64},
    "fft.ifft2": {b8, f32, f64, i32, i64},
    "fft.ifftn": {b8, f32, f64, i32, i64},
    "fft.ihfft": {f32, f64},
    "fft.ihfft2": {f32, f64},
    "fft.ihfftn": {f32, f64},
    "fft.irfft": {b8, f32, f64, i32, i64},
    "fft.irfft2": {b8, f32, f64, i32, i64},
    "fft.irfftn": {b8, f32, f64, i32, i64},
    "fft.rfft": {f32, f64},
    "fft.rfft2": {f32, f64},
    "fft.rfftn": {f32, f64},
    "float": {b8, f16, f32, f64, i32, i64},
    "gather": {b8, f16, f32, f64, i32, i64},
    "gradient": {f16, f32, f64, i32, i64},
    "half": {b8, f16, f32, f64, i32, i64},
    "histogram": {f32, f64},
    "histogramdd": {f32, f64},
    "index_put": {b8, f16, f32, f64, i32, i64},
    "index_reduce": {f16, f32, f64},
    "inner": {f32, f64},
    "int": {b8, f16, f32, f64, i32, i64},
    "isclose": {b8, f16, f32, f64, i32, i64},
    "isfinite": {f16, f32, f64},
    "istft": {f32, f64},
    "kron": {b8, f16, f32, f64, i32, i64},
    "linalg.cond": {f32, f64},
    "linalg.det": {f32, f64},
    "linalg.det.singular": {f32, f64},
    "linalg.eig": {f32, f64},
    "linalg.eigh": {f32, f64},
    "linalg.eigvals": {f32, f64},
    "linalg.eigvalsh": {f32, f64},
    "linalg.householder_product": {f32, f64},
    "linalg.ldl_factor": {f32, f64},
    "linalg.lstsq": {f32, f64},
    "linalg.lstsq.grad_oriented": {f32, f64},
    "linalg.matrix_norm": {f16, f32, f64},
    "linalg.norm": {f16, f32, f64},
    "linalg.norm.subgradients_at_zero": {f16, f32, f64},
    "linalg.pinv": {f32, f64},
    "linspace": {f16, f32, f64, i32, i64},
    "log_softmax": {f32, f64},
    "log_softmax.dtype": {b8, f16, f32, f64, i32, i64},
    "logdet": {f32, f64},
    "logical_not": {f16, f32, f64, i32, i64},
    "long": {b8, f16, f32, f64, i32, i64},
    "masked_fill": {f16},
    "masked_scatter": {f16, f32, f64},
    "masked_select": {b8, f16, f32, f64, i32, i64},
    "max.reduction_with_dim": {b8, f16, f32, f64, i32, i64},
    "min.reduction_with_dim": {b8, f16, f32, f64, i32, i64},
    "multinomial": {f32, f64},
    "nan_to_num": {b8, f16, i32, i64},
    "nanquantile": {f32, f64},
    "narrow": {b8, f16, f32, f64, i32, i64},
    "native_layer_norm": {f32, f64},
    "nn.functional._scaled_dot_product_attention": {f32, f64},
    "nn.functional.adaptive_avg_pool3d": {f16},
    "nn.functional.avg_pool1d": {f32, f64, i64},
    "nn.functional.avg_pool2d": {i64},
    "nn.functional.cosine_embedding_loss": {b8},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.dropout": {f32, f64},
    "nn.functional.dropout2d": {f32, f64},
    "nn.functional.dropout3d": {f32, f64},
    "nn.functional.embedding_bag": {f16, f32, f64},
    "nn.functional.feature_alpha_dropout.with_train": {f32, f64},
    "nn.functional.feature_alpha_dropout.without_train": {b8, f16, f32, f64, i32, i64},
    "nn.functional.fractional_max_pool2d": {f32, f64},
    "nn.functional.fractional_max_pool3d": {f32, f64},
    "nn.functional.gaussian_nll_loss": {f32, f64},
    "nn.functional.huber_loss": {f16, f32, f64},
    "nn.functional.max_pool2d": {f32, f64},
    "nn.functional.one_hot": {i64},
    "nn.functional.pad.circular": {f16},
    "nn.functional.pairwise_distance": {f16},
    "nn.functional.pdist": {f32, f64},
    "nn.functional.rrelu": {f32, f64},
    "nn.functional.smooth_l1_loss": {f16},
    "nn.functional.softmin": {f32, f64},
    "nn.functional.softmin.with_dtype": {f16, f32, f64, i32, i64},
    "nn.functional.triplet_margin_with_distance_loss": {f32, f64, i32, i64},
    "nonzero": {b8, f16, f32, f64, i32, i64},
    "normal": {f16, f32, f64},
    "normal.number_mean": {f16, f32, f64},
    "pca_lowrank": {f32, f64},
    "pinverse": {f32, f64},
    "polar": {f32, f64},
    "quantile": {f32, f64},
    "rad2deg": {f16},
    "rand_like": {f16, f32, f64},
    "randint_like": {f16, f32, f64, i32, i64},
    "randn": {f16, f32, f64},
    "randn_like": {f16, f32, f64},
    "renorm": {f16, f32, f64},
    "repeat_interleave": {b8, f16, f32, f64, i32, i64},
    "rsub": {f16, f32, f64, i32, i64},
    "scatter": {b8, f16, f32, f64, i32, i64},
    "scatter_add": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.amax": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.amin": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.mean": {f16, f32, f64, i32, i64},
    "scatter_reduce.prod": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.sum": {b8, f16, f32, f64, i32, i64},
    "searchsorted": {f16, f32, f64, i32, i64},
    "segment_reduce.lengths": {f16, f32, f64},
    "segment_reduce.offsets": {f16, f32, f64},
    "select": {b8},
    "select_scatter": {b8},
    "sgn": {b8, i32, i64},
    "short": {b8, f16, f32, f64, i32, i64},
    "sign": {b8, i32, i64},
    "slice_scatter": {b8},
    "softmax": {f32, f64},
    "softmax.with_dtype": {b8, f16, f32, f64, i32, i64},
    "sort": {b8, f16, f32, f64, i32, i64},
    "sparse.sampled_addmm": {f32, f64},
    "split": {b8},
    "split.list_args": {b8},
    "split_with_sizes": {b8},
    "squeeze": {b8, f16, f32, f64, i32, i64},
    "std": {f16},
    "std_mean": {f16, f32, f64},
    "stft": {f32, f64},
    "sub": {f16},
    "sum_to_size": {f16},
    "svd_lowrank": {f32, f64},
    "symeig": {f32, f64},
    "to": {b8, f16, f32, f64, i32, i64},
    "to_sparse": {f32, f64},
    "tril": {f16},
    "triu": {f16},
    "unbind": {b8},
    "unflatten": {b8, f16, f32, f64, i32, i64},
    "uniform": {f16, f32, f64},
    "unique": {b8, f32, f64, i32, i64},
    "unique_consecutive": {b8, f32, f64, i32, i64},
    "var": {f16},
    "var_mean": {f16},
    "view_as_complex": {f16, f32, f64},
    "zero_": {b8},
       "_masked.amax": {f32, f64, i32, i64},
   "_masked.amin": {f32, f64, i32, i64},
   "_masked.logsumexp": {f32, f64, i32, i64},
   "_masked.mean": {b8, f32, f64, i32, i64},
   "_masked.norm": {f32, f64},
   "_masked.sum": {b8, f32, f64, i32, i64},
   "_masked.var": {i32, i64},
   "abs": {i32},
   "addr": {f16},
   "all": {b8, f16, f32, f64, i32, i64},
   "amax": {b8, f32, f64, i32, i64},
   "amin": {b8, f32, f64, i32, i64},
   "any": {b8, f16, f32, f64, i32, i64},
   "arange": {f16, f32, f64, i32, i64},
   "cholesky": {f32, f64},
   "clamp": {f32, f64},
   "clamp_max": {f16, f32, f64},
   "clamp_min": {f16, f32, f64},
   "constant_pad_nd": {f16, f32, f64, i32, i64},
   "copysign": {f16, f32, f64},
   "deg2rad": {f16},
   "diff": {b8, f16, f32, f64, i32, i64},
   "dist": {f32, f64},
   "div.floor_rounding": {f16, f32, f64, i32, i64},
   "div.no_rounding_mode": {b8, f16, f32, f64, i32, i64},
   "div.trunc_rounding": {f16, f32, f64, i32, i64},
   "dot": {f32, f64, i32, i64},
   "einsum": {f32, f64, i32, i64},
   "empty": {b8, f16, f32, i32, i64},
   "empty_like": {i32},
   "eq": {b8, f16, f32, f64, i32, i64},
   "erf": {f32, f64, i32, i64},
   "eye": {b8, f16, f32, f64, i32, i64},
   "fft.fft": {b8, i32, i64},
   "fft.ifft": {b8, i32, i64},
   "fft.ihfft": {b8, i32, i64},
   "fft.ihfft2": {b8, i32, i64},
   "fft.ihfftn": {b8, i32, i64},
   "fft.rfft": {b8, i32, i64},
   "fft.rfft2": {b8, i32, i64},
   "fft.rfftn": {b8, i32, i64},
   "floor_divide": {f16, f32, f64, i32, i64},
   "fmax": {f16, f32, f64},
   "fmin": {f16, f32, f64},
   "fmod": {f16, f32, f64},
   "ge": {b8, f16, f32, f64, i32, i64},
   "gt": {b8, f16, f32, f64, i32, i64},
   "index_copy": {f16, f32, f64},
   "inner": {i32, i64},
   "ldexp": {b8, f16, f32, f64, i32, i64},
   "le": {b8, f16, f32, f64, i32, i64},
   "lerp": {f32, f64},
   "linalg.cholesky": {f32, f64},
   "linalg.cholesky_ex": {f32, f64},
   "linalg.matrix_power": {f32, f64},
   "linalg.matrix_rank": {f32, f64},
   "linalg.matrix_rank.hermitian": {f32, f64},
   "linalg.svd": {f32, f64},
   "linalg.vecdot": {f32, f64},
   "linalg.vector_norm": {f16, f32, f64},
   "logsumexp": {b8, f32, f64, i32, i64},
   "matmul": {f32, f64, i32, i64},
   "max.reduction_no_dim": {f16},
   "mean": {f16, f32, f64},
   "min.reduction_no_dim": {f16},
   "nanmean": {f16, f32, f64},
   "new_empty": {b8, f16, f32, i32, i64},
   "new_empty_strided": {b8, f16, f32, f64, i32, i64},
   "new_zeros": {b8, f16, f32, f64, i32, i64},
   "nn.functional.avg_pool2d": {f32, f64},
   "nn.functional.batch_norm": {f32},
   "nn.functional.conv2d": {f32, f64, i64},
   "nn.functional.cosine_similarity": {f32, f64},
   "nn.functional.interpolate.nearest": {f32, f64},
   "nn.functional.layer_norm": {f32, f64},
   "nn.functional.local_response_norm": {i64},
   "nn.functional.normalize": {f32, f64},
   "nn.functional.poisson_nll_loss": {i32, i64},
   "nn.functional.upsample_nearest": {f32, f64},
   "norm": {f16, f32, f64},
   "permute": {b8, f16, f32, f64, i32, i64},
   "pow": {f16},
   "repeat": {b8, f16, f32, f64, i32, i64},
   "select_scatter": {f16, f32, f64, i32, i64},
   "slice": {b8, f16, f32, f64, i32, i64},
   "sum": {b8, f16, f32, f64, i32, i64},
   "svd": {f32, f64},
   "tensor_split": {b8, f16, f32, f64, i32, i64},
   "tensordot": {f32, f64, i32, i64},
   "tile": {b8, f16, f32, f64, i32, i64},
   "trapezoid": {f16, f32, f64, i32, i64},
   "trapz": {f16, f32, f64, i32, i64},
   "triu": {b8},
}

inductor_expected_failures["cuda"] = {
    "__getitem__": {b8, f16, f32, f64, i32, i64},
    "__radd__": {b8, f16, f32, f64, i32, i64},
    "__rand__": {b8, i32, i64},
    "__rmatmul__": {f16, f32, f64},
    "__rmod__": {f16, f32, f64, i32, i64},
    "__rmul__": {b8, f16, f32, f64, i32, i64},
    "__ror__": {b8, i32, i64},
    "__rpow__": {f16, f32, f64, i32, i64},
    "__rsub__": {f16, f32, f64, i32, i64},
    "__rxor__": {b8, i32, i64},
    "_masked.argmax": {f16, f32, f64, i32, i64},
    "_masked.argmin": {f16, f32, f64, i32, i64},
    "_masked.log_softmax": {f16, f32, f64},
    "_masked.mean": {b8},
    "_masked.normalize": {f16, f32, f64},
    "_masked.softmax": {f16, f32, f64},
    "_masked.softmin": {f16, f32, f64},
    "addbmm": {f16},
    "addmm": {f16, f32, f64},
    "addmv": {f16},
    "addr": {f16},
    "allclose": {f16, f32, f64},
    "amax": {b8},
    "amin": {b8},
    "angle": {f32, f64},
    "argmax": {f16, f32, f64, i32, i64},
    "argmin": {f16, f32, f64, i32, i64},
    "argwhere": {b8, f16, f32, f64, i32, i64},
    "atanh": {f16, f32, f64},
    "baddbmm": {f16, f32, f64},
    "bernoulli": {f16, f32, f64},
    "bfloat16": {b8, f16, f32, f64, i32, i64},
    "bincount": {i32, i64},
    "bool": {b8, f16, f32, f64, i32, i64},
    "byte": {b8, f16, f32, f64, i32, i64},
    "cdist": {f32, f64},
    "ceil": {i32, i64},
    "chalf": {b8, f16, f32, f64, i32, i64},
    "char": {b8, f16, f32, f64, i32, i64},
    "chunk": {b8},
    "clamp": {f16, f32, f64},
    "combinations": {b8, f16, f32, f64, i32, i64},
    "complex": {f16, f32, f64},
    "constant_pad_nd": {b8, f16, f32, f64, i32, i64},
    "corrcoef": {f16, f32, f64, i32, i64},
    "cov": {f16, f32, f64, i32, i64},
    "cross": {f16},
    "cumulative_trapezoid": {f16, f32, f64, i32, i64},
    "diff": {f16, f32, f64, i32, i64},
    "dist": {f16, f32, f64},
    "dot": {f16, f32, f64},
    "double": {b8, f16, f32, f64, i32, i64},
    "einsum": {f16, f32, f64},
    "empty_like": {i32, i64},
    "eq": {b8, f16, f32, f64, i32, i64},
    "equal": {b8, f16, f32, f64, i32, i64},
    "erf": {b8, f16, f32, f64, i32, i64},
    "eye": {b8, f16, f32, f64, i32, i64},
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
    "float": {b8, f16, f32, f64, i32, i64},
    "floor": {i32, i64},
    "gather": {b8, f16, f32, f64, i32, i64},
    "gradient": {f16, f32, f64, i32, i64},
    "half": {b8, f16, f32, f64, i32, i64},
    "index_add": {b8, f16, f32, f64, i32, i64},
    "index_put": {b8, f16, f32, f64, i32, i64},
    "index_reduce": {f16, f32, f64},
    "index_select": {f16, f32, f64},
    "inner": {f16, f32, f64},
    "int": {b8, f16, f32, f64, i32, i64},
    "isinf": {b8, i32, i64},
    "isnan": {b8, i32, i64},
    "istft": {f32, f64},
    "jiterator_2inputs_2outputs": {b8, f16, f32, f64, i32, i64},
    "jiterator_4inputs_with_extra_args": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary_return_by_ref": {b8, f16, f32, f64, i32, i64},
    "jiterator_unary": {b8, f16, f32, f64, i32, i64},
    "kron": {b8, f16, f32, f64, i32, i64},
    "linalg.cond": {f32, f64},
    "linalg.cross": {f16},
    "linalg.det": {f32, f64},
    "linalg.det.singular": {f32, f64},
    "linalg.eig": {f32, f64},
    "linalg.eigh": {f32, f64},
    "linalg.eigvals": {f32, f64},
    "linalg.eigvalsh": {f32, f64},
    "linalg.householder_product": {f32, f64},
    "linalg.ldl_factor": {f32, f64},
    "linalg.lstsq": {f32, f64},
    "linalg.lstsq.grad_oriented": {f32, f64},
    "linalg.lu": {f32, f64},
    "linalg.lu_factor": {f32, f64},
    "linalg.lu_factor_ex": {f32, f64},
    "linalg.matrix_norm": {f16, f32, f64},
    "linalg.norm": {f16, f32, f64},
    "linalg.norm.subgradients_at_zero": {f16, f32, f64},
    "linalg.pinv": {f32, f64},
    "linspace": {f16, f32, f64, i32, i64},
    "log_softmax": {f16, f32, f64},
    "log_softmax.dtype": {b8, f16, f32, f64, i32, i64},
    "logical_not": {f16, f32, f64, i32, i64},
    "long": {b8, f16, f32, f64, i32, i64},
    "masked_fill": {b8, i32, i64},
    "masked_scatter": {f16, f32, f64},
    "masked_select": {b8, f16, f32, f64, i32, i64},
    "matrix_exp": {f16},
    "max.reduction_with_dim": {b8, f16, f32, f64, i32, i64},
    "min.reduction_with_dim": {b8, f16, f32, f64, i32, i64},
    "multinomial": {f16, f32, f64},
    "nan_to_num": {b8},
    "nanquantile": {f32, f64},
    "narrow": {b8, f16, f32, f64, i32, i64},
    "native_layer_norm": {f16, f32, f64},
    "new_empty": {b8, f16, f32, f64, i32, i64},
    "new_empty_strided": {b8, f16, f32, f64, i32, i64},
    "new_zeros": {b8, f16, f32, f64, i32, i64},
    "nn.functional._scaled_dot_product_attention": {f16, f32, f64},
    "nn.functional.avg_pool1d": {f16, f32, f64},
    "nn.functional.bilinear": {f16},
    "nn.functional.binary_cross_entropy": {f16},
    "nn.functional.conv_transpose3d": {f16},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.dropout": {f16, f32, f64},
    "nn.functional.dropout2d": {f16, f32, f64},
    "nn.functional.dropout3d": {f16, f32, f64},
    "nn.functional.embedding_bag": {f16, f32, f64},
    "nn.functional.feature_alpha_dropout.with_train": {f16, f32, f64},
    "nn.functional.feature_alpha_dropout.without_train": {b8, f16, f32, f64, i32, i64},
    "nn.functional.fractional_max_pool2d": {f16, f32, f64},
    "nn.functional.fractional_max_pool3d": {f16, f32, f64},
    "nn.functional.gaussian_nll_loss": {f16, f32, f64},
    "nn.functional.huber_loss": {f16, f32, f64},
    "nn.functional.kl_div": {f16},
    "nn.functional.max_pool2d": {f16, f32, f64},
    "nn.functional.pdist": {f32, f64},
    "nn.functional.prelu": {f16},
    "nn.functional.rrelu": {f16, f32, f64},
    "nn.functional.soft_margin_loss": {f16},
    "nn.functional.softmin": {f16, f32, f64},
    "nn.functional.softmin.with_dtype": {f16, f32, f64, i32, i64},
    "nn.functional.triplet_margin_with_distance_loss": {f16, f32, f64, i32, i64},
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
    "renorm": {f16, f32, f64},
    "repeat_interleave": {b8, f16, f32, f64, i32, i64},
    "round": {i32, i64},
    "rsub": {f16, f32, f64, i32, i64},
    "scatter": {b8, f16, f32, f64, i32, i64},
    "scatter_add": {b8, f16, f32, f64, i32, i64},
    "searchsorted": {f16, f32, f64, i32, i64},
    "segment_reduce.lengths": {f16, f32, f64},
    "segment_reduce.offsets": {f16, f32, f64},
    "select": {b8},
    "select_scatter": {b8},
    "sgn": {b8, f16, f32, f64, i32, i64},
    "short": {b8, f16, f32, f64, i32, i64},
    "sign": {b8, i32, i64},
    "slice_scatter": {b8},
    "softmax": {f16, f32, f64},
    "softmax.with_dtype": {b8, f16, f32, f64, i32, i64},
    "sort": {f16, f32, f64, i32, i64},
    "split": {b8},
    "split.list_args": {b8},
    "split_with_sizes": {b8},
    "squeeze": {b8, f16, f32, f64, i32, i64},
    "std_mean": {f16, f32, f64},
    "stft": {f32, f64},
    "svd_lowrank": {f32, f64},
    "symeig": {f32, f64},
    "to": {b8, f16, f32, f64, i32, i64},
    "to_sparse": {f16, f32, f64},
    "trunc": {i32, i64},
    "unbind": {b8},
    "unflatten": {b8, f16, f32, f64, i32, i64},
    "uniform": {f16, f32, f64},
    "unique": {b8, f16, f32, f64, i32, i64},
    "unique_consecutive": {b8, f16, f32, f64, i32, i64},
    "view_as_complex": {f16, f32, f64},
    "zero_": {b8},
    "sum": {b8, f16, f32, f64, i32, i64},
    "svd": {f32, f64},
    "tensor_split": {b8, f16, f32, f64, i32, i64},
    "tensordot": {f16, f32, f64},
    "tile": {b8, f16, f32, f64, i32, i64},
    "trapezoid": {f16, f32, f64, i32, i64},
    "trapz": {f16, f32, f64, i32, i64},
    "triu": {b8},
    "vsplit": {f16},
}

import torchdynamo


class TestInductorOpInfo(TestCase):
    check_model = check_model
    check_model_cuda = check_model_cuda

    @onlyNativeDeviceTypes
    @suppress_warnings
    @skipCUDAMemoryLeakCheckIf(
        True
    )  # inductor kernels failing this test intermittently
    @_ops(op_db)
    def test_comprehensive(self, device, dtype, op):
        torchdynamo.reset()
        op_name = op.name
        if op.variant_test_name:
            op_name += f".{op.variant_test_name}"

        device_type = torch.device(device).type

        if dtype in inductor_skips[device_type].get(op_name, set()):
            test_expect = TestExpect.SKIP
            self.skipTest(f"{op_name} in {dtype} not supported")
        elif dtype in inductor_expected_failures[device_type].get(op_name, set()):
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

        for sample_input in samples:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            try:
                if device_type == "cuda" and op.name >= "softmax":
                    self.check_model_cuda(fn, args, kwargs, check_lowp=False)
                elif device_type == "cpu":
                    self.check_model(fn, args, kwargs, check_lowp=False)

            except Exception as e:

                if test_expect is TestExpect.XFAILURE:
                    return

                seen_failed[device_type].setdefault(op_name, set()).add(dtype)

                if COLLECT_EXPECT:
                    return

                raise e
            else:
                seen_succeeded[device_type].setdefault(op_name, set()).add(dtype)

            if test_expect is TestExpect.XFAILURE and not COLLECT_EXPECT:
                raise RuntimeError(
                    f"unexpected success {op_name}, {dtype}, {device_type}"
                )


instantiate_device_type_tests(TestInductorOpInfo, globals())

if __name__ == "__main__":
    run_tests()
