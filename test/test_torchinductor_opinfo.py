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
    "empty_like": {b8, f16, f32, f64},  # flaky
    "reciprocal": {b8},  # flaky
    "linalg.vander": {f32, f64},  # flaky
    "sgn": {f16, f32, f64},  # flaky
    "index_add": {b8, f16, f32, f64, i32, i64},  # flaky
    "index_select": {f16, f32, f64},  # flaky
}

inductor_skips["cuda"] = {
    # flaky
    "__rdiv__": {b8, f16, f32, f64, i32, i64},
    "mvlgamma.mvlgamma_p_1": {f16, f32, f64, i32, i64},
    "mvlgamma.mvlgamma_p_3": {f16, f32, f64, i32, i64},
    "mvlgamma.mvlgamma_p_5": {f16, f32, f64, i32, i64},
    "cumprod": {f32, f64},
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
}


inductor_expected_failures = defaultdict(dict)

inductor_expected_failures["cpu"] = {
    "H": {b8, f16, f32, f64, i32, i64},
    "T": {b8, f16, f32, f64, i32, i64},
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
    "_masked.log_softmax": {f32, f64},
    "_masked.mean": {f16},
    "_masked.normalize": {f16, f32, f64},
    "_masked.softmax": {f32, f64},
    "_masked.softmin": {f32, f64},
    "_masked.var": {f16},
    "allclose": {f16, f32, f64},
    "amax": {f16},
    "amin": {f16},
    "angle": {f16, f32, f64},
    "argwhere": {b8, f16, f32, f64, i32, i64},
    "as_strided": {b8, f16, f32, f64, i32, i64},
    "as_strided_scatter": {b8, f16, f32, f64, i32, i64},
    "bernoulli": {f32, f64},
    "bfloat16": {b8, f16, f32, f64, i32, i64},
    "bincount": {i32, i64},
    "bool": {b8, f16, f32, f64, i32, i64},
    "byte": {b8, f16, f32, f64, i32, i64},
    "cdist": {f32, f64},
    "chalf": {b8, f16, f32, f64, i32, i64},
    "char": {b8, f16, f32, f64, i32, i64},
    "cholesky_inverse": {f32, f64},
    "chunk": {b8},
    "combinations": {b8, f16, f32, f64, i32, i64},
    "complex": {f16, f32, f64},
    "constant_pad_nd": {b8},
    "contiguous": {b8, f16, f32, f64, i32, i64},
    "corrcoef": {f32, f64, i32, i64},
    "cov": {f32, f64, i32, i64},
    "cumulative_trapezoid": {f32, f64, i32, i64},
    "dist": {f16},
    "double": {b8, f16, f32, f64, i32, i64},
    "einsum": {f32, f64, i32, i64},
    "empty_like": {b8, i32, i64},
    "equal": {b8, f16, f32, f64, i32, i64},
    "erf": {b8},
    "expand": {b8, f16, f32, f64, i32, i64},
    "expand_as": {b8, f16, f32, f64, i32, i64},
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
    "fill": {b8, f16, f32, f64, i32, i64},
    "flip": {b8, f16, f32, f64, i32, i64},
    "float": {b8, f16, f32, f64, i32, i64},
    "gather": {b8, f16, f32, f64, i32, i64},
    "gradient": {f16, f32, f64, i32, i64},
    "half": {b8, f16, f32, f64, i32, i64},
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
    "linalg.householder_product": {f32, f64},
    "linalg.ldl_factor": {f32, f64},
    "linalg.lstsq": {f32, f64},
    "linalg.lstsq.grad_oriented": {f32, f64},
    "linalg.matrix_norm": {f16, f32, f64},
    "linalg.norm": {f16, f32, f64},
    "linalg.norm.subgradients_at_zero": {f16, f32, f64},
    "linalg.solve_triangular": {f32, f64},
    "linspace": {f16, f32, f64, i32, i64},
    "log_softmax": {f32, f64},
    "log_softmax.dtype": {b8, f16, f32, f64, i32, i64},
    "logdet": {f32, f64},
    "logical_not": {f16, f32, f64, i32, i64},
    "long": {b8, f16, f32, f64, i32, i64},
    "mH": {b8, f16, f32, f64, i32, i64},
    "mT": {b8, f16, f32, f64, i32, i64},
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
    "new_empty": {b8, f16, f32, f64, i32, i64},
    "new_empty_strided": {b8, f16, f32, f64, i32, i64},
    "new_full": {b8, f16, f32, f64, i32, i64},
    "new_ones": {b8, f16, f32, f64, i32, i64},
    "new_zeros": {b8, f16, f32, f64, i32, i64},
    "nn.functional._scaled_dot_product_attention": {f32, f64},
    "nn.functional.adaptive_avg_pool3d": {f16},
    "nn.functional.avg_pool1d": {i64},
    "nn.functional.avg_pool2d": {i64},
    "nn.functional.batch_norm": {f32, f64},
    "nn.functional.cosine_embedding_loss": {b8},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.dropout": {f32, f64},
    "nn.functional.dropout2d": {f32, f64},
    "nn.functional.dropout3d": {f32, f64},
    "nn.functional.embedding": {f16, f32, f64},
    "nn.functional.embedding_bag": {f16, f32, f64},
    "nn.functional.feature_alpha_dropout.with_train": {f32, f64},
    "nn.functional.feature_alpha_dropout.without_train": {b8, f16, f32, f64, i32, i64},
    "nn.functional.fractional_max_pool2d": {f32, f64},
    "nn.functional.fractional_max_pool3d": {f32, f64},
    "nn.functional.gaussian_nll_loss": {f32, f64},
    "nn.functional.huber_loss": {f16, f32, f64},
    "nn.functional.max_pool1d": {f32, f64},
    "nn.functional.max_pool2d": {f32, f64},
    "nn.functional.max_pool3d": {f32, f64},
    "nn.functional.max_unpool1d": {f32, f64},
    "nn.functional.max_unpool1d.grad": {f32, f64},
    "nn.functional.max_unpool2d": {f32, f64},
    "nn.functional.max_unpool2d.grad": {f32, f64},
    "nn.functional.max_unpool3d": {f32, f64},
    "nn.functional.max_unpool3d.grad": {f32, f64},
    "nn.functional.one_hot": {i64},
    "nn.functional.pad.circular": {f16},
    "nn.functional.pairwise_distance": {f16},
    "nn.functional.pixel_shuffle": {b8, f16, f32, f64, i32, i64},
    "nn.functional.pixel_unshuffle": {b8, f16, f32, f64, i32, i64},
    "nn.functional.prelu": {f32, f64},
    "nn.functional.rrelu": {f32, f64},
    "nn.functional.smooth_l1_loss": {f16},
    "nn.functional.softmin": {f32, f64},
    "nn.functional.softmin.with_dtype": {f16, f32, f64, i32, i64},
    "nn.functional.upsample_bilinear": {f32, f64},
    "nn.functional.upsample_nearest": {f32, f64},
    "nonzero": {b8, f16, f32, f64, i32, i64},
    "normal": {f16, f32, f64},
    "normal.number_mean": {f16, f32, f64},
    "pca_lowrank": {f32, f64},
    "pinverse": {f32, f64},
    "polar": {f32, f64},
    "polygamma.polygamma_n_0": {b8, f32, f64, i32, i64},
    "polygamma.polygamma_n_1": {b8, f32, f64, i32, i64},
    "polygamma.polygamma_n_2": {b8, f32, f64, i32, i64},
    "polygamma.polygamma_n_3": {b8, f32, f64, i32, i64},
    "polygamma.polygamma_n_4": {b8, f32, f64, i32, i64},
    "quantile": {f32, f64},
    "rad2deg": {f16},
    "rand_like": {f16, f32, f64},
    "randint_like": {f16, f32, f64, i32, i64},
    "randn": {f16, f32, f64},
    "randn_like": {f16, f32, f64},
    "renorm": {f16, f32, f64},
    "repeat": {b8, f16, f32, f64, i32, i64},
    "repeat_interleave": {b8, f16, f32, f64, i32, i64},
    "reshape_as": {b8, f16, f32, f64, i32, i64},
    "resize_": {b8, f16, f32, f64, i32, i64},
    "resize_as_": {b8, f16, f32, f64, i32, i64},
    "scatter": {b8, f16, f32, f64, i32, i64},
    "scatter_add": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.amax": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.amin": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.mean": {f16, f32, f64, i32, i64},
    "scatter_reduce.prod": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.sum": {b8, f16, f32, f64, i32, i64},
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
    "sparse.sampled_addmm": {f32, f64},
    "special.polygamma.special_polygamma_n_0": {b8, f32, f64, i32, i64},
    "split": {b8},
    "split.list_args": {b8},
    "split_with_sizes": {b8},
    "squeeze": {b8, f16, f32, f64, i32, i64},
    "std": {f16},
    "std_mean": {f16, f32, f64},
    "stft": {f32, f64},
    "sum": {f16},
    "sum_to_size": {b8, f16, f32, f64, i32, i64},
    "svd_lowrank": {f32, f64},
    "symeig": {f32, f64},
    "tensordot": {f32, f64, i32, i64},
    "to": {b8, f16, f32, f64, i32, i64},
    "to_sparse": {b8, f16, f32, f64, i32, i64},
    "tril": {f16},
    "triu": {f16},
    "unbind": {b8},
    "unflatten": {b8, f16, f32, f64, i32, i64},
    "unfold": {b8, f16, f32, f64, i32, i64},
    "uniform": {f16, f32, f64},
    "unique": {b8, f32, f64, i32, i64},
    "unique_consecutive": {b8, f32, f64, i32, i64},
    "var": {f16},
    "var_mean": {f16},
    "view": {b8, f16, f32, f64, i32, i64},
    "view_as": {b8, f16, f32, f64, i32, i64},
    "view_as_complex": {f16, f32, f64},
    "where": {b8, f16, f32, f64, i32, i64},
    "zero_": {b8, f16, f32, f64, i32, i64},
}

inductor_expected_failures["cuda"] = {
    "H": {b8, f16, f32, f64, i32, i64},
    "T": {b8, f16, f32, f64, i32, i64},
    "__getitem__": {b8, f16, f32, f64, i32, i64},
    "__radd__": {b8, f16, f32, f64, i32, i64},
    "__rand__": {b8, i32, i64},
    "__rdiv__": {b8, f16, f32, f64, i32, i64},
    "__rmatmul__": {f16, f32, f64},
    "__rmod__": {f16, f32, f64, i32, i64},
    "__rmul__": {b8, f16, f32, f64, i32, i64},
    "__ror__": {b8, i32, i64},
    "__rpow__": {f16, f32, f64, i32, i64},
    "__rsub__": {f16, f32, f64, i32, i64},
    "__rxor__": {b8, i32, i64},
    "_masked.argmin": {i32, i64},
    "_masked.log_softmax": {f16, f32, f64},
    "_masked.mean": {b8},
    "_masked.normalize": {f16, f32, f64},
    "_masked.softmax": {f16, f32, f64},
    "_masked.softmin": {f16, f32, f64},
    "add": {b8},
    "addbmm": {f16},
    "addr": {f16},
    "allclose": {f16, f32, f64},
    "amax": {b8},
    "amin": {b8},
    "angle": {f32, f64},
    "argwhere": {b8, f16, f32, f64, i32, i64},
    "as_strided": {b8, f16, f32, f64, i32, i64},
    "as_strided_scatter": {b8, f16, f32, f64, i32, i64},
    "atanh": {f16, f32, f64},
    "baddbmm": {f16, f32, f64},
    "bernoulli": {f16, f32, f64},
    "bfloat16": {b8, f16, f32, f64, i32, i64},
    "bincount": {i32, i64},
    "bool": {b8, f16, f32, f64, i32, i64},
    "byte": {b8, f16, f32, f64, i32, i64},
    "cdist": {f32, f64},
    "chalf": {b8, f16, f32, f64, i32, i64},
    "char": {b8, f16, f32, f64, i32, i64},
    "chunk": {b8},
    "clamp": {f16, f32, f64},
    "combinations": {b8, f16, f32, f64, i32, i64},
    "complex": {f16, f32, f64},
    "constant_pad_nd": {b8, f16, f32, f64, i32, i64},
    "contiguous": {b8, f16, f32, f64, i32, i64},
    "corrcoef": {f16, f32, f64, i32, i64},
    "cov": {f16, f32, f64, i32, i64},
    "cross": {f16},
    "cumprod": {f16},
    "cumsum": {f16},
    "cumulative_trapezoid": {f16, f32, f64, i32, i64},
    "diff": {f16, f32, f64, i32, i64},
    "dist": {f16, f32, f64},
    "dot": {f16, f32, f64},
    "double": {b8, f16, f32, f64, i32, i64},
    "einsum": {f16, f32, f64},
    "empty_like": {f16, f32, f64, i32, i64},
    "eq": {b8, f16, f32, f64, i32, i64},
    "equal": {b8, f16, f32, f64, i32, i64},
    "erf": {b8, f16, f32},
    "expand": {b8, f16, f32, f64, i32, i64},
    "expand_as": {b8, f16, f32, f64, i32, i64},
    "eye": {b8, f16, f32, f64, i32, i64},
    "fft.fft": {f16, f32, f64},
    "fft.fft2": {b8, f16, f32, f64, i32, i64},
    "fft.fftn": {b8, f16, f32, f64, i32, i64},
    "fft.hfft": {b8, f16, f32, f64, i32, i64},
    "fft.hfft2": {b8, f16, f32, f64, i32, i64},
    "fft.hfftn": {b8, f16, f32, f64, i32, i64},
    "fft.ifft": {f16, f32, f64},
    "fft.ifft2": {b8, f16, f32, f64, i32, i64},
    "fft.ifftn": {b8, f16, f32, f64, i32, i64},
    "fft.ihfft": {f16, f32, f64},
    "fft.ihfft2": {f16, f32, f64},
    "fft.ihfftn": {f16, f32, f64},
    "fft.irfft": {b8, f16, f32, f64, i32, i64},
    "fft.irfft2": {b8, f16, f32, f64, i32, i64},
    "fft.irfftn": {b8, f16, f32, f64, i32, i64},
    "fft.rfft": {f16, f32, f64},
    "fft.rfft2": {f16, f32, f64},
    "fft.rfftn": {f16, f32, f64},
    "fill": {b8, f16, f32, f64, i32, i64},
    "flip": {b8, f16, f32, f64, i32, i64},
    "float": {b8, f16, f32, f64, i32, i64},
    "gather": {b8, f16, f32, f64, i32, i64},
    "gradient": {f16, f32, f64, i32, i64},
    "half": {b8, f16, f32, f64, i32, i64},
    # "hsplit": {f16, f64},
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
    "linalg.householder_product": {f32, f64},
    "linalg.ldl_factor": {f32, f64},
    "linalg.lstsq": {f32, f64},
    "linalg.lstsq.grad_oriented": {f32, f64},
    "linalg.matrix_norm": {f16, f32, f64},
    "linalg.norm": {f16, f32, f64},
    "linalg.norm.subgradients_at_zero": {f16, f32, f64},
    "linalg.solve_triangular": {f32, f64},
    "linspace": {f16, f32, f64, i32, i64},
    "log_softmax": {f16, f32, f64},
    "log_softmax.dtype": {b8, f16, f32, f64, i32, i64},
    "logical_not": {f16, f32, f64, i32, i64},
    "long": {b8, f16, f32, f64, i32, i64},
    "mH": {b8, f16, f32, f64, i32, i64},
    "mT": {b8, f16, f32, f64, i32, i64},
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
    "new_full": {b8, f16, f32, f64, i32, i64},
    "new_ones": {b8, f16, f32, f64, i32, i64},
    "new_zeros": {b8, f16, f32, f64, i32, i64},
    "nn.functional._scaled_dot_product_attention": {f16, f32, f64},
    "nn.functional.batch_norm": {f16, f32, f64},
    "nn.functional.batch_norm.without_cudnn": {f16, f32, f64},
    "nn.functional.bilinear": {f16},
    "nn.functional.conv_transpose3d": {f16},
    "nn.functional.cross_entropy": {f16},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.dropout": {f16, f32, f64},
    "nn.functional.dropout2d": {f16, f32, f64},
    "nn.functional.dropout3d": {f16, f32, f64},
    "nn.functional.embedding": {f16, f32, f64},
    "nn.functional.embedding_bag": {f16, f32, f64},
    "nn.functional.feature_alpha_dropout.with_train": {f16, f32, f64},
    "nn.functional.feature_alpha_dropout.without_train": {b8, f16, f32, f64, i32, i64},
    "nn.functional.fractional_max_pool2d": {f16, f32, f64},
    "nn.functional.fractional_max_pool3d": {f16, f32, f64},
    "nn.functional.gaussian_nll_loss": {f16, f32, f64},
    "nn.functional.grid_sample": {f16},
    "nn.functional.huber_loss": {f16, f32, f64},
    "nn.functional.max_pool1d": {f16, f32, f64},
    "nn.functional.max_pool2d": {f16, f32, f64},
    "nn.functional.max_pool3d": {f16, f32, f64},
    "nn.functional.max_unpool1d": {f16, f32, f64},
    "nn.functional.max_unpool1d.grad": {f16, f32, f64},
    "nn.functional.max_unpool2d": {f16, f32, f64},
    "nn.functional.max_unpool2d.grad": {f16, f32, f64},
    "nn.functional.max_unpool3d": {f16, f32, f64},
    "nn.functional.max_unpool3d.grad": {f16, f32, f64},
    "nn.functional.multilabel_soft_margin_loss": {f16},
    "nn.functional.one_hot": {i64},
    "nn.functional.pixel_shuffle": {b8, f16, f32, f64, i32, i64},
    "nn.functional.pixel_unshuffle": {b8, f16, f32, f64, i32, i64},
    "nn.functional.prelu": {f16, f32, f64},
    "nn.functional.rrelu": {f16, f32, f64},
    "nn.functional.soft_margin_loss": {f16},
    "nn.functional.softmin": {f16, f32, f64},
    "nn.functional.softmin.with_dtype": {f16, f32, f64, i32, i64},
    "nn.functional.upsample_bilinear": {f16, f32, f64},
    "nn.functional.upsample_nearest": {f16, f32, f64},
    "nonzero": {b8, f16, f32, f64, i32, i64},
    "normal": {f16, f32, f64},
    "normal.number_mean": {f16, f32, f64},
    "pca_lowrank": {f32, f64},
    "pinverse": {f32, f64},
    "polar": {f32, f64},
    "polygamma.polygamma_n_0": {b8, f16, f32, f64, i32, i64},
    "polygamma.polygamma_n_1": {b8, f16, f32, f64, i32, i64},
    "polygamma.polygamma_n_2": {b8, f16, f32, f64, i32, i64},
    "polygamma.polygamma_n_3": {b8, f16, f32, f64, i32, i64},
    "polygamma.polygamma_n_4": {b8, f16, f32, f64, i32, i64},
    "quantile": {f32, f64},
    "rand_like": {f16, f32, f64},
    "randint_like": {f16, f32, f64, i32, i64},
    "randn": {f16, f32, f64},
    "randn_like": {f16, f32, f64},
    "renorm": {f16, f32, f64},
    "repeat": {b8, f16, f32, f64, i32, i64},
    "repeat_interleave": {b8, f16, f32, f64, i32, i64},
    "reshape_as": {b8, f16, f32, f64, i32, i64},
    "resize_": {b8, f16, f32, f64, i32, i64},
    "resize_as_": {b8, f16, f32, f64, i32, i64},
    "scatter": {b8, f16, f32, f64, i32, i64},
    "scatter_add": {b8, f16, f32, f64, i32, i64},
    "scatter_reduce.amax": {f16, f32, f64, i32, i64},
    "scatter_reduce.amin": {f16, f32, f64, i32, i64},
    "scatter_reduce.mean": {f16, f32, f64, i32, i64},
    "scatter_reduce.prod": {f16, f32, f64, i32, i64},
    "scatter_reduce.sum": {b8, f16, f32, f64, i32, i64},
    "segment_reduce.lengths": {f16, f32, f64},
    "segment_reduce.offsets": {f16, f32, f64},
    "select": {b8},
    "select_scatter": {b8},
    "sgn": {b8, i32, i64},
    "short": {b8, f16, f32, f64, i32, i64},
    "sign": {b8, i32, i64},
    "slice_scatter": {b8},
    "softmax": {f16, f32, f64},
    "softmax.with_dtype": {b8, f16, f32, f64, i32, i64},
    "special.polygamma.special_polygamma_n_0": {b8, f16, f32, f64, i32, i64},
    "split": {b8},
    "split.list_args": {b8},
    "split_with_sizes": {b8},
    "squeeze": {b8, f16, f32, f64, i32, i64},
    "std_mean": {f16, f32, f64},
    "stft": {f32, f64},
    "sum_to_size": {b8, f16, f32, f64, i32, i64},
    "svd_lowrank": {f32, f64},
    "symeig": {f32, f64},
    "tensordot": {f16, f32, f64},
    "to": {b8, f16, f32, f64, i32, i64},
    "to_sparse": {b8, f16, f32, f64, i32, i64},
    "unbind": {b8},
    "unflatten": {b8, f16, f32, f64, i32, i64},
    "unfold": {b8, f16, f32, f64, i32, i64},
    "uniform": {f16, f32, f64},
    "unique": {b8, f16, f32, f64, i32, i64},
    "unique_consecutive": {b8, f16, f32, f64, i32, i64},
    "view": {b8, f16, f32, f64, i32, i64},
    "view_as": {b8, f16, f32, f64, i32, i64},
    "view_as_complex": {f16, f32, f64},
    "where": {b8, f16, f32, f64, i32, i64},
    "zero_": {b8, f16, f32, f64, i32, i64},
}


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
            # kwargs = sample_input.kwargs

        try:
            if device_type == "cuda":
                self.check_model_cuda(fn, args, check_lowp=False)
            elif device_type == "cpu":
                self.check_model(fn, args, check_lowp=False)

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
