from collections import defaultdict
from enum import Enum
import os
import atexit

import torch.autograd

from torch.utils._python_dispatch import enable_torch_dispatch_mode
from torch._decomp import decomposition_table

from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils._mode_utils import no_dispatch
from torch.testing._internal.common_utils import (
    is_iterable_of_tensors,
    TestCase,
    suppress_warnings,
    TEST_WITH_ASAN,
    run_tests,
    skipIfSlowGradcheckEnv,
    dtype_abbrs,
    skipCUDAMemoryLeakCheckIf
)
from torch.testing._internal.common_device_type import (
    onlyNativeDeviceTypes,
    ops,
    instantiate_device_type_tests,
    OpDTypes
)
from torch.testing._internal.common_methods_invocations import op_db

import itertools
import functools
from functools import partial
import unittest

import torchdynamo
from torchdynamo.testing import rand_strided
from torchdynamo.testing import same

import importlib

from unittest.mock import patch


try:
    import sympy

    importlib.import_module("functorch")

    from torch._decomp import get_decompositions

    import torchinductor.config
    from torchinductor import config
    from torchinductor.compile_fx import compile_fx
    from torchinductor.ir import IndexingDiv
    from torchinductor.ir import ModularIndexing
    from torchinductor.sizevars import SizeVarAllocator
    from torchinductor.utils import has_torchvision_roi_align

    # This will only pass on pytorch builds newer than roughly 5/15/2022
    assert get_decompositions([torch.ops.aten.trace])
    # Requires functorch
    from torchinductor.compile_fx import compile_fx_inner
except (ImportError, ModuleNotFoundError, AssertionError):
    raise unittest.SkipTest("requires sympy/functorch")


bf16 = torch.bfloat16
f64 = torch.float64
f32 = torch.float32
f16 = torch.float16
c32 = torch.complex32
c64 = torch.complex64
c128 = torch.complex128
i8 = torch.int8
i16 = torch.int16
i32 = torch.int32
i64 = torch.int64
b8 = torch.bool
u8 = torch.uint8

_ops = partial(ops, dtypes=OpDTypes.supported,
                allowed_dtypes=[torch.float32])

# Success forces pass; failure forces fail; skip unconditionally skips testing
TestExpect = Enum("TestExpect", ("SUCCESS", "XFAILURE", "SKIP"))

COLLECT_EXPECT = os.getenv('PYTORCH_COLLECT_EXPECT', '0') == '1'


EXCLUDE_SET = {

    ("cuda", torch.float64, "_masked.logsumexp"),       # Call parameter type does not match function signature!
    ("cuda", torch.float64, "cos"),           # Call parameter type does not match function signature!

    ("cuda", torch.float64, "logsumexp"),           # Both operands to a binary operator are not of the same type!

    ("cuda", torch.float64, "nn.functional.binary_cross_entropy_with_logits"),           # Both operands to a binary operator are not of the same type!

    ("cuda", torch.float64, "special.log_ndtr"),           # Both operands to a binary operator are not of the same type!

    ("cpu", torch.float32, "lu_unpack"),   # invalid pointer

    ("cuda", torch.float32, "isclose"),           # LLVM ERROR
    ("cuda", torch.float32, "isfinite"),           # LLVM ERROR
    ("cuda", torch.float32, "nan_to_num"),           # LLVM ERROR


    ("cuda", torch.float32, "lu_unpack"),   # RuntimeError: CUDA error


}

seen_succeeded = {
    "cuda": {},
    "cpu": {}
}
seen_failed = {
    "cuda": {},
    "cpu": {}
}
failed_reasons = defaultdict(set)

def print_seen():
    expected_failures = {
        "cpu": [],
        "cuda": []
    }
    skips = {
        "cpu": [],
        "cuda": []
    }

    def fmt_dtypes(dtypes):
        r = ', '.join(sorted(dtype_abbrs[d] for d in dtypes))
        return '{' + r + '}'

    def process(device_type):

        for op, failed_dtypes in seen_failed[device_type].items():

            succeeded_dtypes = seen_succeeded.get(op, set())
            expected_failures_dtypes = failed_dtypes - succeeded_dtypes

            skips_dtypes = failed_dtypes & succeeded_dtypes
            reasons = ""
            if failed_reasons[op]:
                reasons = "  # " + ", ".join(sorted(failed_reasons[op]))
            if expected_failures_dtypes:
                expected_failures[device_type].append(f"    \"{op}\": {fmt_dtypes(expected_failures_dtypes)},{reasons}")
            if skips_dtypes:
                skips[device_type].append(f"    \"{op}\": {fmt_dtypes(skips_dtypes)},")

        expected_failures[device_type].sort()
        skips[device_type].sort()
        nl = '\n'
        print(f"""
inductor_expected_failures[\"{device_type}\"] = {{
{nl.join(expected_failures[device_type])}
}}

inductor_skips[\"{device_type}\"] = {{
{nl.join(skips[device_type])}
}}
""")

    process("cpu")
    process("cuda")

if COLLECT_EXPECT:
    atexit.register(print_seen)

inductor_expected_failures = defaultdict(dict)

inductor_expected_failures["cpu"] = {
    "H": {f32},
    "T": {f32},
    "__getitem__": {f32},
    "__radd__": {f32},
    "__rdiv__": {f32},
    "__rmatmul__": {f32},
    "__rmod__": {f32},
    "__rmul__": {f32},
    "__rpow__": {f32},
    "__rsub__": {f32},
    "_masked.amax": {f32},
    "_masked.amin": {f32},
    "_masked.argmax": {f32},
    "_masked.argmin": {f32},
    "_masked.log_softmax": {f32},
    "_masked.logsumexp": {f32},
    "_masked.mean": {f32},
    "_masked.norm": {f32},
    "_masked.normalize": {f32},
    "_masked.prod": {f32},
    "_masked.softmax": {f32},
    "_masked.softmin": {f32},
    "_masked.std": {f32},
    "_masked.sum": {f32},
    "_masked.var": {f32},
    "abs": {f32},
    "add": {f32},
    "addcmul": {f32},
    "all": {f32},
    "allclose": {f32},
    "amax": {f32},
    "amin": {f32},
    "angle": {f32},
    "any": {f32},
    "arange": {f32},
    "argmax": {f32},
    "argmin": {f32},
    "argwhere": {f32},
    "as_strided": {f32},
    "as_strided_scatter": {f32},
    "atanh": {f32},
    "baddbmm": {f32},
    "bernoulli": {f32},
    "bfloat16": {f32},
    "bool": {f32},
    "byte": {f32},
    "cartesian_prod": {f32},
    "cat": {f32},
    "cdist": {f32},
    "ceil": {f32},
    "chalf": {f32},
    "char": {f32},
    "cholesky_inverse": {f32},
    "clamp": {f32},
    "clone": {f32},
    "column_stack": {f32},
    "combinations": {f32},
    "complex": {f32},
    "constant_pad_nd": {f32},
    "contiguous": {f32},
    "corrcoef": {f32},
    "cos": {f32},
    "cov": {f32},
    "cumulative_trapezoid": {f32},
    "diff": {f32},
    "dist": {f32},
    "dot": {f32},
    "double": {f32},
    "dstack": {f32},
    "einsum": {f32},
    "empty_like": {f32},
    "eq": {f32},
    "equal": {f32},
    "erf": {f32},
    "exp": {f32},
    "expand": {f32},
    "expand_as": {f32},
    "eye": {f32},
    "fft.fft": {f32},
    "fft.fft2": {f32},
    "fft.fftn": {f32},
    "fft.fftshift": {f32},
    "fft.hfft": {f32},
    "fft.hfft2": {f32},
    "fft.hfftn": {f32},
    "fft.ifft": {f32},
    "fft.ifft2": {f32},
    "fft.ifftn": {f32},
    "fft.ifftshift": {f32},
    "fft.ihfft": {f32},
    "fft.ihfft2": {f32},
    "fft.ihfftn": {f32},
    "fft.irfft": {f32},
    "fft.irfft2": {f32},
    "fft.irfftn": {f32},
    "fft.rfft": {f32},
    "fft.rfft2": {f32},
    "fft.rfftn": {f32},
    "fill": {f32},
    "flip": {f32},
    "float": {f32},
    "floor": {f32},
    "fmax": {f32},
    "fmin": {f32},
    "full_like": {f32},
    "gather": {f32},
    "gradient": {f32},
    "half": {f32},
    "hstack": {f32},
    "index_put": {f32},
    "index_reduce": {f32},
    "index_select": {f32},
    "inner": {f32},
    "int": {f32},
    "isclose": {f32},
    "isfinite": {f32},
    "isinf": {f32},
    "isnan": {f32},
    "isreal": {f32},
    "istft": {f32},
    "kron": {f32},
    "linalg.cond": {f32},
    "linalg.det": {f32},
    "linalg.det.singular": {f32},
    "linalg.eig": {f32},
    "linalg.eigh": {f32},
    "linalg.eigvals": {f32},
    "linalg.householder_product": {f32},
    "linalg.ldl_factor": {f32},
    "linalg.lstsq": {f32},
    "linalg.lstsq.grad_oriented": {f32},
    "linalg.matrix_norm": {f32},
    "linalg.matrix_rank": {f32},
    "linalg.matrix_rank.hermitian": {f32},
    "linalg.norm": {f32},
    "linalg.norm.subgradients_at_zero": {f32},
    "linalg.solve_triangular": {f32},
    "linalg.vander": {f32},
    "linalg.vecdot": {f32},
    "linalg.vector_norm": {f32},
    "linspace": {f32},
    "log": {f32},
    "log1p": {f32},
    "log2": {f32},
    "log_softmax": {f32},
    "log_softmax.dtype": {f32},
    "logdet": {f32},
    "logical_not": {f32},
    "logit": {f32},
    "logsumexp": {f32},
    "long": {f32},
    "mH": {f32},
    "mT": {f32},
    "masked_fill": {f32},
    "masked_scatter": {f32},
    "masked_select": {f32},
    "matmul": {f32},
    "max.binary": {f32},
    "max.reduction_no_dim": {f32},
    "max.reduction_with_dim": {f32},
    "maximum": {f32},
    "mean": {f32},
    "median": {f32},
    "min.binary": {f32},
    "min.reduction_no_dim": {f32},
    "min.reduction_with_dim": {f32},
    "minimum": {f32},
    "multinomial": {f32},
    "mv": {f32},
    "nan_to_num": {f32},
    "nanmean": {f32},
    "nanmedian": {f32},
    "nanquantile": {f32},
    "nansum": {f32},
    "narrow": {f32},
    "native_layer_norm": {f32},
    "neg": {f32},
    "new_empty": {f32},
    "new_empty_strided": {f32},
    "new_full": {f32},
    "new_ones": {f32},
    "new_zeros": {f32},
    "nn.functional._scaled_dot_product_attention": {f32},
    "nn.functional.adaptive_avg_pool1d": {f32},
    "nn.functional.adaptive_avg_pool2d": {f32},
    "nn.functional.avg_pool1d": {f32},
    "nn.functional.avg_pool2d": {f32},
    "nn.functional.batch_norm": {f32},
    "nn.functional.bilinear": {f32},
    "nn.functional.binary_cross_entropy_with_logits": {f32},
    "nn.functional.conv1d": {f32},
    "nn.functional.conv2d": {f32},
    "nn.functional.conv_transpose1d": {f32},
    "nn.functional.conv_transpose2d": {f32},
    "nn.functional.conv_transpose3d": {f32},
    "nn.functional.cosine_embedding_loss": {f32},
    "nn.functional.cosine_similarity": {f32},
    "nn.functional.cross_entropy": {f32},
    "nn.functional.ctc_loss": {f32},
    "nn.functional.dropout": {f32},
    "nn.functional.dropout2d": {f32},
    "nn.functional.dropout3d": {f32},
    "nn.functional.elu": {f32},
    "nn.functional.embedding": {f32},
    "nn.functional.embedding_bag": {f32},
    "nn.functional.feature_alpha_dropout.with_train": {f32},
    "nn.functional.feature_alpha_dropout.without_train": {f32},
    "nn.functional.fractional_max_pool2d": {f32},
    "nn.functional.fractional_max_pool3d": {f32},
    "nn.functional.gaussian_nll_loss": {f32},
    "nn.functional.gelu": {f32},
    "nn.functional.glu": {f32},
    "nn.functional.group_norm": {f32},
    "nn.functional.hardsigmoid": {f32},
    "nn.functional.hardswish": {f32},
    "nn.functional.hardtanh": {f32},
    "nn.functional.hinge_embedding_loss": {f32},
    "nn.functional.huber_loss": {f32},
    "nn.functional.interpolate.bilinear": {f32},
    "nn.functional.kl_div": {f32},
    "nn.functional.l1_loss": {f32},
    "nn.functional.layer_norm": {f32},
    "nn.functional.leaky_relu": {f32},
    "nn.functional.margin_ranking_loss": {f32},
    "nn.functional.max_pool1d": {f32},
    "nn.functional.max_pool2d": {f32},
    "nn.functional.max_pool3d": {f32},
    "nn.functional.max_unpool1d": {f32},
    "nn.functional.max_unpool1d.grad": {f32},
    "nn.functional.max_unpool2d": {f32},
    "nn.functional.max_unpool2d.grad": {f32},
    "nn.functional.max_unpool3d": {f32},
    "nn.functional.max_unpool3d.grad": {f32},
    "nn.functional.mse_loss": {f32},
    "nn.functional.multilabel_soft_margin_loss": {f32},
    "nn.functional.normalize": {f32},
    "nn.functional.pad.circular": {f32},
    "nn.functional.pad.constant": {f32},
    "nn.functional.pad.reflect": {f32},
    "nn.functional.pairwise_distance": {f32},
    "nn.functional.pixel_shuffle": {f32},
    "nn.functional.pixel_unshuffle": {f32},
    "nn.functional.poisson_nll_loss": {f32},
    "nn.functional.prelu": {f32},
    "nn.functional.relu": {f32},
    "nn.functional.relu6": {f32},
    "nn.functional.rrelu": {f32},
    "nn.functional.selu": {f32},
    "nn.functional.silu": {f32},
    "nn.functional.softmin": {f32},
    "nn.functional.softmin.with_dtype": {f32},
    "nn.functional.softsign": {f32},
    "nn.functional.tanhshrink": {f32},
    "nn.functional.threshold": {f32},
    "nn.functional.triplet_margin_loss": {f32},
    "nn.functional.triplet_margin_with_distance_loss": {f32},
    "nn.functional.unfold": {f32},
    "nn.functional.upsample_bilinear": {f32},
    "nn.functional.upsample_nearest": {f32},
    "nonzero": {f32},
    "norm": {f32},
    "norm.fro": {f32},
    "norm.inf": {f32},
    "norm.nuc": {f32},
    "normal": {f32},
    "normal.number_mean": {f32},
    "ones": {f32},
    "ones_like": {f32},
    "outer": {f32},
    "pca_lowrank": {f32},
    "pinverse": {f32},
    "polar": {f32},
    "polygamma.polygamma_n_0": {f32},
    "polygamma.polygamma_n_1": {f32},
    "polygamma.polygamma_n_2": {f32},
    "polygamma.polygamma_n_3": {f32},
    "polygamma.polygamma_n_4": {f32},
    "quantile": {f32},
    "rand_like": {f32},
    "randint_like": {f32},
    "randn": {f32},
    "randn_like": {f32},
    "ravel": {f32},
    "reciprocal": {f32},
    "renorm": {f32},
    "repeat": {f32},
    "repeat_interleave": {f32},
    "reshape_as": {f32},
    "resize_": {f32},
    "resize_as_": {f32},
    "roll": {f32},
    "round": {f32},
    "round.decimals_0": {f32},
    "round.decimals_3": {f32},
    "round.decimals_neg_3": {f32},
    "rsqrt": {f32},
    "rsub": {f32},
    "scatter": {f32},
    "scatter_add": {f32},
    "scatter_reduce.amax": {f32},
    "scatter_reduce.amin": {f32},
    "scatter_reduce.mean": {f32},
    "scatter_reduce.prod": {f32},
    "scatter_reduce.sum": {f32},
    "searchsorted": {f32},
    "segment_reduce.lengths": {f32},
    "segment_reduce.offsets": {f32},
    "select_scatter": {f32},
    "sgn": {f32},
    "short": {f32},
    "sigmoid": {f32},
    "sign": {f32},
    "signbit": {f32},
    "sin": {f32},
    "slice_scatter": {f32},
    "softmax": {f32},
    "softmax.with_dtype": {f32},
    "sparse.sampled_addmm": {f32},
    "special.erfcx": {f32},
    "special.i0e": {f32},
    "special.i1": {f32},
    "special.i1e": {f32},
    "special.log_ndtr": {f32},
    "special.ndtr": {f32},
    "special.polygamma.special_polygamma_n_0": {f32},
    "special.xlog1py": {f32},
    "sqrt": {f32},
    "square": {f32},
    "squeeze": {f32},
    "std": {f32},
    "std_mean": {f32},
    "stft": {f32},
    "sub": {f32},
    "sum": {f32},
    "sum_to_size": {f32},
    "svd_lowrank": {f32},
    "symeig": {f32},
    "take": {f32},
    "take_along_dim": {f32},
    "tanh": {f32},
    "tensordot": {f32},
    "to": {f32},
    "to_sparse": {f32},
    "trapezoid": {f32},
    "trapz": {f32},
    "tril": {f32},
    "triu": {f32},
    "trunc": {f32},
    "unflatten": {f32},
    "unfold": {f32},
    "uniform": {f32},
    "unique": {f32},
    "unique_consecutive": {f32},
    "var": {f32},
    "var_mean": {f32},
    "view": {f32},
    "view_as": {f32},
    "view_as_complex": {f32},
    "vstack": {f32},
    "where": {f32},
    "xlogy": {f32},
    "zero_": {f32},
    "zeros": {f32},
    "zeros_like": {f32},
}

inductor_expected_failures["cuda"] = {
    "H": {f32},
    "T": {f32},
    "__getitem__": {f32},
    "__radd__": {f32},
    "__rdiv__": {f32},
    "__rmatmul__": {f32},
    "__rmod__": {f32},
    "__rmul__": {f32},
    "__rpow__": {f32},
    "__rsub__": {f32},
    "_masked.log_softmax": {f32},
    "_masked.normalize": {f32},
    "_masked.softmax": {f32},
    "_masked.softmin": {f32},
    "allclose": {f32},
    "angle": {f32},
    "arange": {f32},
    "argwhere": {f32},
    "as_strided": {f32},
    "as_strided_scatter": {f32},
    "bernoulli": {f32},
    "bfloat16": {f32},
    "bool": {f32},
    "byte": {f32},
    "cdist": {f32},
    "chalf": {f32},
    "char": {f32},
    "combinations": {f32},
    "complex": {f32},
    "contiguous": {f32},
    "corrcoef": {f32},
    "cov": {f32},
    "cumulative_trapezoid": {f32},
    "double": {f32},
    "einsum": {f32},
    "empty_like": {f32},
    "equal": {f32},
    "expand": {f32},
    "expand_as": {f32},
    "eye": {f32},
    "fft.fft": {f32},
    "fft.fft2": {f32},
    "fft.fftn": {f32},
    "fft.hfft": {f32},
    "fft.hfft2": {f32},
    "fft.hfftn": {f32},
    "fft.ifft": {f32},
    "fft.ifft2": {f32},
    "fft.ifftn": {f32},
    "fft.ihfft": {f32},
    "fft.ihfft2": {f32},
    "fft.ihfftn": {f32},
    "fft.irfft": {f32},
    "fft.irfft2": {f32},
    "fft.irfftn": {f32},
    "fft.rfft": {f32},
    "fft.rfft2": {f32},
    "fft.rfftn": {f32},
    "fill": {f32},
    "flip": {f32},
    "float": {f32},
    "gather": {f32},
    "gradient": {f32},
    "half": {f32},
    "index_put": {f32},
    "index_reduce": {f32},
    "int": {f32},
    "istft": {f32},
    "jiterator_2inputs_2outputs": {f32},
    "jiterator_4inputs_with_extra_args": {f32},
    "jiterator_binary": {f32},
    "jiterator_binary_return_by_ref": {f32},
    "jiterator_unary": {f32},
    "linalg.eig": {f32},
    "linalg.eigh": {f32},
    "linalg.eigvals": {f32},
    "linalg.ldl_factor": {f32},
    "linalg.lstsq": {f32},
    "linalg.lstsq.grad_oriented": {f32},
    "linalg.norm": {f32},
    "linalg.norm.subgradients_at_zero": {f32},
    "linalg.solve_triangular": {f32},
    "linspace": {f32},
    "log_softmax": {f32},
    "log_softmax.dtype": {f32},
    "logical_not": {f32},
    "long": {f32},
    "mH": {f32},
    "mT": {f32},
    "masked_scatter": {f32},
    "masked_select": {f32},
    "max.reduction_with_dim": {f32},
    "min.reduction_with_dim": {f32},
    "multinomial": {f32},
    "nanquantile": {f32},
    "narrow": {f32},
    "native_layer_norm": {f32},
    "new_empty": {f32},
    "new_empty_strided": {f32},
    "new_full": {f32},
    "new_ones": {f32},
    "new_zeros": {f32},
    "nn.functional._scaled_dot_product_attention": {f32},
    "nn.functional.batch_norm": {f32},
    "nn.functional.batch_norm.without_cudnn": {f32},
    "nn.functional.ctc_loss": {f32},
    "nn.functional.dropout": {f32},
    "nn.functional.dropout2d": {f32},
    "nn.functional.dropout3d": {f32},
    "nn.functional.embedding": {f32},
    "nn.functional.embedding_bag": {f32},
    "nn.functional.feature_alpha_dropout.with_train": {f32},
    "nn.functional.feature_alpha_dropout.without_train": {f32},
    "nn.functional.fractional_max_pool2d": {f32},
    "nn.functional.fractional_max_pool3d": {f32},
    "nn.functional.gaussian_nll_loss": {f32},
    "nn.functional.huber_loss": {f32},
    "nn.functional.max_pool1d": {f32},
    "nn.functional.max_pool2d": {f32},
    "nn.functional.max_pool3d": {f32},
    "nn.functional.max_unpool1d": {f32},
    "nn.functional.max_unpool1d.grad": {f32},
    "nn.functional.max_unpool2d": {f32},
    "nn.functional.max_unpool2d.grad": {f32},
    "nn.functional.max_unpool3d": {f32},
    "nn.functional.max_unpool3d.grad": {f32},
    "nn.functional.pixel_shuffle": {f32},
    "nn.functional.pixel_unshuffle": {f32},
    "nn.functional.prelu": {f32},
    "nn.functional.rrelu": {f32},
    "nn.functional.softmin": {f32},
    "nn.functional.softmin.with_dtype": {f32},
    "nn.functional.upsample_bilinear": {f32},
    "nn.functional.upsample_nearest": {f32},
    "nonzero": {f32},
    "normal": {f32},
    "normal.number_mean": {f32},
    "ones": {f32},
    "pca_lowrank": {f32},
    "pinverse": {f32},
    "polar": {f32},
    "polygamma.polygamma_n_0": {f32},
    "polygamma.polygamma_n_1": {f32},
    "polygamma.polygamma_n_2": {f32},
    "polygamma.polygamma_n_3": {f32},
    "polygamma.polygamma_n_4": {f32},
    "quantile": {f32},
    "rand_like": {f32},
    "randint_like": {f32},
    "randn": {f32},
    "randn_like": {f32},
    "renorm": {f32},
    "repeat": {f32},
    "repeat_interleave": {f32},
    "reshape_as": {f32},
    "resize_": {f32},
    "resize_as_": {f32},
    "scatter": {f32},
    "scatter_add": {f32},
    "scatter_reduce.amax": {f32},
    "scatter_reduce.amin": {f32},
    "scatter_reduce.mean": {f32},
    "scatter_reduce.prod": {f32},
    "scatter_reduce.sum": {f32},
    "segment_reduce.lengths": {f32},
    "segment_reduce.offsets": {f32},
    "sgn": {f32},
    "short": {f32},
    "softmax": {f32},
    "softmax.with_dtype": {f32},
    "special.polygamma.special_polygamma_n_0": {f32},
    "squeeze": {f32},
    "std_mean": {f32},
    "stft": {f32},
    "sum_to_size": {f32},
    "svd_lowrank": {f32},
    "symeig": {f32},
    "tensordot": {f32},
    "to": {f32},
    "to_sparse": {f32},
    "unflatten": {f32},
    "unfold": {f32},
    "uniform": {f32},
    "unique": {f32},
    "unique_consecutive": {f32},
    "view": {f32},
    "view_as": {f32},
    "view_as_complex": {f32},
    "where": {f32},
    "zero_": {f32},
    "zeros": {f32},
}


class TestInductorOpInfo(TestCase):

    @patch.object(torchinductor.config.triton, "cudagraphs", False)
    @patch("torchdynamo.config.raise_on_backend_error", True)
    def check_model(
        self,
        model,
        example_inputs,
        tol=1e-4,
        *,
        check_lowp=True,
        exact_dtype=True,
    ):
        torchdynamo.reset()

        # check_lowp is ignored here, it's kept just to be able to call `common` with extra arg
        has_lowp_args = False

        def upcast_fn(x):
            nonlocal has_lowp_args
            if isinstance(x, torch.Tensor) and (
                x.dtype == torch.float16 or x.dtype == torch.bfloat16
            ):
                has_lowp_args = True
                return x.float()
            else:
                return x

        upcasted_inputs = list(map(upcast_fn, example_inputs))
        if has_lowp_args:
            if hasattr(model, "to"):
                model = model.to(torch.float)
        torch.manual_seed(0)
        correct = model(*upcasted_inputs)
        # downcast the model back if needed
        if has_lowp_args:
            if hasattr(model, "to"):
                model = model.to(torch.half)

        torchinductor.metrics.reset()

        @torchdynamo.optimize_assert(compile_fx)
        def run(*ex):
            return model(*ex)

        torch.manual_seed(0)
        actual = run(*example_inputs)

        assert type(actual) == type(correct)
        correct_flat, correct_spec = tree_flatten(correct)
        actual_flat, _ = tree_flatten(actual)
        correct_flat = tuple(
            y.to(x.dtype)
            if isinstance(y, torch.Tensor) and y.dtype.is_floating_point
            else y
            for x, y in zip(actual_flat, correct_flat)
        )
        correct = tree_unflatten(correct_flat, correct_spec)

        # print(correct)
        # print(actual)
        # print(correct - actual)
        self.assertTrue(
            same(actual, correct, tol=tol, equal_nan=True, exact_dtype=exact_dtype)
        )


    @patch.object(torchinductor.config.triton, "cudagraphs", False)
    def check_model_cuda(
        self, model, example_inputs, *, check_lowp=True, exact_dtype=True
    ):
        if hasattr(model, "to"):
            model = model.to("cuda")

        def copy_fn(x):
            # preserve strides of the input on the device
            if not isinstance(x, torch.Tensor):
                return x
            return torch.empty_strided(
                x.size(), x.stride(), device="cuda", dtype=x.dtype
            ).copy_(x)

        example_inputs = tuple(copy_fn(x) for x in example_inputs)
        self.check_model(model, example_inputs, exact_dtype=exact_dtype)

        # if check_lowp:

        #     def downcast_fn(x):
        #         if not isinstance(x, torch.Tensor) or not x.dtype == torch.float:
        #             return x
        #         return torch.empty_strided(
        #             x.size(), x.stride(), device="cuda", dtype=torch.half
        #         ).copy_(x)

        #     example_inputs = list(map(downcast_fn, example_inputs))
        #     if hasattr(model, "to"):
        #         model = model.to(torch.half)
        #     self.check_model(model, example_inputs, 2e-3, exact_dtype=exact_dtype)


    @onlyNativeDeviceTypes
    @suppress_warnings
    @skipCUDAMemoryLeakCheckIf(True)    # inductor kernels failing this test intermittently
    @_ops(op_db)
    def test_comprehensive(self, device, dtype, op):

        # breakpoint()
        op_name = op.name
        if op.variant_test_name:
            op_name += f".{op.variant_test_name}"

        device_type = torch.device(device).type

        if (device_type, dtype, op_name) in EXCLUDE_SET or \
           (None, dtype, op_name) in EXCLUDE_SET or \
           (None, None, op_name) in EXCLUDE_SET:
            self.skipTest(f"{op_name} in {dtype} not supported")


        test_expect = TestExpect.SUCCESS

        if dtype in inductor_expected_failures[device_type].get(op_name, set()):
            test_expect = TestExpect.XFAILURE

        func = op.get_op()


        def fn(*args, **kwargs):
            return func(*args, **kwargs)

        requires_grad = (
            op.supports_autograd
            and dtype in op.supported_backward_dtypes(torch.device(device).type)
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
            # breakpoint()
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
                raise RuntimeError(f"unexpected success {op_name}, {dtype}, {device_type}")




instantiate_device_type_tests(TestInductorOpInfo, globals())


if __name__ == "__main__":
    run_tests()
