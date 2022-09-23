from collections import defaultdict
from enum import Enum
import os
import atexit

import torch
from torch.utils._python_dispatch import enable_torch_dispatch_mode

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


from functools import partial
import unittest

import importlib

from .test_torchinductor import check_model_cuda, check_model

try:
    importlib.import_module("functorch")

    from torch._decomp import get_decompositions

    from torchinductor import config
    from torchinductor.compile_fx import compile_fx

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
    ("cpu", torch.float32, "lu_unpack"),   # invalid pointer

    ("cuda", torch.float64, "_masked.logsumexp"),       # Call parameter type does not match function signature!
    ("cuda", torch.float64, "cos"),           # Call parameter type does not match function signature!
    ("cuda", torch.float64, "logsumexp"),           # Both operands to a binary operator are not of the same type!
    ("cuda", torch.float64, "nn.functional.binary_cross_entropy_with_logits"),           # Both operands to a binary operator are not of the same type!
    ("cuda", torch.float64, "special.log_ndtr"),           # Both operands to a binary operator are not of the same type!
    ("cuda", torch.float32, "isclose"),           # LLVM ERROR
    ("cuda", torch.float32, "isfinite"),           # LLVM ERROR
    ("cuda", torch.float32, "nan_to_num"),           # LLVM ERROR
    ("cuda", torch.float32, "lu_unpack"),   # RuntimeError: CUDA error
}

seen_succeeded = defaultdict(dict)
seen_failed = defaultdict(dict)

failed_reasons = defaultdict(set)

def print_seen():
    expected_failures = defaultdict(list)
    skips = defaultdict(list)

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
    "_masked.log_softmax": {f32},
    "_masked.normalize": {f32},
    "_masked.softmax": {f32},
    "_masked.softmin": {f32},
    "allclose": {f32},
    "angle": {f32},
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
    "cholesky_inverse": {f32},
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
    "linalg.eig": {f32},
    "linalg.eigh": {f32},
    "linalg.eigvals": {f32},
    "linalg.ldl_factor": {f32},
    "linalg.lstsq": {f32},
    "linalg.lstsq.grad_oriented": {f32},
    "linalg.norm.subgradients_at_zero": {f32},
    "linalg.solve_triangular": {f32},
    "linspace": {f32},
    "log_softmax": {f32},
    "log_softmax.dtype": {f32},
    "logdet": {f32},
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
    "sparse.sampled_addmm": {f32},
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
}

class TestInductorOpInfo(TestCase):
    check_model = check_model
    check_model_cuda = check_model_cuda

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
