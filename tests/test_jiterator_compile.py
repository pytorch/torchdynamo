#!/usr/bin/env pytest
import importlib
import unittest

import torch
from torch.nn import functional as F

try:
    importlib.import_module("functorch")
    from torchinductor import config
except (ImportError, ModuleNotFoundError):
    raise unittest.SkipTest("requires functorch")

from tests.test_torchinductor import (
    TestCase,
    SweepInputs2,
    InputGen,
    CommonTemplate,
    check_model_cuda,
)

aten = torch.ops.aten

config.cuda_backend = "Jiteraor"

HAS_CUDA = torch.cuda.is_available()


if HAS_CUDA:

    def check_model_jiterator(self: TestCase, model, example_inputs):
        sample_outputs = model(*example_inputs)

        if hasattr(model, "to"):
            model = model.to("cuda")

        # Jiterator currently only supports single output
        # TODO: remove this workaround when jiterator supports multiple outputs
        for i in range(len(sample_outputs)):

            def single_output_fn(*args, **kwargs):
                return model(*args, **kwargs)[i]

            check_model_cuda(self, single_output_fn, example_inputs)

    class SweepInputsJiteratorTest(SweepInputs2, TestCase):
        gen = InputGen(10, "cuda")

    SweepInputsJiteratorTest.populate(config.cuda_backend)

    class JiteratorTests(TestCase):
        common = check_model_jiterator
        device = "cuda"

    # Jiteratort currently only support elementwise ops
    # Any tests that uses non-elementwise ops should be skipped here
    exclude_tests = {
        "test_addmm",
        "test_alexnet_prefix",
        "test_arange",
        "test_batch_norm_2d",
        "test_bmm",
        "test_cat",
        "test_embedding",
        "test_gather",
        "test_glu",
        "test_log_softmax",
        "test_max_pool2d1",
        "test_max_pool2d2",
        "test_max_pool2d3",
        "test_max_pool2d4",
        "test_mean",
        "test_min_max_reduction",
        "test_permute",
        "test_pow",  # TODO: should work with jiterator, failing during decomp
        "test_repeat",
        "test_slice1",
        "test_slice2",
        "test_softmax",
        "test_split",
        "test_squeeze1",
        "test_squeeze2",
        "test_std",
        "test_sum1",
        "test_sum2",
        "test_sum3",
        "test_sum4",
        "test_sum5",
        "test_sum_keepdims",
        "test_transpose",
        "test_unsqueeze",
        "test_views1",
        "test_views2",
        "test_avg_pool2d1",
        "test_avg_pool2d2",
        "test_avg_pool2d3",
        "test_avg_pool2d4",
        "test_expand",
        "test_full_like",
        "test_index",
        "test_linear2",
        "test_linspace",
        "test_logsumexp",
        "test_tensor1",
        "test_tensor2",
        "test_tensor3",
        "test_to_device",
        "test_to_dtype",
        "test_unbind",
        "test_zeros",
    }

    CommonTemplate.install(JiteratorTests, config.cuda_backend, exclude_tests)
