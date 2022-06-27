#!/usr/bin/env python
import gc
import importlib
import logging
import os
import re
import sys
import warnings
from os.path import abspath
from os.path import exists

import torch
from common import BenchmarkRunner
from common import main

from torchdynamo.testing import collect_results
from torchdynamo.testing import reduce_to_scalar_loss
from torchdynamo.utils import clone_inputs

# We are primarily interested in tf32 datatype
torch.backends.cuda.matmul.allow_tf32 = True

os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
for torchbench_dir in (
    "../torchbenchmark",
    "../torchbench",
    "../benchmark",
    "../../torchbenchmark",
    "../../torchbench",
    "../../benchmark",
):
    if exists(torchbench_dir):
        break
assert exists(torchbench_dir), "../../torchbenchmark does not exist"
original_dir = abspath(os.getcwd())
torchbench_dir = abspath(torchbench_dir)
os.chdir(torchbench_dir)
sys.path.append(torchbench_dir)


# Some models have large dataset that doesn't fit in memory. Lower the batch
# size to test the accuracy.
USE_SMALL_BATCH_SIZE = {
    "demucs": 4,
    "densenet121": 4,
    "hf_Reformer": 4,
    "timm_efficientdet": 1,
}


# Additional models that are skipped in training
SKIP_TRAIN = {
    # not designed for training
    "pyhpc_equation_of_state",
    "pyhpc_isoneutral_mixing",
    "pyhpc_turbulent_kinetic_energy",
    # Unusual training setup
    "opacus_cifar10",
    "maml",
}

# Some models have bad train dataset. We read eval dataset.
# yolov3 - seems to have different number of inputs between eval and train
# timm_efficientdet - loader only exists for eval mode.
ONLY_EVAL_DATASET = {"yolov3", "timm_efficientdet"}


# These models support only train mode. So accuracy checking can't be done in
# eval mode.
ONLY_TRAINING_MODE = {"tts_angular", "tacotron2", "demucs"}


# Need lower tolerance on GPU. GPU kernels have non deterministic kernels for these models.
REQUIRE_HIGHER_TOLERANCE = {
    "alexnet",
    "attention_is_all_you_need_pytorch",
    "densenet121",
    "hf_Albert",
    "vgg16",
    "mobilenet_v3_large",
    "nvidia_deeprecommender",
    "timm_efficientdet",
    "vision_maskrcnn",
}

SKIP = {
    # https://github.com/pytorch/torchdynamo/issues/101
    "detectron2_maskrcnn",
    # https://github.com/pytorch/torchdynamo/issues/145
    "fambench_xlmr",
}


# These models need >1e-3 tolerance
REQUIRE_EVEN_HIGHER_TOLERANCE = {
    "soft_actor_critic",
    "tacotron2",
}


# non-deterministic output / cant check correctness
NONDETERMINISTIC = set()


# These benchmarks took >600s on an i9-11900K CPU
VERY_SLOW_BENCHMARKS = {
    "hf_BigBird",  # 3339s
    "hf_Longformer",  # 3062s
    "hf_T5",  # 930s
}


# These benchmarks took >60s on an i9-11900K CPU
SLOW_BENCHMARKS = {
    *VERY_SLOW_BENCHMARKS,
    "BERT_pytorch",  # 137s
    "demucs",  # 116s
    "fastNLP_Bert",  # 242s
    "hf_Albert",  # 221s
    "hf_Bart",  # 400s
    "hf_Bert",  # 334s
    "hf_DistilBert",  # 187s
    "hf_GPT2",  # 470s
    "hf_Reformer",  # 141s
    "speech_transformer",  # 317s
    "vision_maskrcnn",  # 99s
}

# https://github.com/pytorch/torchdynamo/issues/331
PYTHON_KEY_NOT_YET_WORKING = {
    # RuntimeError: expected scalar type Half but found Float
    "hf_BigBird",
    # AttributeError: 'int' object has no attribute 'proxy'
    "moco",
    # AttributeError: 'Tensor' object has no attribute 'proxy'
    "speech_transformer",
    # torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
    "tacotron2",
    # requires training mode
    "maml",
}


# https://github.com/pytorch/torchdynamo/issues/332
TORCHINDUCTOR_NOT_YET_WORKING = {
    *PYTHON_KEY_NOT_YET_WORKING,
    # Crash with no warning message
    "Super_SloMo",
    "fastNLP_Bert",
    # RuntimeError: CUDA: Error- invalid value
    "dlrm",
    "vision_maskrcnn",
    # LLVM ERROR: Broken function found, compilation aborted!
    # torch.randn missing
    "hf_Reformer",
    # as_strided issue
    "hf_Longformer",
    # out of memory
    "timm_efficientdet",
}


TRT_NOT_YET_WORKING = {
    "alexnet",
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "mnasnet1_0",
    "squeezenet1_1",
    "shufflenetv2_x1_0",
    "vgg16",
    "resnext50_32x4d",
}


DYNAMIC_SHAPES_NOT_YET_WORKING = {
    "demucs",
    "timm_nfnet",
}

SKIP = {
    # https://github.com/pytorch/torchdynamo/issues/101
    "detectron2_maskrcnn",
    # https://github.com/pytorch/torchdynamo/issues/145
    "fambench_xlmr",
}


class TorchBenchmarkRunner(BenchmarkRunner):
    @property
    def skip_models(self):
        return SKIP

    @property
    def slow_models(self):
        return SLOW_BENCHMARKS

    @property
    def very_slow_models(self):
        return VERY_SLOW_BENCHMARKS

    @property
    def non_deterministic_models(self):
        return NONDETERMINISTIC

    @property
    def skip_not_suitable_for_training_models(self):
        return SKIP_TRAIN

    @property
    def failing_python_key_models(self):
        return PYTHON_KEY_NOT_YET_WORKING

    @property
    def failing_torchinductor_models(self):
        return TORCHINDUCTOR_NOT_YET_WORKING

    @property
    def failing_fx2trt_models(self):
        return TRT_NOT_YET_WORKING

    @property
    def failing_dynamic_shape_models(self):
        return DYNAMIC_SHAPES_NOT_YET_WORKING

    def load_model(self, device, model_name, is_training, use_eval_mode):
        module = importlib.import_module(f"torchbenchmark.models.{model_name}")
        benchmark_cls = getattr(module, "Model", None)
        if not hasattr(benchmark_cls, "name"):
            benchmark_cls.name = model_name
        batch_size = None
        if is_training and model_name in USE_SMALL_BATCH_SIZE:
            batch_size = USE_SMALL_BATCH_SIZE[model_name]

        if is_training and model_name not in ONLY_EVAL_DATASET:
            benchmark = benchmark_cls(
                test="train", device=device, jit=False, batch_size=batch_size
            )
        else:
            benchmark = benchmark_cls(
                test="eval", device=device, jit=False, batch_size=batch_size
            )
        model, example_inputs = benchmark.get_module()

        # Models that must be in train mode while training
        if is_training and (not use_eval_mode or model_name in ONLY_TRAINING_MODE):
            model.train()
        else:
            model.eval()
        gc.collect()
        # global current_name, current_device
        # current_device = device
        # current_name = benchmark.name
        return device, benchmark.name, model, example_inputs

    def iter_models(self, args):
        for model_name in self.iter_model_names(args):
            for device in args.devices:
                try:
                    yield self.load_model(
                        device, model_name, args.training, args.use_eval_mode
                    )
                except NotImplementedError:
                    continue  # bad benchmark implementation

    def iter_model_names(self, args):
        from torchbenchmark import _list_model_paths

        for model_path in _list_model_paths():
            model_name = os.path.basename(model_path)
            if (
                not re.search("|".join(args.filter), model_name, re.I)
                or re.search("|".join(args.exclude), model_name, re.I)
                or model_name in SKIP
            ):
                continue

            yield model_name

    def pick_grad(self, name, is_training):
        if is_training or name in ("maml",):
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def get_tolerance(self, is_training, current_device, name):
        tolerance = 1e-4
        # Increase the tolerance for torch allclose
        if (
            is_training
            and current_device == "cuda"
            and name in REQUIRE_HIGHER_TOLERANCE
        ):
            tolerance = 1e-3
        elif (
            is_training
            and current_device == "cuda"
            and name in REQUIRE_EVEN_HIGHER_TOLERANCE
        ):
            tolerance = 8 * 1e-3
        return tolerance

    def compute_loss(self, pred):
        return reduce_to_scalar_loss(pred)

    def forward_pass(self, mod, inputs, collect_outputs=True):
        return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        mod.zero_grad(True)
        with self.autocast():
            pred = mod(*cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return None


if __name__ == "__main__":

    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TorchBenchmarkRunner(), original_dir)
