#!/usr/bin/env python
import logging
import re
import warnings

import torch
from common import BenchmarkRunner
from common import main
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForSeq2SeqLM
from transformers import BertConfig

import torchdynamo
from torchdynamo.testing import collect_results

# from transformers import ReformerConfig
# from transformers import XLNetConfig
# from transformers import XLNetModel


# We are primarily interested in tf32 datatype
torch.backends.cuda.matmul.allow_tf32 = True

SKIP = {}

ALL_MODELS = {
    "bert": (BertConfig(), AutoModelForMaskedLM, (4, 512), []),
    "albert": (
        AutoConfig.from_pretrained("albert-base-v2"),
        AutoModelForMaskedLM,
        (8, 512),
        [],
    ),
    "gpt2": (AutoConfig.from_pretrained("gpt2"), AutoModelForCausalLM, (4, 512), []),
    # "allenai/longformer-base-4096":
    # (
    #     AutoConfig.from_pretrained("allenai/longformer-base-4096"),  # Longformer is not good for fx2trt
    #     AutoModelForMaskedLM,
    #     (2, 1024),
    #     [torch.bfloat16], # trilu not implemented for bfloat16
    # ),
    "t5-small": (
        AutoConfig.from_pretrained("t5-small"),
        AutoModelForSeq2SeqLM,
        (1, 1024),
        [torch.bfloat16],
    ),
    # "reformer": (ReformerConfig(), AutoModelForMaskedLM, (8, 4096), []), # Reformer is not good for fx2trt
    "distilbert-base-uncased": (
        AutoConfig.from_pretrained("distilbert-base-uncased"),
        AutoModelForMaskedLM,
        (8, 512),
        [],
    ),
    "roberta-base": (
        AutoConfig.from_pretrained("roberta-base"),
        AutoModelForMaskedLM,
        (16, 512),
        [],
    ),
    # "bigbird": (
    #     BigBirdConfig(attention_type="block_sparse"),
    #     AutoModelForMaskedLM,
    #     (2, 1024),
    #     [torch.bfloat16, torch.float16],
    # ),  # Currently quite slow - needs investigation
    "distilgpt2": (
        AutoConfig.from_pretrained("distilgpt2"),
        AutoModelForCausalLM,
        (16, 512),
        [],
    ),
    "google/electra-base-discriminator": (
        AutoConfig.from_pretrained("google/electra-base-discriminator"),
        AutoModelForMaskedLM,
        (8, 512),
        [],
    ),
    "google/fnet-base": (
        AutoConfig.from_pretrained("google/fnet-base"),
        AutoModelForMaskedLM,
        (8, 512),
        [torch.bfloat16, torch.float16],
    ),
    "YituTech/conv-bert-base": (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),
        AutoModelForMaskedLM,
        (8, 512),
        [torch.bfloat16],
    ),
    "google/mobilebert-uncased": (
        AutoConfig.from_pretrained("google/mobilebert-uncased"),
        AutoModelForMaskedLM,
        (4, 512),
        [],
    ),
    "camembert-base": (
        AutoConfig.from_pretrained("camembert-base"),
        AutoModelForMaskedLM,
        (8, 512),
        [],
    ),
    "microsoft/layoutlm-base-uncased": (
        AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased"),
        AutoModelForMaskedLM,
        (8, 512),
        [],
    ),
}


class HuggingfaceRunner(BenchmarkRunner):
    def __init__(self):
        super(HuggingfaceRunner, self).__init__()

    @property
    def skip_models(self):
        return set()

    @property
    def slow_models(self):
        return set()

    @property
    def very_slow_models(self):
        return set()

    @property
    def non_deterministic_models(self):
        return set()

    @property
    def skip_not_suitable_for_training_models(self):
        return set()

    @property
    def failing_python_key_models(self):
        return set()

    @property
    def failing_torchinductor_models(self):
        return set()

    @property
    def failing_fx2trt_models(self):
        return set()

    @property
    def failing_dynamic_shape_models(self):
        return set()

    def load_model(self, device, model_name, is_training, use_eval_mode):
        dtype = torch.float32
        config, model_type, input_size, not_supported_dtypes = ALL_MODELS[model_name]
        if dtype in not_supported_dtypes:
            raise NotImplementedError()

        model = model_type.from_config(config).to(device, dtype=dtype)
        model_name = type(model).__name__

        # So we can check for correct gradients without eliminating the dropout computation
        for attr in dir(config):
            if "drop" in attr and isinstance(getattr(config, attr), float):
                setattr(config, attr, 1e-30)

        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        # Prepare inputs
        input_ids = torch.randint(0, config.vocab_size, input_size).to(device)
        decoder_ids = torch.randint(0, config.vocab_size, input_size).to(device)
        example_inputs = {"input_ids": input_ids, "labels": decoder_ids}
        return device, model_name, model, example_inputs

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
        for model_name in ALL_MODELS:
            if (
                not re.search("|".join(args.filter), model_name, re.I)
                or re.search("|".join(args.exclude), model_name, re.I)
                or model_name in SKIP
            ):
                continue

            yield model_name

    def pick_grad(self, name, is_training):
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def set_tolerance(self, is_training, current_device, name):
        return 1e-3

    def compute_loss(self, pred):
        return pred[0]

    @torchdynamo.skip
    def forward_pass(self, mod, inputs, collect_outputs=True):
        return mod(**inputs)

    @torchdynamo.skip
    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        mod.zero_grad(True)
        pred = mod(**inputs)
        loss = self.compute_loss(pred)
        loss.backward()
        if collect_outputs:
            return collect_results(mod, pred, loss, inputs)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(HuggingfaceRunner())
