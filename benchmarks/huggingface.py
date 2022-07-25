#!/usr/bin/env python
import importlib
import logging
import os
import re
import subprocess
import sys
import warnings

import torch
from common import BenchmarkRunner
from common import main
from transformers import ReformerConfig

import torchdynamo
from torchdynamo.testing import collect_results
from torchdynamo.utils import clone_inputs


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    importlib.import_module("transformers")
except ModuleNotFoundError:
    print("Installing HuggingFace Transformers...")
    pip_install("git+https://github.com/huggingface/transformers.git#egg=transformers")
finally:
    from transformers import AlbertForPreTraining
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM
    from transformers import AutoModelForMaskedLM
    from transformers import AutoModelForSeq2SeqLM
    from transformers import BigBirdConfig
    from transformers import BlenderbotForConditionalGeneration
    from transformers import BlenderbotModel
    from transformers import BlenderbotSmallForConditionalGeneration
    from transformers import BlenderbotSmallModel
    from transformers import CLIPModel
    from transformers import CLIPVisionModel
    from transformers import ElectraForPreTraining
    from transformers import GPT2ForSequenceClassification
    from transformers import GPTJForSequenceClassification
    from transformers import GPTNeoForSequenceClassification
    from transformers import HubertForSequenceClassification
    from transformers import LxmertForPreTraining
    from transformers import LxmertForQuestionAnswering
    from transformers import MarianForCausalLM
    from transformers import MarianModel
    from transformers import MarianMTModel
    from transformers import PegasusForConditionalGeneration
    from transformers import PegasusModel
    from transformers import SwinForImageClassification
    from transformers import SwinForMaskedImageModeling
    from transformers import SwinModel
    from transformers import ViTForImageClassification
    from transformers import ViTForMaskedImageModeling
    from transformers import ViTModel
    from transformers import XLNetConfig
    from transformers import XLNetLMHeadModel


HF_FX_SUPPORTED_MODELS = dict()

REFRESH_MODEL_NAMES = False

USE_HALF_BATCH_SIZE = True

filename = "huggingface_models_list.txt"
if os.path.exists("benchmarks"):
    filename = "benchmarks/" + filename
with open(filename, "r") as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(",")
        batch_size = int(batch_size)
        HF_FX_SUPPORTED_MODELS[model_name] = batch_size


SKIP = {
    # Difficult to run and compare
    "Reformer",
    # Fails deepcopy
    "BlenderbotForCausalLM",
    "BlenderbotForConditionalGeneration",
    "GPTJForCausalLM",
    "GPTJForQuestionAnswering",
}

# TODO - Fails even after fake tensors
USE_SMALL_BATCH_SIZE = {
    "AlbertForPreTraining": 4,
    "XLNetLMHeadModel": 8,
}


def get_module_cls_by_model_name(model_cls_name):
    _module_by_model_name = {
        "Speech2Text2Decoder": "transformers.models.speech_to_text_2.modeling_speech_to_text_2",
        "TrOCRDecoder": "transformers.models.trocr.modeling_trocr",
    }
    module_name = _module_by_model_name.get(model_cls_name, "transformers")
    module = importlib.import_module(module_name)
    return getattr(module, model_cls_name)


def get_sequence_length(model_cls, model_name):
    if model_name.startswith(("Bert", "Roberta", "Blenderbot")):
        seq_length = 128
    elif model_name.startswith(("GPT2", "Bart", "T5")):
        seq_length = 1024
    elif model_name in ("AllenaiLongformerBase", "BigBird"):
        seq_length = 1024
    elif "Reformer" in model_name:
        seq_length = 4096
    elif model_name.startswith(
        ("Albert", "Deberta", "Layout", "Electra", "XLNet")
    ) or model_name in ("DistillGPT2", "GoogleFnet", "YituTechConvBert", "CamemBert"):
        seq_length = 512
    else:
        logging.warn(
            f"Sequence Length not defined for {model_name}. Choosing 512 arbitrarily"
        )
        seq_length = 512
    return seq_length


def generate_inputs_for_model(
    model_cls, model, model_name, bs, device, include_loss_args=False
):
    # TODO - Check if following values are representative
    num_choices = 3
    num_visual_features = 42
    seq_length = get_sequence_length(model_cls, model_name)
    vocab_size = model.config.vocab_size
    if model_name.endswith("MultipleChoice"):
        input = rand_int_tensor(device, 0, vocab_size, (bs, num_choices, seq_length))
    elif model_name.startswith("Roberta"):
        input = rand_int_tensor(device, 0, 1, (bs, seq_length))
    else:
        input = rand_int_tensor(device, 0, vocab_size, (bs, seq_length))

    if "Bart" in model_name:
        input[:, -1] = model.config.eos_token_id

    input_dict = {"input_ids": input}

    if (
        model_name.startswith("T5")
        or model_name.startswith("M2M100")
        or model_name.startswith("MT5")
        or model_cls
        in [
            BlenderbotModel,
            BlenderbotSmallModel,
            BlenderbotForConditionalGeneration,
            BlenderbotSmallForConditionalGeneration,
            PegasusModel,
            PegasusForConditionalGeneration,
            MarianModel,
            MarianMTModel,
        ]
    ):
        input_dict.update({"decoder_input_ids": input})

    if model_name.startswith("Lxmert"):
        visual_feat_dim, visual_pos_dim = (
            model.config.visual_feat_dim,
            model.config.visual_pos_dim,
        )
        input_dict.update(
            {
                "visual_feats": torch.randn(bs, num_visual_features, visual_feat_dim),
                "visual_pos": torch.randn(bs, num_visual_features, visual_pos_dim),
            }
        )

    if include_loss_args:
        if model_name.endswith("PreTraining"):
            if model_cls in [ElectraForPreTraining, LxmertForPreTraining]:
                input_dict.update(
                    {
                        "labels": rand_int_tensor(device, 0, 1, (bs, seq_length)),
                    }
                )
            else:
                label_name = (
                    "sentence_order_label"
                    if model_cls in [AlbertForPreTraining]
                    else "next_sentence_label"
                )
                input_dict.update(
                    {
                        "labels": rand_int_tensor(
                            device, 0, vocab_size, (bs, seq_length)
                        ),
                        label_name: rand_int_tensor(device, 0, 1, (bs,)),
                    }
                )
        elif model_name.endswith("QuestionAnswering"):
            input_dict.update(
                {
                    "start_positions": rand_int_tensor(device, 0, seq_length, (bs,)),
                    "end_positions": rand_int_tensor(device, 0, seq_length, (bs,)),
                }
            )
        elif (
            model_name.endswith("MaskedLM")
            or model_name.endswith("HeadModel")
            or model_name.endswith("CausalLM")
            or model_name.endswith("DoubleHeadsModel")
        ):
            input_dict.update(
                {
                    "labels": rand_int_tensor(device, 0, vocab_size, (bs, seq_length)),
                }
            )
        elif model_name.endswith("TokenClassification"):
            input_dict.update(
                {
                    "labels": rand_int_tensor(
                        device, 0, model.config.num_labels - 1, (bs, seq_length)
                    ),
                }
            )
        elif model_name.endswith("MultipleChoice"):
            input_dict.update(
                {
                    "labels": rand_int_tensor(device, 0, num_choices, (bs,)),
                }
            )
        elif model_name.endswith("SequenceClassification"):
            input_dict.update(
                {
                    "labels": rand_int_tensor(
                        device, 0, model.config.num_labels - 1, (bs,)
                    ),
                }
            )
        elif model_name.endswith("NextSentencePrediction"):
            input_dict.update(
                {
                    "labels": rand_int_tensor(device, 0, 1, (bs,)),
                }
            )
        elif model_name.endswith("ForConditionalGeneration"):
            input_dict.update(
                {
                    "labels": rand_int_tensor(
                        device, 0, vocab_size - 1, (bs, seq_length)
                    ),
                }
            )
        else:
            raise NotImplementedError(
                f"Class {model_name} unsupported for training test "
            )

    return input_dict


def rand_int_tensor(device, low, high, shape):
    return torch.randint(
        low,
        high,
        shape,
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


EXTRA_MODELS = {
    "XLNetLMHeadModel": (
        XLNetConfig.from_pretrained("xlnet-large-cased"),
        XLNetLMHeadModel,
        16,
    ),
    "AllenaiLongformerBase": (
        AutoConfig.from_pretrained("allenai/longformer-base-4096"),
        AutoModelForMaskedLM,
        2,
    ),
    "Reformer": (
        ReformerConfig(),
        AutoModelForMaskedLM,
        8,
    ),
    "T5Small": (
        AutoConfig.from_pretrained("t5-small"),
        AutoModelForSeq2SeqLM,
        1,
    ),
    "BigBird": (
        BigBirdConfig(attention_type="block_sparse"),
        AutoModelForMaskedLM,
        2,
    ),
    "DistillGPT2": (
        AutoConfig.from_pretrained("distilgpt2"),
        AutoModelForCausalLM,
        16,
    ),
    "GoogleFnet": (
        AutoConfig.from_pretrained("google/fnet-base"),
        AutoModelForMaskedLM,
        8,
    ),
    "YituTechConvBert": (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),
        AutoModelForMaskedLM,
        8,
    ),
    "CamemBert": (
        AutoConfig.from_pretrained("camembert-base"),
        AutoModelForMaskedLM,
        8,
    ),
}


class HuggingfaceRunner(BenchmarkRunner):
    def __init__(self):
        super(HuggingfaceRunner, self).__init__()

    def load_model(
        self,
        device,
        model_name,
        is_training,
        use_eval_mode,
        batch_size=None,
        dynamic_shapes=False,
    ):
        dtype = torch.float32
        if model_name not in EXTRA_MODELS:
            model_cls = get_module_cls_by_model_name(model_name)
            config_cls = model_cls.config_class
            config = config_cls()

            # NB: some models need a pad token defined to handle BS > 1
            if (
                model_cls
                in [
                    GPT2ForSequenceClassification,
                    GPTNeoForSequenceClassification,
                    GPTJForSequenceClassification,
                ]
                or model_cls.__name__.startswith("Roberta")
                or model_cls.__name__.startswith("Marian")
            ):
                config.pad_token_id = 0

            batch_size_default = HF_FX_SUPPORTED_MODELS[model_name]
        else:
            config, model_cls, batch_size_default = EXTRA_MODELS[model_name]

        if "auto" in model_cls.__module__:
            # Handle auto classes
            model = model_cls.from_config(config).to(device, dtype=dtype)
        else:
            model = model_cls(config).to(device, dtype=dtype)

        batch_size = batch_size or batch_size_default

        if model_name in USE_SMALL_BATCH_SIZE:
            batch_size = USE_SMALL_BATCH_SIZE[model_name]
        elif USE_HALF_BATCH_SIZE:
            batch_size = int(batch_size / 2)

        example_inputs = generate_inputs_for_model(
            model_cls, model, model_name, batch_size, device, include_loss_args=True
        )

        # So we can check for correct gradients without eliminating the dropout computation
        for attr in dir(config):
            if "drop" in attr and isinstance(getattr(config, attr), float):
                setattr(config, attr, 1e-30)

        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        return device, model_name, model, example_inputs

    def iter_models(self, args):
        for model_name in self.iter_model_names(args):
            for device in args.devices:
                try:
                    yield self.load_model(
                        device,
                        model_name,
                        args.training,
                        args.use_eval_mode,
                        args.batch_size,
                    )
                except NotImplementedError:
                    continue  # bad benchmark implementation

    def iter_model_names(self, args):
        model_names = list(HF_FX_SUPPORTED_MODELS.keys()) + list(EXTRA_MODELS.keys())
        model_names = sorted(model_names)
        for model_name in model_names:
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

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        cosine = self.args.cosine
        if is_training:
            return 1e-2, cosine
        return 1e-3, cosine

    def compute_loss(self, pred):
        return pred[0]

    @torchdynamo.skip
    def forward_pass(self, mod, inputs, collect_outputs=True):
        return mod(**inputs)

    @torchdynamo.skip
    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        mod.zero_grad(True)
        with self.autocast():
            pred = mod(**cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return None


def refresh_model_names():
    import transformers.utils.fx as hf_fx

    family = dict()
    lm_seen = set()
    family_seen = set()
    for cls_name in hf_fx._SUPPORTED_MODELS:

        if "For" not in cls_name:
            continue

        model_cls = get_module_cls_by_model_name(cls_name)

        # TODO: AttributeError: '*Config' object has no attribute 'vocab_size'
        if model_cls in [
            CLIPModel,
            CLIPVisionModel,
            SwinForImageClassification,
            SwinForImageClassification,
            SwinForMaskedImageModeling,
            SwinModel,
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
        ]:
            continue

        # TODO: AssertionError: Padding_idx must be within num_embeddings
        if model_cls in [MarianForCausalLM, MarianMTModel, MarianModel]:
            continue

        # TODO: "model is not supported yet" from HFTracer
        if model_cls in [HubertForSequenceClassification]:
            continue

        # TODO: shape mismatch in loss calculation
        if model_cls in [LxmertForQuestionAnswering]:
            continue

        family_name = cls_name.split("For")[0]
        if family_name not in family:
            family[family_name] = []
        if cls_name.endswith(("MaskedLM", "CausalLM")) and family_name not in lm_seen:
            family[family_name].append(cls_name)
            lm_seen.add(family_name)
        elif (
            cls_name.endswith(
                ("SequenceClassification", "ConditionalGeneration", "QuestionAnswering")
            )
            and family_name not in family_seen
        ):
            family[family_name].append(cls_name)
            family_seen.add(family_name)
        elif cls_name.endswith("ImageClassification"):
            family[family_name].append(cls_name)

    chosen_models = set()
    for members in family.values():
        chosen_models.update(set(members))

    chosen_models.update(set(EXTRA_MODELS.keys()))
    filename = "huggingface_models_list.txt"
    if os.path.exists("benchmarks"):
        filename = "benchmarks/" + filename
    for model_name in sorted(chosen_models):
        try:
            subprocess.check_call(
                [sys.executable]
                + sys.argv
                + ["--find-batch-sizes"]
                + [f"--only={model_name}"]
                + [f"--output={filename}"]
            )
        except subprocess.SubprocessError:
            print("ERROR")


if __name__ == "__main__":
    if REFRESH_MODEL_NAMES and "--find-batch-sizes" not in sys.argv:
        refresh_model_names()
    else:
        logging.basicConfig(level=logging.WARNING)
        warnings.filterwarnings("ignore")
        main(HuggingfaceRunner())
