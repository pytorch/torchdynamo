#!/usr/bin/env python
import importlib
import logging
import os
import re
import subprocess
import sys
import warnings
from functools import partial

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

    # from transformers import *

    from transformers import AlbertConfig
    from transformers import AlbertForPreTraining
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM
    from transformers import AutoModelForMaskedLM
    from transformers import AutoModelForSeq2SeqLM
    from transformers import BartConfig
    from transformers import BartForConditionalGeneration
    from transformers import BertConfig
    from transformers import BertForPreTraining
    from transformers import BigBirdConfig
    from transformers import BlenderbotForConditionalGeneration
    from transformers import BlenderbotModel
    from transformers import BlenderbotSmallForConditionalGeneration
    from transformers import BlenderbotSmallModel
    from transformers import DebertaConfig
    from transformers import DebertaForMaskedLM
    from transformers import ElectraForPreTraining
    from transformers import GPT2Config
    from transformers import GPT2ForSequenceClassification
    from transformers import GPT2LMHeadModel
    from transformers import GPTJForSequenceClassification
    from transformers import GPTNeoForSequenceClassification
    from transformers import LxmertForPreTraining
    from transformers import MarianModel
    from transformers import MarianMTModel
    from transformers import PegasusForConditionalGeneration
    from transformers import PegasusModel
    from transformers import RobertaConfig
    from transformers import RobertaForMaskedLM
    from transformers import T5Config
    from transformers import T5ForConditionalGeneration
    from transformers import XLNetConfig
    from transformers import XLNetLMHeadModel


SKIP = {
    # Difficult to run and compare
    "Reformer"
}

# TODO - Fails even after fake tensors
USE_SMALL_BATCH_SIZE = {
    "AlbertForPreTraining": 4,
    "XLNetLMHeadModel": 8,
}


HF_FX_SUPPORTED_MODELS = dict()
filename = "huggingface_models_list.txt"
if os.path.exists("benchmarks"):
    filename = "benchmarks/" + filename
with open(filename, "r") as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(" ")
        batch_size = int(batch_size)
        # TODO - Check why Nvidia folks are not using the largest batch size.
        HF_FX_SUPPORTED_MODELS[model_name] = min(64, batch_size)


def get_module_cls_by_model_name(model_cls_name):
    _module_by_model_name = {
        "Speech2Text2Decoder": "transformers.models.speech_to_text_2.modeling_speech_to_text_2",
        "TrOCRDecoder": "transformers.models.trocr.modeling_trocr",
    }
    module_name = _module_by_model_name.get(model_cls_name, "transformers")
    module = importlib.import_module(module_name)
    return getattr(module, model_cls_name)


def generate_inputs_for_model(model_cls, model, bs, include_loss_args=False):
    # TODO - Check if following values are representative
    if model_cls.__name__.startswith(("Bert", "Roberta", "T5")):
        seq_length = 128
    elif model_cls.__name__.startswith(("GPT2", "Bart")):
        seq_length = 1024
    elif model_cls.__name__.startswith(("Albert", "Deberta", "Layout", "Electra")):
        seq_length = 512
    else:
        logging.warn(
            f"Sequence Length not defined for {model_cls.__name__}. Choosing 128 arbitrarily"
        )
        seq_length = 128
    num_choices = 3
    num_visual_features = 42
    if model_cls.__name__.endswith("MultipleChoice"):
        input = torch.empty(bs, num_choices, seq_length, dtype=torch.long).random_(
            model.config.vocab_size
        )
    elif model_cls.__name__.startswith("Roberta"):
        input = torch.zeros(bs, seq_length, dtype=torch.long)
    else:
        input = torch.empty(bs, seq_length, dtype=torch.long).random_(
            model.config.vocab_size
        )

    if "Bart" in model_cls.__name__:
        input[:, -1] = model.config.eos_token_id

    input_dict = {"input_ids": input}

    if (
        model_cls.__name__.startswith("T5")
        or model_cls.__name__.startswith("M2M100")
        or model_cls.__name__.startswith("MT5")
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

    if model_cls.__name__.startswith("Lxmert"):
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
        if model_cls.__name__.endswith("PreTraining"):
            if model_cls in [ElectraForPreTraining, LxmertForPreTraining]:
                input_dict.update(
                    {
                        "labels": torch.empty(bs, seq_length, dtype=torch.long).random_(
                            1
                        ),
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
                        "labels": torch.empty(bs, seq_length, dtype=torch.long).random_(
                            model.config.vocab_size
                        ),
                        label_name: torch.empty(bs, dtype=torch.long).random_(1),
                    }
                )
        elif model_cls.__name__.endswith("QuestionAnswering"):
            input_dict.update(
                {
                    "start_positions": torch.empty(bs, dtype=torch.long).random_(
                        seq_length
                    ),
                    "end_positions": torch.empty(bs, dtype=torch.long).random_(
                        seq_length
                    ),
                }
            )
        elif (
            model_cls.__name__.endswith("MaskedLM")
            or model_cls.__name__.endswith("HeadModel")
            or model_cls.__name__.endswith("CausalLM")
            or model_cls.__name__.endswith("DoubleHeadsModel")
        ):
            input_dict.update(
                {
                    "labels": torch.empty(bs, seq_length, dtype=torch.long).random_(
                        model.config.vocab_size
                    ),
                }
            )
        elif model_cls.__name__.endswith("TokenClassification"):
            input_dict.update(
                {
                    "labels": torch.empty(bs, seq_length, dtype=torch.long).random_(
                        model.config.num_labels - 1
                    ),
                }
            )
        elif model_cls.__name__.endswith("MultipleChoice"):
            input_dict.update(
                {
                    "labels": torch.empty(bs, dtype=torch.long).random_(num_choices),
                }
            )
        elif model_cls.__name__.endswith("SequenceClassification"):
            input_dict.update(
                {
                    "labels": torch.empty(bs, dtype=torch.long).random_(
                        model.config.num_labels - 1
                    ),
                }
            )
        elif model_cls.__name__.endswith("NextSentencePrediction"):
            input_dict.update(
                {
                    "labels": torch.empty(bs, dtype=torch.long).random_(1),
                }
            )
        elif model_cls.__name__.endswith("ForConditionalGeneration"):
            input_dict.update(
                {
                    "labels": torch.empty(bs, seq_length, dtype=torch.long).random_(
                        model.config.vocab_size - 1
                    ),
                }
            )
        else:
            raise NotImplementedError(
                f"Class {model_cls.__name__} unsupported for training test "
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


def hf_general_inputs(
    dtype,
    device,
    vocab_size,
    batch_size,
    seq_len,
    tgt_seq_len=None,
    no_attention_mask=False,
):
    # dtype arg is rarely used, because inputs are mostly of type int
    if tgt_seq_len is None:
        tgt_seq_len = seq_len
    input_ids = rand_int_tensor(device, 0, vocab_size, (batch_size, seq_len))
    attention_mask = rand_int_tensor(device, 0, 2, (batch_size, seq_len))
    labels = rand_int_tensor(device, 0, vocab_size, (batch_size, tgt_seq_len))
    x = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if no_attention_mask:
        del x["attention_mask"]
    return x


def bert_input_func(device, dtype, vocab_size, batch_size, seq_len, tgt_seq_len=None):
    res = hf_general_inputs(dtype, device, vocab_size, batch_size, seq_len)
    next_sentence_label = rand_int_tensor(device, 0, 2, (batch_size,))
    res.update(
        {
            "next_sentence_label": next_sentence_label,
        }
    )
    return res


def albert_input_func(device, dtype, vocab_size, batch_size):
    seq_len = 512
    res = hf_general_inputs(dtype, device, vocab_size, batch_size, seq_len)
    sentence_order_label = rand_int_tensor(device, 0, 2, (batch_size,))
    res.update({"sentence_order_label": sentence_order_label})
    return res


EXTRA_MODELS = {
    "AlbertForPreTraining": (
        AlbertConfig.from_pretrained("albert-xxlarge-v2"),
        AlbertForPreTraining,
        8,
        albert_input_func,
    ),
    "XLNetLMHeadModel": (
        XLNetConfig.from_pretrained("xlnet-large-cased"),
        XLNetLMHeadModel,
        16,
        partial(hf_general_inputs, seq_len=512),
    ),
    "AllenaiLongformerBase": (
        AutoConfig.from_pretrained("allenai/longformer-base-4096"),
        AutoModelForMaskedLM,
        2,
        partial(hf_general_inputs, seq_len=1024),
    ),
    "Reformer": (
        ReformerConfig(),
        AutoModelForMaskedLM,
        8,
        partial(hf_general_inputs, seq_len=4096),
    ),
    "T5Small": (
        AutoConfig.from_pretrained("t5-small"),
        AutoModelForSeq2SeqLM,
        1,
        partial(hf_general_inputs, seq_len=1024),
    ),
    "BigBird": (
        BigBirdConfig(attention_type="block_sparse"),
        AutoModelForMaskedLM,
        2,
        partial(hf_general_inputs, seq_len=1024),
    ),
    "DistillGPT2": (
        AutoConfig.from_pretrained("distilgpt2"),
        AutoModelForCausalLM,
        16,
        partial(hf_general_inputs, seq_len=512),
    ),
    "GoogleFnet": (
        AutoConfig.from_pretrained("google/fnet-base"),
        AutoModelForMaskedLM,
        8,
        partial(hf_general_inputs, seq_len=512, no_attention_mask=True),
    ),
    "YituTechConvBert": (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),
        AutoModelForMaskedLM,
        8,
        partial(hf_general_inputs, seq_len=512),
    ),
    "CamemBert": (
        AutoConfig.from_pretrained("camembert-base"),
        AutoModelForMaskedLM,
        8,
        partial(hf_general_inputs, seq_len=512),
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

            model = model_cls(config)
            batch_size_default = 1
            batch_size = batch_size or batch_size_default

            if model_name in USE_SMALL_BATCH_SIZE:
                batch_size = USE_SMALL_BATCH_SIZE[model_name]

            example_inputs = generate_inputs_for_model(
                model_cls, model, batch_size, include_loss_args=True
            )
        else:
            config, model_cls, batch_size_default, input_fn = EXTRA_MODELS[model_name]
            batch_size = batch_size or batch_size_default

            if model_name in USE_SMALL_BATCH_SIZE:
                batch_size = USE_SMALL_BATCH_SIZE[model_name]

            if "auto" in model_cls.__module__:
                # Handle auto classes
                model = model_cls.from_config(config).to(device, dtype=dtype)
            else:
                model = model_cls(config).to(device, dtype=dtype)
            # Prepare inputs
            example_inputs = input_fn(
                dtype=dtype,
                device=device,
                vocab_size=config.vocab_size,
                batch_size=batch_size,
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
            # TODO - Some issue with Albert automatic model. Use it from manual.
            if "Albert" in model_name and model_name in HF_FX_SUPPORTED_MODELS:
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

    filename = "huggingface_models_list.txt"
    if os.path.exists("benchmarks"):
        filename = "benchmarks/" + filename
    with open(filename, "w") as fw:
        for model_name in sorted(chosen_models):
            fw.write(model_name + "\n")


if __name__ == "__main__":
    # refresh_model_names()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(HuggingfaceRunner())
