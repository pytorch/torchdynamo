#!/usr/bin/env python
import importlib
import logging
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
    from transformers import DebertaConfig
    from transformers import DebertaForMaskedLM
    from transformers import GPT2Config
    from transformers import GPT2LMHeadModel
    from transformers import RobertaConfig
    from transformers import RobertaForMaskedLM
    from transformers import T5Config
    from transformers import T5ForConditionalGeneration
    from transformers import XLNetConfig
    from transformers import XLNetLMHeadModel


# We are primarily interested in tf32 datatype
torch.backends.cuda.matmul.allow_tf32 = True

SKIP = {}


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


def albert_input_func(device, dtype, vocab_size, batch_size, seq_len, tgt_seq_len=None):
    batch_size = 8
    seq_len = 512
    res = hf_general_inputs(dtype, device, vocab_size, batch_size, seq_len)
    sentence_order_label = rand_int_tensor(device, 0, 2, (batch_size,))
    res.update({"sentence_order_label": sentence_order_label})
    return res


ALL_MODELS = {
    "BertForPreTraining_P1_bert": (
        BertConfig.from_pretrained("bert-large-uncased"),
        BertForPreTraining,
        partial(bert_input_func, batch_size=64, seq_len=128),
    ),
    "BertForPreTraining_P2_bert": (
        BertConfig.from_pretrained("bert-large-uncased"),
        BertForPreTraining,
        partial(bert_input_func, batch_size=16, seq_len=512),
    ),
    "GPT2LMHeadModel_gpt2": (
        GPT2Config.from_pretrained("gpt2-large"),
        GPT2LMHeadModel,
        partial(hf_general_inputs, batch_size=2, seq_len=1024),
    ),
    "RobertaForMaskedLM_roberta": (
        RobertaConfig.from_pretrained("roberta-large"),
        RobertaForMaskedLM,
        partial(hf_general_inputs, batch_size=64, seq_len=128),
    ),
    "AlbertForPreTraining_albert": (
        AlbertConfig.from_pretrained("albert-xxlarge-v2"),
        AlbertForPreTraining,
        albert_input_func,
    ),
    "T5ForConditionalGeneration_t5": (
        T5Config.from_pretrained("t5-large"),
        T5ForConditionalGeneration,
        partial(hf_general_inputs, batch_size=8, seq_len=512, tgt_seq_len=128),
    ),
    "BartForConditionalGeneration_bart": (
        BartConfig.from_pretrained("facebook/bart-large"),
        BartForConditionalGeneration,
        partial(hf_general_inputs, batch_size=8, seq_len=1024, tgt_seq_len=128),
    ),
    "DebertaForMaskedLM_deberata": (
        DebertaConfig.from_pretrained("microsoft/deberta-large"),
        DebertaForMaskedLM,
        partial(hf_general_inputs, batch_size=8, seq_len=512),
    ),
    "XLNetLMHeadModel_xlnet": (
        XLNetConfig.from_pretrained("xlnet-large-cased"),
        XLNetLMHeadModel,
        partial(hf_general_inputs, batch_size=16, seq_len=512),
    ),
    "allenai-longformer-base-4096": (
        AutoConfig.from_pretrained("allenai/longformer-base-4096"),
        AutoModelForMaskedLM,
        partial(hf_general_inputs, batch_size=2, seq_len=1024),
    ),
    "Reformer": (
        ReformerConfig(),
        AutoModelForMaskedLM,
        partial(hf_general_inputs, batch_size=8, seq_len=4096),
    ),
    "t5-small": (
        AutoConfig.from_pretrained("t5-small"),
        AutoModelForSeq2SeqLM,
        partial(hf_general_inputs, batch_size=1, seq_len=1024),
    ),
    "distilbert-base-uncased": (
        AutoConfig.from_pretrained("distilbert-base-uncased"),
        AutoModelForMaskedLM,
        partial(hf_general_inputs, batch_size=8, seq_len=512),
    ),
    "bigbird": (
        BigBirdConfig(attention_type="block_sparse"),
        AutoModelForMaskedLM,
        partial(hf_general_inputs, batch_size=2, seq_len=1024),
    ),
    "distilgpt2": (
        AutoConfig.from_pretrained("distilgpt2"),
        AutoModelForCausalLM,
        partial(hf_general_inputs, batch_size=16, seq_len=512),
    ),
    "google-electra-base-discriminator": (
        AutoConfig.from_pretrained("google/electra-base-discriminator"),
        AutoModelForMaskedLM,
        partial(hf_general_inputs, batch_size=8, seq_len=512),
    ),
    "google-fnet-base": (
        AutoConfig.from_pretrained("google/fnet-base"),
        AutoModelForMaskedLM,
        partial(hf_general_inputs, batch_size=8, seq_len=512, no_attention_mask=True),
    ),
    "YituTech-conv-bert-base": (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),
        AutoModelForMaskedLM,
        partial(hf_general_inputs, batch_size=8, seq_len=512),
    ),
    "google-mobilebert-uncased": (
        AutoConfig.from_pretrained("google/mobilebert-uncased"),
        AutoModelForMaskedLM,
        partial(hf_general_inputs, batch_size=4, seq_len=512),
    ),
    "camembert-base": (
        AutoConfig.from_pretrained("camembert-base"),
        AutoModelForMaskedLM,
        partial(hf_general_inputs, batch_size=8, seq_len=512),
    ),
    "microsoft-layoutlm-base-uncased": (
        AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased"),
        AutoModelForMaskedLM,
        partial(hf_general_inputs, batch_size=8, seq_len=512),
    ),
}


class HuggingfaceRunner(BenchmarkRunner):
    def __init__(self):
        super(HuggingfaceRunner, self).__init__()

    def load_model(self, device, model_name, is_training, use_eval_mode):
        dtype = torch.float32
        config, model_cls, input_fn = ALL_MODELS[model_name]

        if "auto" in model_cls.__module__:
            # Handle auto classes
            model = model_cls.from_config(config).to(device, dtype=dtype)
        else:
            model = model_cls(config).to(device, dtype=dtype)

        # So we can check for correct gradients without eliminating the dropout computation
        for attr in dir(config):
            if "drop" in attr and isinstance(getattr(config, attr), float):
                setattr(config, attr, 1e-30)

        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        # Prepare inputs
        example_inputs = input_fn(
            dtype=dtype, device=device, vocab_size=config.vocab_size
        )
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

    def get_tolerance(self, is_training, current_device, name):
        return 1e-3

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(HuggingfaceRunner())
