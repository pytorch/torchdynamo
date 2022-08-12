import unittest

import torch
from torch.fx.tensor_type import Dyn

import torchdynamo
import torchdynamo.testing

import inspect
from enum import Enum
from torch.fx.tensor_type import Dyn
from transformers import *

try:
    import z3  # noqa
    from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import (  # noqa
        evaluate_conditional_with_constraints,
    )

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

skipIfNoZ3 = unittest.skipIf(not HAS_Z3, "no z3")


bs = 4
num_choices = 3
seq_length = 32


class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2



def generate_concrete_args_for_model(model, input_names=None):
    input_names = input_names if input_names else model.dummy_inputs.keys()
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
    return concrete_args


def generate_hf_model(model_cls, hidden_layers=None):
    config_cls = model_cls.config_class
    config = config_cls()

    # we simplify the model for now by removing the hidden layers
    if hidden_layers is not None:
        config.num_hidden_layers = hidden_layers
    if model_cls in [GPT2ForSequenceClassification, GPTNeoForSequenceClassification, GPTJForSequenceClassification] or \
            model_cls.__name__.startswith("Roberta") or model_cls.__name__.startswith("Marian"):
        config.pad_token_id = 0
    model = model_cls(config)
    model.eval()

    return model


def generate_inputs_for_model(model_cls, model, include_loss_args=False):
    if model_cls.__name__.endswith('MultipleChoice'):
        input = torch.zeros(bs, num_choices, seq_length, dtype=torch.long).random_(model.config.vocab_size)
    elif model_cls.__name__.startswith("Roberta"):
        input = torch.zeros(bs, seq_length, dtype=torch.long)
    else:
        input = torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size)

    if 'Bart' in model_cls.__name__:
        input[:, -1] = model.config.eos_token_id

    input_dict = {'input_ids': input}

    if model_cls.__name__.startswith("T5") or model_cls.__name__.startswith("M2M100") \
            or model_cls.__name__.startswith("MT5") or model_cls in [BlenderbotModel, BlenderbotSmallModel,
                                                                     BlenderbotForConditionalGeneration,
                                                                     BlenderbotSmallForConditionalGeneration,
                                                                     PegasusModel, PegasusForConditionalGeneration,
                                                                     MarianModel, MarianMTModel]:
        input_dict.update({'decoder_input_ids': input})

    if include_loss_args:
        if model_cls.__name__.endswith('PreTraining'):
            if model_cls == ElectraForPreTraining:
                input_dict.update({
                    'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(1),
                })
            else:
                label_name = 'sentence_order_label' if model_cls in [AlbertForPreTraining] else 'next_sentence_label'
                input_dict.update({
                    'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size),
                    label_name: torch.zeros(bs, dtype=torch.long).random_(1),
                })
        elif model_cls.__name__.endswith('QuestionAnswering'):
            input_dict.update({
                'start_positions': torch.zeros(bs, dtype=torch.long).random_(seq_length),
                'end_positions': torch.zeros(bs, dtype=torch.long).random_(seq_length)
            })
        elif (model_cls.__name__.endswith('MaskedLM') or model_cls.__name__.endswith('HeadModel') or
              model_cls.__name__.endswith('CausalLM') or model_cls.__name__.endswith('DoubleHeadsModel')):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size),
            })
        elif model_cls.__name__.endswith('TokenClassification'):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.num_labels - 1),
            })
        elif model_cls.__name__.endswith('MultipleChoice'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(num_choices),
            })
        elif model_cls.__name__.endswith('SequenceClassification'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(model.config.num_labels - 1),
            })
        elif model_cls.__name__.endswith('NextSentencePrediction'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(1),
            })
        elif model_cls.__name__.endswith('ForConditionalGeneration'):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size - 1),
            })
        else:
            raise NotImplementedError(f'Class {model_cls.__name__} unsupported for training test ')

    return input_dict


model_classes = [XGLMModel]


class TorchDynamoUseCases(unittest.TestCase):
    @skipIfNoZ3
    def test_reshape(self):
        """
        Here, we expect a single graph because
        we proved that the conditional is always false
        """

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: Dyn):
                y = x.view(100)
                tmp = y.size()[0]
                if tmp < 100:
                    return torch.dropout(x, p=0.5, train=False)
                else:
                    return torch.relu(x)

        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()

        with torchdynamo.optimize(cnts):
            BasicBlock().forward(torch.rand(50, 2))

        self.assertEqual(cnts.frame_count, 1)

    @skipIfNoZ3
    def test_fake_condition(self):
        """
        We use a gt node, but it is not actually
        a conditional. Here, we should do nothing.
        """

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: Dyn):
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                view = x.view(-1, getitem)
                lt = arange > view
                masked_fill = x.masked_fill_(lt, 0)
                return masked_fill

        torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            BasicBlock().forward(torch.rand(50, 2))

        # Nothing should change here. No graph breaks.
        self.assertEqual(cnts.frame_count, 1)


    @skipIfNoZ3
    # framecount before is 14
    def test_XGLM(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()



        with torchdynamo.optimize(cnts):
            m = generate_hf_model(XGLMModel, hidden_layers=1)
            m.forward(torch.ones([4, 32], dtype=torch.long))

        print(cnts.frame_count)
        # print(CompileProfiler)
        self.assertEqual(cnts.frame_count, 5)


    @skipIfNoZ3
    def test_M2M100Model(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()

        with torchdynamo.optimize(cnts):
            m = generate_hf_model(M2M100Model, hidden_layers=0)
            m.forward(torch.ones([4, 32], dtype=torch.int), decoder_input_ids=torch.ones([4, 32], dtype=torch.int))

        # before was 47
        self.assertEqual(cnts.frame_count, 35)


    @skipIfNoZ3
    def test_MegatronBertModel(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            m = generate_hf_model(MegatronBertModel, hidden_layers=1)
            m.forward(torch.ones([4, 32], dtype=torch.int))
        self.assertEqual(cnts.frame_count, 8)


    @skipIfNoZ3
    def test_MobileBertModel(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            m = generate_hf_model(MobileBertModel, hidden_layers=1)
            m.forward(torch.ones([4, 32], dtype=torch.int))

        self.assertEqual(cnts.frame_count, 5)


    @skipIfNoZ3
    def test_RobertaModel(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            m = generate_hf_model(RobertaModel, hidden_layers=1)
            m.forward(torch.ones([4, 32], dtype=torch.int))

        self.assertEqual(cnts.frame_count, 8)


    @skipIfNoZ3
    def test_ElectraModel(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            m = generate_hf_model(ElectraModel, hidden_layers=1)
            m.forward(torch.ones([4, 32], dtype=torch.int))

        print(cnts.frame_count)

        # self.assertEqual(cnts.frame_count, 8)



    @skipIfNoZ3
    def test_BertModel(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            m = generate_hf_model(BertModel, hidden_layers=1)
            m.forward(torch.ones([4, 32], dtype=torch.int))

        print(cnts.frame_count)
        # self.assertEqual(cnts.frame_count, 8)


    @skipIfNoZ3
    def test_T5Model(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            m = generate_hf_model(T5Model, hidden_layers=0)
            m.forward(torch.ones([4, 32], dtype=torch.int), decoder_input_ids=torch.ones([4, 32], dtype=torch.int))

        print(cnts.frame_count)
        # self.assertEqual(cnts.frame_count, 8)



    @skipIfNoZ3
    def test_MT5Model(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            m = generate_hf_model(MT5Model)
            m.forward(torch.ones([4, 32], dtype=torch.int), decoder_input_ids=torch.ones([4, 32], dtype=torch.int))

        print(cnts.frame_count)


    @skipIfNoZ3
    def test_MarianMTModel(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts):
            m = generate_hf_model(MarianModel, hidden_layers=1)
            m.forward(torch.ones([4, 32], dtype=torch.int), decoder_input_ids=torch.ones([4, 32], dtype=torch.int))
        self.assertEqual(cnts.frame_count, 30)
