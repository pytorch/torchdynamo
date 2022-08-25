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
        evaluate_conditional_with_constraints,)
    from torch.fx.experimental.migrate_gradual_types.z3_types import tensor_type, D

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

skipIfNoZ3 = unittest.skipIf(not HAS_Z3, "no z3")


bs = 4
num_choices = 3
seq_length = 32

s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
input = z3.Const(1, tensor_type)
input_embeds = z3.Const(3, tensor_type)
self_weights = z3.Const(3, tensor_type)
stack_0 = z3.Const(1, tensor_type)
attention_mask = z3.Const(2, tensor_type)
input_embeds_2 = z3.Const(2, tensor_type)

dimension_var2 = z3.Int(2)

# try repeating a condition multiple times
heuristic = [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                     s1 > 0,
                     s2 > 1,
                     s2 < 2000])] * 20



heuristic2 = [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                      s1 > 0,
                      s2 > 1,
                      s2 < 2000])] * 10

false_constraints = [False] * 20


user_constraints_M2M100Model = [z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2)), s1 > 0,  s2 > 1, s2 < 1024,
                                        self_weights == tensor_type.tensor2(D(1, 2050), D(1, 1024))])] + \
                               [z3.And([input_embeds_2 == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                        s1 > 0,
                                        s2 > 1,
                                        s2 < 2000,
                                        input_embeds_2 == stack_0])] * 8 + [False] * 100 + [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                              s1 > 0,
                                                                                              s2 > 1,
                                                                                              s2 < 2000])] * 9 + [False] +  [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                                                                     s1 > 0,
                                                                                                                                     s2 > 1,
                                                                                                                                     s2 < 2000])] * 9 + [False] * 60




user_constraints_blenderbot = [False, False] \
                              + \
                              [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, s3)),
                                       s1 > 0,
                                       s2 > 1,
                                       s2 < 2000])] * 9 + [False] + [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, s3)),
                                                                             s1 > 0,
                                                                             s2 > 1,
                                                                             s2 < 2000])] * 12 + [False] * 40


bert_user_constraints = [True,
                         True,
                         True,
                         z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                         z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                         z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))])]


user_constraint_mt5model = [True,
                            True,
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            # True,
                            z3.And([dimension_var2 == 2]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                            z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),

                            ]


T5_model_constraints = [True,
                        True,
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                        ]


bert_user_constraints_2 = [z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                           z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                           z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                           z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                           z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))]),
                           z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))])]




user_constraints_XGLM = [z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2)), s1 > 0,  s2 > 1, s2 < 2000]),

                         False,

                         z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2)), s1 > 0,  s2 > 1, s2 < 2000,
                                 self_weights == tensor_type.tensor2(D(1, 2050), D(1, 1024))]),


                         z3.And([input_embeds == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                 s1 > 0,
                                 s2 > 1,
                                 s2 < 2000,
                                 input_embeds == stack_0]),

                         z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                 s1 > 0,
                                 s2 > 1,
                                 s2 < 2000,
                                 input_embeds == stack_0, input_embeds == attention_mask]),

                         z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                 s1 > 0,
                                 s2 > 1,
                                 s2 < 2000]),
                         True,
                         True,
                         True,
                         True,
                         True,
                         True,
                         True]




user_constraints_marian_mt = [z3.And([input_embeds_2 == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                      s1 > 0,
                                      s2 > 1,
                                      s2 < 2000,
                                      input_embeds_2 == stack_0])] * 7 + [False] * 100 + [z3.And([input == tensor_type.tensor3(D(1,s1), D(1, s2), D(1, s3)),
                                                                                            input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                            s1 > 0,
                                                                                            s2 > 1,
                                                                                            s2 < 2000])] * 6 + [False] + \
                             [z3.And([input == tensor_type.tensor3(D(1,s1), D(1, s2), D(1, s3)),
                                      input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                      s1 > 0,
                                      s2 > 1,
                                      s2 < 2000])] * 8 + [False] * 40



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


        with torchdynamo.optimize(cnts, user_constraints=user_constraints_XGLM):
            m = generate_hf_model(XGLMModel, hidden_layers=1)
            m.forward(torch.ones([4, 32], dtype=torch.long))

        # print(cnts.frame_count)
        # print(CompileProfiler)
        self.assertEqual(cnts.frame_count, 5)


    @skipIfNoZ3
    def test_M2M100Model(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()

        with torchdynamo.optimize(cnts, user_constraints=user_constraints_M2M100Model):
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
        with torchdynamo.optimize(cnts, user_constraints=bert_user_constraints):
            m = generate_hf_model(BertModel, hidden_layers=1)
            m.forward(torch.ones([4, 32], dtype=torch.int))

        print(cnts.frame_count)
        # self.assertEqual(cnts.frame_count, 8)


    # @skipIfNoZ3
    # def test_T5Model(self):
    #     # torchdynamo.config.debug = True
    #     torchdynamo.config.dynamic_shapes = True
    #     cnts = torchdynamo.testing.CompileCounter()
    #     with torchdynamo.optimize(cnts):
    #         m = generate_hf_model(T5Model, hidden_layers=0)
    #         m.forward(torch.ones([4, 32], dtype=torch.int), decoder_input_ids=torch.ones([4, 32], dtype=torch.int))
    #
    #     print(cnts.frame_count)
    #     # self.assertEqual(cnts.frame_count, 8)



    # @skipIfNoZ3
    # def test_MT5Model(self):
    #     # torchdynamo.config.debug = True
    #     torchdynamo.config.dynamic_shapes = True
    #     cnts = torchdynamo.testing.CompileCounter()
    #     with torchdynamo.optimize(cnts):
    #         m = generate_hf_model(MT5Model)
    #         m.forward(torch.ones([4, 32], dtype=torch.int), decoder_input_ids=torch.ones([4, 32], dtype=torch.int))
    #
    #     print(cnts.frame_count)


    @skipIfNoZ3
    def test_MarianModel(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts, user_constraints=user_constraints_marian_mt):
            m = generate_hf_model(MarianModel, hidden_layers=0)
            m.forward(torch.ones([4, 32], dtype=torch.int), decoder_input_ids=torch.ones([4, 32], dtype=torch.int))
        self.assertEqual(cnts.frame_count, 30)

    @skipIfNoZ3
    def test_blenderbot(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts, user_constraints=user_constraints_blenderbot):
            m = generate_hf_model(BlenderbotSmallModel, hidden_layers=0)
            m.forward(torch.ones([4, 32], dtype=torch.int), decoder_input_ids=torch.ones([4, 32], dtype=torch.int))
            print(cnts.frame_count)

        # before: 30
        # after:
        # self.assertEqual(cnts.frame_count, 30)



    @skipIfNoZ3
    def test_marianMT(self):
        # torchdynamo.config.debug = True
        torchdynamo.config.dynamic_shapes = True
        cnts = torchdynamo.testing.CompileCounter()
        with torchdynamo.optimize(cnts, user_constraints=user_constraints_marian_mt):
            m = generate_hf_model(MarianMTModel, hidden_layers=0)
            m.forward(torch.ones([4, 32], dtype=torch.int), decoder_input_ids=torch.ones([4, 32], dtype=torch.int))
            # print(cnts.frame_count)
        # before: 42
        # before: 36

        # self.assertEqual(cnts.frame_count, 30)
