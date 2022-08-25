
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
                                        input_embeds_2 == stack_0])] * 6 + [False] + [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                              s1 > 0,
                                                                                              s2 > 1,
                                                                                              s2 < 2000])] * 7 + [False] + [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                                                                     s1 > 0,
                                                                                                                                     s2 > 1,
                                                                                                                                     s2 < 2000])] * 6 + [False] * 2 + [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                                                                                                               s1 > 0,
                                                                                                                                                                               s2 > 1,
                                                                                                                                                                               s2 < 2000])] * 7 + [False] +  [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                                                                                                                                                      s1 > 0,
                                                                                                                                                                                                                      s2 > 1,
                                                                                                                                                                                                                      s2 < 2000])] * 8 + [False]* 40




user_constraints_blenderbot = [False, False] \
                              + \
                              [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, s3)),
                                       s1 > 0,
                                       s2 > 1,
                                       s2 < 2000])] * 8 + [False] + [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, s3)),
                                                                             s1 > 0,
                                                                             s2 > 1,
                                                                             s2 < 2000])] * 13 + [False] * 40






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
                                      input_embeds_2 == stack_0])] * 6 + [False] * 2 + [z3.And([input == tensor_type.tensor3(D(1,s1), D(1, s2), D(1, s3)),
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



def run_function(model, user_constraints):

    torchdynamo.config.dynamic_shapes = True
    cnts = torchdynamo.testing.CompileCounter()
    with torchdynamo.optimize(cnts, user_constraints=user_constraints):
        m = generate_hf_model(model, hidden_layers=0)
        m.forward(torch.ones([4, 32], dtype=torch.int), decoder_input_ids=torch.ones([4, 32], dtype=torch.int))
        print(cnts.frame_count)


# run_function(BlenderbotSmallModel, user_constraints_blenderbot)

# run_function(MarianModel, user_constraints_marian_mt)

# run_function(MarianMTModel, user_constraints_marian_mt)


run_function(M2M100Model, user_constraints_M2M100Model)