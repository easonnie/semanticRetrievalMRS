import math
from enum import Enum

import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertLayerNorm

import flint.span_util as span_util
import flint.torch_util as torch_util

import torch


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, 'tanh': F.tanh}


def init_bert_weights(module):
    """ Initialize the weights.
    """
    initializer_range = 0.02
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.beta.data.normal_(mean=0.0, std=initializer_range)
        module.gamma.data.normal_(mean=0.0, std=initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class BertPairMaxOutMatcher(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1

    def __init__(self, bert_encoder: BertModel, num_of_class, act_type="gelu", dropout_rate=None, num_of_out_layers=4,
                 use_sigmoid=False):
        super(BertPairMaxOutMatcher, self).__init__()
        self.bert_encoder = bert_encoder
        self.num_of_out_layers = num_of_out_layers
        self.num_of_class = num_of_class

        self.matching_hidden_dimension = \
            4 * self.num_of_out_layers * self.bert_encoder.config.hidden_size
        self.matching_intermediate_dimension = self.bert_encoder.config.intermediate_size

        if dropout_rate is None:
            dropout_rate = self.bert_encoder.config.hidden_dropout_prob

        self.matching_layer1 = nn.Linear(self.matching_hidden_dimension, self.matching_intermediate_dimension)
        self.matching_layer2 = nn.Linear(self.matching_intermediate_dimension, num_of_class)
        self.dropout = nn.Dropout(dropout_rate)

        self.activation = ACT2FN[act_type]

        self.match_layers = nn.Sequential(*[self.matching_layer1, nn.ReLU(), self.dropout, self.matching_layer2])

        self.use_sigmoid = False
        if self.num_of_class == 1 and use_sigmoid:
            self.use_sigmoid = use_sigmoid
        elif self.num_of_class != 1 and use_sigmoid:
            raise ValueError("Can not use sigmoid when number of labels is 1.")

    @staticmethod
    def span_maxpool(input_seq, span):  # [B, T, D]
        selected_seq, selected_length = span_util.span_select(input_seq, span)  # [B, T, D]
        maxout_r = torch_util.max_along_time(selected_seq, selected_length)
        return maxout_r

    def forward(self, seq, token_type_ids, attention_mask, s1_span, s2_span, mode, labels=None):
        # Something
        encoded_layers, _ = self.bert_encoder(seq, token_type_ids, attention_mask,
                                              output_all_encoded_layers=True)
        selected_output_layers = encoded_layers[-self.num_of_out_layers:]  # [[B, T, D]]   0, 1, 2
        # context_length = att_mask.sum(dim=1)
        selected_output = torch.cat(selected_output_layers, dim=2)  # Concat at last layer.

        s1_out = self.span_maxpool(selected_output, s1_span)
        s2_out = self.span_maxpool(selected_output, s2_span)

        s1_out = self.dropout(s1_out)
        s2_out = self.dropout(s2_out)
        paired_out = torch.cat([s1_out, s2_out, torch.abs(s1_out - s2_out), s1_out * s2_out], dim=1)

        paired_out_1 = self.dropout(self.activation(self.matching_layer1(paired_out)))
        logits = self.matching_layer2(paired_out_1)

        if mode == BertPairMaxOutMatcher.ForwardMode.TRAIN:
            if self.use_sigmoid:
                assert labels is not None
                loss_fn = nn.BCEWithLogitsLoss()

                loss = loss_fn(logits, labels.unsqueeze(-1).float())
                return loss
            else:
                assert labels is not None
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_of_class), labels.view(-1))
                return loss
        else:
            return logits


class BertSupervisedVecClassifier(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1

    def __init__(self, bert_content2vec_model, num_of_class, dropout_rate=None):
        super(BertSupervisedVecClassifier, self).__init__()
        self.bert_content2vec_model = bert_content2vec_model
        self.num_of_class = num_of_class

        self.matching_hidden_dimension = \
            4 * self.bert_content2vec_model.num_of_out_layers * self.bert_content2vec_model.bert_encoder.config.hidden_size
        self.matching_intermediate_dimension = self.bert_content2vec_model.bert_encoder.config.intermediate_size
        if dropout_rate is None:
            dropout_rate = self.bert_content2vec_model.bert_encoder.config.hidden_dropout_prob

        self.matching_layer1 = nn.Linear(self.matching_hidden_dimension, self.matching_intermediate_dimension)
        self.matching_layer2 = nn.Linear(self.matching_intermediate_dimension, self.num_of_class)
        self.dropout = nn.Dropout(dropout_rate)

        self.match_layers = nn.Sequential(*[self.matching_layer1, nn.ReLU(), self.dropout, self.matching_layer2])

    def forward(self, s1_seq, s1_mask, s2_seq, s2_mask, mode, labels=None):
        s1_out = self.bert_content2vec_model(s1_seq, attention_mask=s1_mask)
        s2_out = self.bert_content2vec_model(s2_seq, attention_mask=s2_mask)

        s1_out = self.dropout(s1_out)
        s2_out = self.dropout(s2_out)

        logits = self.match_layers(torch.cat([s1_out, s2_out, torch.abs(s1_out - s2_out), s1_out * s2_out], dim=1))

        if mode == BertSupervisedVecClassifier.ForwardMode.TRAIN:
            assert labels is not None
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_of_class), labels.view(-1))
            return loss
        else:
            return logits
