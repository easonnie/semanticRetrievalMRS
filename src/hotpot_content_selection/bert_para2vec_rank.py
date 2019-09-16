from enum import Enum

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
from pytorch_pretrained_bert.modeling import BertLayerNorm, gelu
from data_utils.readers.span_pred_reader import BertSpanPredReader

import flint.span_util as span_util
import flint.torch_util as torch_util
import torch.nn as nn
import torch.nn.functional as F
from flint import torch_util
from allennlp.nn import util as allen_util
from torch.nn.functional import nll_loss

import torch

import config


class BertContent2Vec(nn.Module):
    def __init__(self, bert_encoder, num_of_out_layers=4):
        super(BertContent2Vec, self).__init__()
        self.bert_encoder = bert_encoder
        self.num_of_out_layers = num_of_out_layers

        # self.final_linear = nn.Linear(self.bert_encoder.config.hidden_size, 2)  # Should we have dropout here? Later?
        # self.dropout = nn.Dropout(self.bert_encoder.config.hidden_dropout_prob)
        # init_bert_weights(self.qa_outputs, initializer_range=0.02)  # Hard code this value

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Precomputing of the max_context_length is important
        # because we want the same value to be shared to different GPUs, dynamic calculating is not feasible.
        encoded_layers, pooled_output = self.bert_encoder(input_ids, token_type_ids, attention_mask,
                                                          output_all_encoded_layers=True)
        selected_output_layers = encoded_layers[-self.num_of_out_layers:]
        context_length = attention_mask.sum(dim=1)

        output_layer_list = []
        for i, output_layer in enumerate(selected_output_layers):
            output_layer_list.append(torch_util.max_along_time(output_layer, context_length))   # [B, T, D] -> [B, D]

        packed_output = torch.cat(output_layer_list, dim=1)

        return packed_output
        # context_mask = allen_util.get_mask_from_sequence_lengths(context_length, max_context_length)

        # pooled_output = self.dropout(pooled_output)
        # logits = self.final_linear(pooled_output)


class BertSupervisedVecMatcher(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1

    def __init__(self, bert_content2vec_model, dropout_rate=None):
        super(BertSupervisedVecMatcher, self).__init__()
        self.bert_content2vec_model = bert_content2vec_model

        self.matching_hidden_dimension = \
            4 * self.bert_content2vec_model.num_of_out_layers * self.bert_content2vec_model.bert_encoder.config.hidden_size
        self.matching_intermediate_dimension = self.bert_content2vec_model.bert_encoder.config.intermediate_size
        if dropout_rate is None:
            dropout_rate = self.bert_content2vec_model.bert_encoder.config.hidden_dropout_prob

        self.matching_layer1 = nn.Linear(self.matching_hidden_dimension, self.matching_intermediate_dimension)
        self.matching_layer2 = nn.Linear(self.matching_intermediate_dimension, 1)
        self.dropout = nn.Dropout(dropout_rate)

        self.match_layers = nn.Sequential(*[self.matching_layer1, nn.ReLU(), self.dropout, self.matching_layer2])

    def forward(self, s1_seq, s1_mask, s2_seq, s2_mask, mode, labels=None):
        s1_out = self.bert_content2vec_model(s1_seq, attention_mask=s1_mask)
        s2_out = self.bert_content2vec_model(s2_seq, attention_mask=s2_mask)

        s1_out = self.dropout(s1_out)
        s2_out = self.dropout(s2_out)

        logits = self.match_layers(torch.cat([s1_out, s2_out, torch.abs(s1_out - s2_out), s1_out * s2_out], dim=1))

        if mode == BertSupervisedVecMatcher.ForwardMode.TRAIN:
            assert labels is not None
            loss_fn = nn.BCEWithLogitsLoss()
            # batch_size = logits.size(0)
            # labels_logits = logits.new_zeros(batch_size, 2)
            # labels_logits.scatter_(1, labels.unsqueeze(-1), 1)
            loss = loss_fn(logits, labels.unsqueeze(-1).float())
            return loss
        else:
            return logits


# This is the raw Matcher, deprecated!
class BertVecMatcher(nn.Module):
    def __init__(self, bert_content2vec_model):
        super(BertVecMatcher, self).__init__()
        self.bert_content2vec_model = bert_content2vec_model

    def forward(self, s1_seq, s1_mask, s2_seq, s2_mask):
        s1_out = self.bert_content2vec_model(s1_seq, attention_mask=s1_mask)
        s2_out = self.bert_content2vec_model(s2_seq, attention_mask=s2_mask)
        cosine_simi_score = F.cosine_similarity(s1_out, s2_out)
        # batch_size = s1_out.size(0)
        # hidden_size = s1_out.size(1)
        # m_scores = torch.bmm(s1_out.view(batch_size, 1, hidden_size), s2_out.view(batch_size, hidden_size, 1))
        # return m_scores
        return cosine_simi_score
