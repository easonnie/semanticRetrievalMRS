from enum import Enum

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
from pytorch_pretrained_bert.modeling import BertLayerNorm
from data_utils.readers.span_pred_reader import BertSpanPredReader

import flint.span_util as span_util
import flint.torch_util as torch_util
import torch.nn as nn
from torch.nn.functional import nll_loss
import torch

import config


def init_bert_weights(module, initializer_range):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.beta.data.normal_(mean=0.0, std=initializer_range)
        module.gamma.data.normal_(mean=0.0, std=initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class BertSeqClassifier(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1

    def __init__(self, bert_encoder):
        super(BertSeqClassifier, self).__init__()
        self.bert_encoder = bert_encoder
        self.final_linear = nn.Linear(self.bert_encoder.config.hidden_size, 2)  # Should we have dropout here? Later?
        self.dropout = nn.Dropout(self.bert_encoder.config.hidden_dropout_prob)
        init_bert_weights(self.qa_outputs, initializer_range=0.02)  # Hard code this value

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mode=ForwardMode.TRAIN, labels=None):
        # Precomputing of the max_context_length is important
        # because we want the same value to be shared to different GPUs, dynamic calculating is not feasible.
        _, pooled_output = self.bert_encoder(input_ids, token_type_ids, attention_mask,
                                             output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.final_linear(pooled_output)

        if mode == BertSeqClassifier.ForwardMode.TRAIN:
            assert labels is not None

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
