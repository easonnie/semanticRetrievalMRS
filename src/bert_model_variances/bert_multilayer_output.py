from enum import Enum

from pytorch_pretrained_bert import BertModel
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertLayerNorm
import torch
import math


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


class BertMultiLayerPooler(nn.Module):
    def __init__(self, hidden_size, act_type="gelu", num_of_pooling_layer=4):
        super(BertMultiLayerPooler, self).__init__()
        self.num_of_pooling_layer = num_of_pooling_layer
        self.dense = nn.Linear(hidden_size * num_of_pooling_layer, hidden_size * num_of_pooling_layer)
        self.activation = ACT2FN[act_type]

    def forward(self, hidden_states):
        # We "pool" the model by simply taking multiple layers of the hidden state corresponding
        # to the first token.
        # first_token_tensor = hidden_states[:, 0]
        pooled_layer_list = hidden_states[-self.num_of_pooling_layer:]
        first_token_list = []

        for selected_layer in pooled_layer_list:
            first_token_list.append(selected_layer[:, 0])

        concatenated_first_token_output = torch.cat(first_token_list, dim=1)

        pooled_output = self.dense(concatenated_first_token_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertMultiLayerSeqClassification(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1

    def __init__(self, bert_encoder: BertModel, num_labels, num_of_pooling_layer=4, act_type='gelu',
                 use_pretrained_pooler=False, use_sigmoid=False):

        super(BertMultiLayerSeqClassification, self).__init__()
        self.num_labels = num_labels
        self.bert_encoder = bert_encoder
        self.use_pretrained_pooler = use_pretrained_pooler
        if not self.use_pretrained_pooler:
            self.multilayer_pooler = BertMultiLayerPooler(bert_encoder.config.hidden_size, act_type,
                                                          num_of_pooling_layer)
        else:
            self.num_of_pooling_layer = 1   # If we use pretrained pooler, we can only use the last layer
            self.multilayer_pooler = self.bert_encoder.pooler

        self.dropout = nn.Dropout(bert_encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_encoder.config.hidden_size * num_of_pooling_layer, num_labels)

        # Apply init value for model except bert_encoder
        if not self.use_pretrained_pooler:
            self.multilayer_pooler.apply(init_bert_weights)

        self.classifier.apply(init_bert_weights)

        self.use_sigmoid = False
        if num_labels == 1 and use_sigmoid:
            self.use_sigmoid = use_sigmoid
        elif num_labels != 1 and use_sigmoid:
            raise ValueError("Can not use sigmoid when number of labels is 1.")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mode=ForwardMode.TRAIN, labels=None):
        sequence_output, pooled_output = self.bert_encoder(input_ids, token_type_ids, attention_mask,
                                                           output_all_encoded_layers=True)

        if not self.use_pretrained_pooler:
            multilayer_pooled_output = self.multilayer_pooler(sequence_output)
        else:
            multilayer_pooled_output = pooled_output

        dr_pooled_output = self.dropout(multilayer_pooled_output)
        logits = self.classifier(dr_pooled_output)

        if mode == BertMultiLayerSeqClassification.ForwardMode.TRAIN:
            if self.use_sigmoid:
                assert labels is not None
                loss_fn = nn.BCEWithLogitsLoss()

                loss = loss_fn(logits, labels.unsqueeze(-1).float())
                return loss
            else:   # use softmax
                assert labels is not None
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss

        elif mode == BertMultiLayerSeqClassification.ForwardMode.EVAL:
            return logits


if __name__ == '__main__':
    torch.manual_seed(8)
    bert_model_name = 'bert-base-uncased'
    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertMultiLayerSeqClassification(bert_encoder, num_labels=3)

    print(model)

    print(model.bert_encoder.pooler.dense.weight)

    print(model.multilayer_pooler.dense.weight)