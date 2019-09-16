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
from span_prediction_task_utils.squad_utils import preprocessing_squad, preprocssing_span_prediction_item, eitems_to_fitems
from utils import common
from allennlp.data.iterators import BasicIterator
from allennlp.nn import util as allen_util
from tqdm import tqdm


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


class BertSpan(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1

    def __init__(self, bert_encoder):
        super(BertSpan, self).__init__()
        self.bert_encoder = bert_encoder
        self.qa_outputs = nn.Linear(self.bert_encoder.config.hidden_size, 2)  # Should we have dropout here? Later?
        init_bert_weights(self.qa_outputs, initializer_range=0.02)  # Hard code this value

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, context_span=None,
                gt_span=None, max_context_length=0, mode=ForwardMode.TRAIN):
        # Precomputing of the max_context_length is important
        # because we want the same value to be shared to different GPUs, dynamic calculating is not feasible.
        sequence_output, _ = self.bert_encoder(input_ids, token_type_ids, attention_mask,
                                               output_all_encoded_layers=False)

        joint_seq_logits = self.qa_outputs(sequence_output)
        context_logits, context_length = span_util.span_select(joint_seq_logits, context_span, max_context_length)
        context_mask = allen_util.get_mask_from_sequence_lengths(context_length, max_context_length)

        # The following line is from AllenNLP bidaf.
        start_logits = allen_util.replace_masked_values(context_logits[:, :, 0], context_mask, -1e18)
        # B, T, 2
        end_logits = allen_util.replace_masked_values(context_logits[:, :, 1], context_mask, -1e18)

        if mode == BertSpan.ForwardMode.TRAIN:
            assert gt_span is not None
            gt_start = gt_span[:, 0]  # gt_span: [B, 2]
            gt_end = gt_span[:, 1]

            start_loss = nll_loss(allen_util.masked_log_softmax(start_logits, context_mask), gt_start.squeeze(-1))
            end_loss = nll_loss(allen_util.masked_log_softmax(end_logits, context_mask), gt_end.squeeze(-1))

            loss = start_loss + end_loss
            return loss
        else:
            return start_logits, end_logits, context_length


def non_answer_filter(fitem):
    if fitem['start_position'] == -1 or fitem['end_position'] == -1:
        return True


def go_model():
    bert_model_name = "bert-base-uncased"
    do_lower_case = True
    batch_size = 32
    learning_rate = 5e-5
    num_train_optimization_steps = 200
    debug = True
    warmup_rate = 0.1
    max_pre_context_length = 200
    max_query_length = 64
    lazy = False

    print("Potential total length:", max_pre_context_length + max_query_length + 3)
    # Important: "max_pre_context_length + max_query_length + 3" is total length

    # debug = False

    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)

    squad_train_v11 = common.load_json(config.SQUAD_TRAIN_1_1)
    squad_dev_v11 = common.load_json(config.SQUAD_DEV_1_1)

    train_eitem_list = preprocessing_squad(squad_train_v11)
    dev_eitem_list = preprocessing_squad(squad_dev_v11)

    if debug:
        train_eitem_list = [train_eitem_list[0], train_eitem_list[100], train_eitem_list[200],
                            train_eitem_list[300], train_eitem_list[400]]

    train_fitem_dict, train_fitem_list = eitems_to_fitems(train_eitem_list, tokenizer, is_training=True,
                                                          max_tokens_for_doc=max_pre_context_length)
    dev_fitem_dict, dev_fitem_list = eitems_to_fitems(dev_eitem_list, tokenizer, is_training=False,
                                                      max_tokens_for_doc=max_pre_context_length)
    # Something test

    if debug:
        train_fitem_list = train_fitem_list[:5]

    print("Total train fitems:", len(train_fitem_list))

    span_pred_reader = BertSpanPredReader(tokenizer, max_query_length=max_query_length, lazy=lazy,
                                          example_filter=non_answer_filter)
    train_instances = span_pred_reader.read(train_fitem_list)
    dev_instances = span_pred_reader.read(dev_fitem_list)

    print("Total train instances:", len(train_instances))

    iterator = BasicIterator(batch_size=batch_size)

    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertSpan(bert_encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    # sinlge gpu

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_rate,
                         t_total=num_train_optimization_steps)

    for iteration in tqdm(range(200)):
        t_iter = iterator(train_instances, num_epochs=1, shuffle=False)
        for batch in tqdm(t_iter):
            # print(batch['paired_sequence'])
            # print(span_util.span_select(batch['paired_sequence'], batch['bert_s1_span']))
            # print(span_util.span_select(batch['paired_sequence'], batch['bert_s2_span']))

            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            seq_context_span = batch['bert_s2_span']  # Context span is s2.
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
            b_max_context_length = max([end - start for (start, end) in batch['bert_s2_span']]) # THis is a int
            gt_span = batch['gt_span']

            paired_sequence = paired_sequence.to(device)
            paired_segments_ids = paired_segments_ids.to(device)
            att_mask = att_mask.to(device)
            seq_context_span = seq_context_span.to(device)
            gt_span = gt_span.to(device)

            # b_fids = batch['fid']
            # b_uids = batch['uid']
            # print(gt_span)

            loss = model(mode=BertSpan.ForwardMode.TRAIN,
                         input_ids=paired_sequence,
                         token_type_ids=paired_segments_ids,
                         attention_mask=att_mask,
                         context_span=seq_context_span,
                         max_context_length=b_max_context_length,
                         gt_span=gt_span)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == '__main__':
    go_model()