import os
import random
from enum import Enum
from pathlib import Path

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
from pytorch_pretrained_bert.modeling import BertLayerNorm

from data_utils.readers.paired_span_pred_reader import BertPairedSpanPredReader
# from data_utils.readers.span_pred_reader import BertSpanPredReader

import flint.torch_util as torch_util
import torch.nn as nn
from torch.nn.functional import nll_loss
import torch

import config
from evaluation import ext_hotpot_eval
from hotpot_fact_selection_sampler.sampler_s_level_to_qa import get_qa_item_with_upstream_sentence
from neural_modules.model_EMA import EMA, get_ema_gpu_id_list
from open_domain_sampler.qa_sampler import get_open_qa_item_with_upstream_paragraphs
from span_prediction_task_utils.common_utils import write_to_predicted_fitem, merge_predicted_fitem_to_eitem
from span_prediction_task_utils.squad_utils import preprocessing_squad
from span_prediction_task_utils.span_preprocess_tool import eitems_to_fitems
from utils import common, list_dict_data_tool, save_tool
from allennlp.data.iterators import BasicIterator
from allennlp.nn import util as allen_util
from tqdm import tqdm
import evaluation.squad_eval_v1
from evaluation import open_domain_qa_eval


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

    def __init__(self, bert_encoder, num_of_layers=1):
        super(BertSpan, self).__init__()
        self.bert_encoder = bert_encoder
        if num_of_layers == 1:
            self.qa_outputs = nn.Linear(self.bert_encoder.config.hidden_size, 2)  # Should we have dropout here? Later?
            init_bert_weights(self.qa_outputs, initializer_range=0.02)  # Hard code this value

        elif num_of_layers == 2:
            self.output_layer1 = nn.Linear(self.bert_encoder.config.hidden_size, self.bert_encoder.config.hidden_size)
            self.activation = nn.Tanh()
            self.dropout = nn.Dropout(self.bert_encoder.config.hidden_dropout_prob)
            self.output_layer2 = nn.Linear(self.bert_encoder.config.hidden_size, 2)

            self.qa_outputs = nn.Sequential(self.output_layer1, self.activation, self.dropout, self.output_layer2)

            init_bert_weights(self.output_layer1, initializer_range=0.02)
            init_bert_weights(self.output_layer2, initializer_range=0.02)
        else:
            raise ValueError("Number of layers not supported.")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                gt_span=None, mode=ForwardMode.TRAIN):
        sequence_output, _ = self.bert_encoder(input_ids, token_type_ids, attention_mask,
                                               output_all_encoded_layers=False)
        joint_length = allen_util.get_lengths_from_binary_sequence_mask(attention_mask)

        joint_seq_logits = self.qa_outputs(sequence_output)

        # The following line is from AllenNLP bidaf.
        start_logits = allen_util.replace_masked_values(joint_seq_logits[:, :, 0], attention_mask, -1e18)
        # B, T, 2
        end_logits = allen_util.replace_masked_values(joint_seq_logits[:, :, 1], attention_mask, -1e18)

        if mode == BertSpan.ForwardMode.TRAIN:
            assert gt_span is not None
            gt_start = gt_span[:, 0]  # gt_span: [B, 2] -> [B]
            gt_end = gt_span[:, 1]

            start_loss = nll_loss(allen_util.masked_log_softmax(start_logits, attention_mask), gt_start)
            end_loss = nll_loss(allen_util.masked_log_softmax(end_logits, attention_mask), gt_end)
            # We delete squeeze bc it will cause problem when the batch size is 1, and remember the gt_start and gt_end should have shape [B].
            # start_loss = nll_loss(allen_util.masked_log_softmax(start_logits, context_mask), gt_start.squeeze(-1))
            # end_loss = nll_loss(allen_util.masked_log_softmax(end_logits, context_mask), gt_end.squeeze(-1))

            loss = start_loss + end_loss
            return loss
        else:
            return start_logits, end_logits, joint_length


def span_eval(model, data_iter, do_lower_case, fitem_dict, device_num, pred_no_answer=True, save_path=None):
    # fitem_dict in the parameter is the original fitem_dict
    output_fitem_dict = {}

    with torch.no_grad():
        model.eval()

        for batch_idx, batch in enumerate(data_iter):
            batch = allen_util.move_to_device(batch, device_num)
            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
            gt_span = batch['gt_span']

            start_logits, end_logits, context_length = model(mode=BertSpan.ForwardMode.EVAL,
                                                             input_ids=paired_sequence,
                                                             token_type_ids=paired_segments_ids,
                                                             attention_mask=att_mask,
                                                             gt_span=gt_span)
            b_fids = batch['fid']
            b_uids = batch['uid']

            write_to_predicted_fitem(start_logits, end_logits, context_length, b_fids, b_uids, gt_span, fitem_dict,
                                     output_fitem_dict, do_lower_case)

    if save_path is not None:
        common.save_json(output_fitem_dict, save_path)

    eitem_list, eval_dict = merge_predicted_fitem_to_eitem(output_fitem_dict, None, pred_no_answer=pred_no_answer)
    return eitem_list, eval_dict


def model_transfer_go():
    seed = 12
    torch.manual_seed(seed)

    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    bert_model_name = "bert-base-uncased"
    lazy = False
    forward_size = 16
    batch_size = 32
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_rate = 0.1
    learning_rate = 5e-5
    num_train_epochs = 5
    eval_frequency = 2000

    do_lower_case = True

    debug = False

    max_pre_context_length = 315
    max_query_length = 64
    doc_stride = 128
    qa_num_of_layer = 2
    do_ema = True
    ema_device_num = 1
    s_filter_value = 0.5
    s_top_k = 5

    experiment_name = f'model_transfer_(s_top_k:{s_top_k},s_fv:{s_filter_value},qa_layer:{qa_num_of_layer})'

    print("Potential total length:", max_pre_context_length + max_query_length + 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1

    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                              cache_dir=bert_pretrain_path)

    # Load SQuAD v2
    # squad_train_v2 = common.load_json(config.SQUAD_TRAIN_2_0)
    # squad_dev_v2 = common.load_json(config.SQUAD_DEV_2_0)
    #
    # train_eitem_list = preprocessing_squad(squad_train_v2)
    # dev_eitem_list = preprocessing_squad(squad_dev_v2)
    # dev_fitem_dict, dev_fitem_list = eitems_to_fitems(dev_eitem_list, tokenizer, is_training=False,
    #                                                   max_tokens_for_doc=max_pre_context_length, doc_stride=doc_stride,
    #                                                   debug=debug)
    #
    # train_fitem_dict, train_fitem_list = eitems_to_fitems(train_eitem_list, tokenizer, is_training=True,
    #                                                       max_tokens_for_doc=max_pre_context_length,
    #                                                       doc_stride=doc_stride,
    #                                                       debug=debug)

    # Load SQuAD v11
    squad_train_v11 = common.load_json(config.SQUAD_TRAIN_1_1)
    squad_dev_v11 = common.load_json(config.SQUAD_DEV_1_1)

    squad_train_eitem_list = preprocessing_squad(squad_train_v11)
    squad_dev_eitem_list = preprocessing_squad(squad_dev_v11)

    squad_dev_fitem_dict, squad_dev_fitem_list = eitems_to_fitems(squad_dev_eitem_list, tokenizer, is_training=False,
                                                                  max_tokens_for_doc=max_pre_context_length,
                                                                  doc_stride=doc_stride,
                                                                  debug=debug)

    squad_train_fitem_dict, squad_train_fitem_list = eitems_to_fitems(squad_train_eitem_list, tokenizer,
                                                                      is_training=True,
                                                                      max_tokens_for_doc=max_pre_context_length,
                                                                      doc_stride=doc_stride,
                                                                      debug=debug)

    squad_dev_o_dict = list_dict_data_tool.list_to_dict(squad_dev_eitem_list, 'uid')

    # if debug:
    #     dev_list = squad_dev_eitem_list[:100]
    #     eval_frequency = 2

    est_datasize = len(squad_train_fitem_dict)

    span_pred_reader = BertPairedSpanPredReader(bert_tokenizer=tokenizer, lazy=lazy,
                                                example_filter=None)
    bert_encoder = BertModel.from_pretrained(bert_model_name, cache_dir=bert_pretrain_path)
    model = BertSpan(bert_encoder, qa_num_of_layer)

    ema = None
    if do_ema:
        ema = EMA(model, model.named_parameters(), device_num=ema_device_num)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    iterator = BasicIterator(batch_size=forward_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    total_length = len(squad_train_fitem_list)
    print("Total train instances:", total_length)

    num_train_optimization_steps = int(est_datasize / forward_size / gradient_accumulate_step) * \
                                   num_train_epochs

    if debug:
        eval_frequency = 2
        num_train_optimization_steps = 100

    print("Estimated training size", est_datasize)
    print("Number of optimization steps:", num_train_optimization_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_rate,
                         t_total=num_train_optimization_steps)

    squad_dev_instances = span_pred_reader.read(squad_dev_fitem_list)

    forbackward_step = 0
    update_step = 0

    logging_agent = save_tool.ScoreLogger({})

    # # # Create Log File
    file_path_prefix = None
    if not debug:
        file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
        # Save the source code.
        script_name = os.path.basename(__file__)
        with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
            out_f.write(it.read())
            out_f.flush()
    # # # Log File end

    for epoch_i in range(num_train_epochs):
        print("Epoch:", epoch_i)

        print("Resampling:")

        all_train_fitem_list = squad_train_fitem_list
        print("All train size:", len(all_train_fitem_list))
        random.shuffle(all_train_fitem_list)
        train_instances = span_pred_reader.read(all_train_fitem_list)
        train_iter = iterator(train_instances, num_epochs=1, shuffle=True)

        for batch in tqdm(train_iter, desc="Batch Loop"):
            model.train()
            batch = allen_util.move_to_device(batch, device_num)
            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
            gt_span = batch['gt_span']

            loss = model(mode=BertSpan.ForwardMode.TRAIN,
                         input_ids=paired_sequence,
                         token_type_ids=paired_segments_ids,
                         attention_mask=att_mask,
                         gt_span=gt_span)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if gradient_accumulate_step > 1:
                loss = loss / gradient_accumulate_step

            loss.backward()
            forbackward_step += 1

            if forbackward_step % gradient_accumulate_step == 0:
                optimizer.step()
                if ema is not None and do_ema:
                    updated_model = model.module if hasattr(model, 'module') else model
                    ema(updated_model.named_parameters())
                optimizer.zero_grad()
                update_step += 1

                if update_step % eval_frequency == 0:
                    print("Non-EMA EVAL:")
                    # eval_iter = iterator(dev_instances, num_epochs=1, shuffle=False)
                    # cur_eitem_list, cur_eval_dict = span_eval(model, eval_iter, do_lower_case, dev_fitem_dict,
                    #                                           device_num)
                    # cur_results_dict = dict()
                    # cur_results_dict['p_answer'] = cur_eval_dict
                    # cur_results_dict['sp'] = dev_sp_results_dict
                    #
                    # _, metrics = ext_hotpot_eval.eval(cur_results_dict, dev_list, verbose=False)
                    # # print(metrics)
                    #
                    logging_item = {
                        'score': "None",
                    }
                    #
                    # joint_f1 = metrics['joint_f1']
                    # joint_em = metrics['joint_em']
                    #
                    # print(logging_item)

                    # if not debug:
                    #     save_file_name = f'i({update_step})|e({epoch_i})' \
                    #         f'|j_f1({joint_f1})|j_em({joint_em})|seed({seed})'
                    #
                    #     # print(save_file_name)
                    #     logging_agent.incorporate_results({}, save_file_name, logging_item)
                    #     logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")
                    #
                    #     model_to_save = model.module if hasattr(model, 'module') else model
                    #     output_model_file = Path(file_path_prefix) / save_file_name
                    #     torch.save(model_to_save.state_dict(), str(output_model_file))

                    if do_ema and ema is not None:
                        print("EMA EVAL")
                        ema_model = ema.get_inference_model()
                        ema_model.eval()
                        ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=ema_device_num)
                        ema_model = ema_model.to(ema_device_num)
                        ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)
                        dev_iter = iterator(squad_dev_instances, num_epochs=1, shuffle=False)
                        cur_eitem_list, cur_eval_dict = span_eval(ema_model, dev_iter, do_lower_case,
                                                                  squad_dev_fitem_dict,
                                                                  ema_device_num, pred_no_answer=True)

                        # cur_results_dict = dict()
                        # cur_results_dict['p_answer'] = cur_eval_dict
                        # cur_results_dict['sp'] = dev_sp_results_dict

                        print(update_step, epoch_i)
                        evaluation.squad_eval_v1.get_score(cur_eval_dict, squad_dev_v11['data'])

                        # _, metrics = ext_hotpot_eval.eval(cur_results_dict, dev_list, verbose=False)
                        # print(metrics)
                        # print("---------------" * 3)
                        #
                        # logging_item = {
                        #     'label': 'ema',
                        #     'score': metrics,
                        # }
                        #
                        # joint_f1 = metrics['joint_f1']
                        # joint_em = metrics['joint_em']
                        #
                        # print(logging_item)

                        if not debug:
                            save_file_name = f'ema_i({update_step})|e({epoch_i})|seed({seed})'
                            # print(save_file_name)
                            logging_agent.incorporate_results({}, save_file_name, logging_item)
                            logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

                            model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
                            output_model_file = Path(file_path_prefix) / save_file_name
                            torch.save(model_to_save.state_dict(), str(output_model_file))


def pure_transfer_eval(model_path):
    seed = 12
    torch.manual_seed(seed)
    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    bert_model_name = "bert-base-uncased"
    lazy = False
    forward_size = 16
    batch_size = 32
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_rate = 0.1
    learning_rate = 5e-5
    num_train_epochs = 5
    eval_frequency = 2000

    do_lower_case = True

    debug = False

    max_pre_context_length = 315
    max_query_length = 64
    doc_stride = 128
    qa_num_of_layer = 2
    do_ema = True
    ema_device_num = 1

    top_k = 10
    filter_value = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1

    n_gpu = torch.cuda.device_count()

    # CuratedTREC
    d_list = common.load_jsonl(config.OPEN_CURATEDTERC_TEST_GT)
    # d_list = common.load_jsonl(config.OPEN_WEBQ_TEST_GT)
    # d_list = common.load_jsonl(config.OPEN_WIKIM_TEST_GT)
    # upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-08:44:38_mtr_open_qa_p_level_(num_train_epochs:3)/i(2000)|e(2)|squad|top10(0.6909176915799432)|top20(0.7103122043519394)|seed(12)_eval_results.jsonl'
    upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|curatedtrec|top10(0.8069164265129684)|top20(0.8170028818443804)|seed(12)_eval_results.jsonl'
    # upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|webq|top10(0.655511811023622)|top20(0.6756889763779528)|seed(12)_eval_results.jsonl'
    # upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|wikimovie|top10(0.8491760450160771)|top20(0.8512861736334405)|seed(12)_eval_results.jsonl'
    cur_eval_results_list = common.load_jsonl(upstream_p_file_name)

    # match_type = 'string'
    match_type = 'regex'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                              cache_dir=bert_pretrain_path)

    dev_fitems_dict, dev_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(d_list, cur_eval_results_list,
                                                                                    False,
                                                                                    tokenizer, max_pre_context_length,
                                                                                    max_query_length, doc_stride,
                                                                                    debug, top_k, filter_value,
                                                                                    match_type)
    # dev_gt_list = common.load_jsonl(config.OPEN_WIKIM_TEST_GT)
    # dev_gt_list = common.load_jsonl(config.OPEN_WEBQ_TEST_GT)
    dev_gt_list = common.load_jsonl(config.OPEN_CURATEDTERC_TEST_GT)

    # Train it todo later
    est_datasize = len([])

    span_pred_reader = BertPairedSpanPredReader(bert_tokenizer=tokenizer, lazy=lazy,
                                                example_filter=None)
    bert_encoder = BertModel.from_pretrained(bert_model_name, cache_dir=bert_pretrain_path)
    model = BertSpan(bert_encoder, qa_num_of_layer)

    model.load_state_dict(torch.load(model_path))

    ema = None
    if do_ema:
        ema = EMA(model, model.named_parameters(), device_num=ema_device_num)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    iterator = BasicIterator(batch_size=forward_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    total_length = len([])
    print("Total train instances:", total_length)

    num_train_optimization_steps = int(est_datasize / forward_size / gradient_accumulate_step) * \
                                   num_train_epochs

    if debug:
        eval_frequency = 2
        num_train_optimization_steps = 100

    print("Estimated training size", est_datasize)
    print("Number of optimization steps:", num_train_optimization_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_rate,
                         t_total=num_train_optimization_steps)

    dev_instances = span_pred_reader.read(dev_fitems_list)

    forbackward_step = 0
    update_step = 0

    logging_agent = save_tool.ScoreLogger({})

    ema_model = ema.get_inference_model()
    ema_model.eval()
    ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=ema_device_num)
    ema_model = ema_model.to(ema_device_num)
    ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)
    dev_iter = iterator(dev_instances, num_epochs=1, shuffle=False)
    cur_eitem_list, cur_eval_dict = span_eval(ema_model, dev_iter, do_lower_case,
                                              dev_fitems_dict,
                                              ema_device_num, pred_no_answer=False)
    predict_dict = {
        'p_answer': cur_eval_dict
    }

    open_domain_qa_eval.qa_eval(predict_dict, dev_gt_list, type=match_type)

    # # # Create Log File
    # file_path_prefix = None
    # if not debug:
    #     file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
    #     Save the source code.
    # script_name = os.path.basename(__file__)
    # with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
    #     out_f.write(it.read())
    #     out_f.flush()
    # # # Log File end

    # for epoch_i in range(num_train_epochs):
    #     print("Epoch:", epoch_i)
    #
    #     print("Resampling:")
    #
    #     all_train_fitem_list = squad_train_fitem_list
    #     print("All train size:", len(all_train_fitem_list))
    #     random.shuffle(all_train_fitem_list)
    #     train_instances = span_pred_reader.read(all_train_fitem_list)
    #     train_iter = iterator(train_instances, num_epochs=1, shuffle=True)
    #
    #     for batch in tqdm(train_iter, desc="Batch Loop"):
    #         model.train()
    #         batch = allen_util.move_to_device(batch, device_num)
    #         paired_sequence = batch['paired_sequence']
    #         paired_segments_ids = batch['paired_segments_ids']
    #         att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
    #         gt_span = batch['gt_span']
    #
    #         loss = model(mode=BertSpan.ForwardMode.TRAIN,
    #                      input_ids=paired_sequence,
    #                      token_type_ids=paired_segments_ids,
    #                      attention_mask=att_mask,
    #                      gt_span=gt_span)
    #
    #         if n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu.
    #
    #         if gradient_accumulate_step > 1:
    #             loss = loss / gradient_accumulate_step
    #
    #         loss.backward()
    #         forbackward_step += 1
    #
    #         if forbackward_step % gradient_accumulate_step == 0:
    #             optimizer.step()
    #             if ema is not None and do_ema:
    #                 updated_model = model.module if hasattr(model, 'module') else model
    #                 ema(updated_model.named_parameters())
    #             optimizer.zero_grad()
    #             update_step += 1
    #
    #             if update_step % eval_frequency == 0:
    #                 print("Non-EMA EVAL:")
    #                 # eval_iter = iterator(dev_instances, num_epochs=1, shuffle=False)
    #                 # cur_eitem_list, cur_eval_dict = span_eval(model, eval_iter, do_lower_case, dev_fitem_dict,
    #                 #                                           device_num)
    #                 # cur_results_dict = dict()
    #                 # cur_results_dict['p_answer'] = cur_eval_dict
    #                 # cur_results_dict['sp'] = dev_sp_results_dict
    #                 #
    #                 # _, metrics = ext_hotpot_eval.eval(cur_results_dict, dev_list, verbose=False)
    #                 # # print(metrics)
    #                 #
    #                 logging_item = {
    #                     'score': "None",
    #                 }
    #                 #
    #                 # joint_f1 = metrics['joint_f1']
    #                 # joint_em = metrics['joint_em']
    #                 #
    #                 # print(logging_item)
    #
    #                 # if not debug:
    #                 #     save_file_name = f'i({update_step})|e({epoch_i})' \
    #                 #         f'|j_f1({joint_f1})|j_em({joint_em})|seed({seed})'
    #                 #
    #                 #     # print(save_file_name)
    #                 #     logging_agent.incorporate_results({}, save_file_name, logging_item)
    #                 #     logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")
    #                 #
    #                 #     model_to_save = model.module if hasattr(model, 'module') else model
    #                 #     output_model_file = Path(file_path_prefix) / save_file_name
    #                 #     torch.save(model_to_save.state_dict(), str(output_model_file))
    #
    #                 if do_ema and ema is not None:
    #                     print("EMA EVAL")
    #                     ema_model = ema.get_inference_model()
    #                     ema_model.eval()
    #                     ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=ema_device_num)
    #                     ema_model = ema_model.to(ema_device_num)
    #                     ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)
    #                     dev_iter = iterator(squad_dev_instances, num_epochs=1, shuffle=False)
    #                     cur_eitem_list, cur_eval_dict = span_eval(ema_model, dev_iter, do_lower_case,
    #                                                               squad_dev_fitem_dict,
    #                                                               ema_device_num, pred_no_answer=True)
    #
    #                     # cur_results_dict = dict()
    #                     # cur_results_dict['p_answer'] = cur_eval_dict
    #                     # cur_results_dict['sp'] = dev_sp_results_dict
    #
    #                     print(update_step, epoch_i)
    #                     evaluation.squad_eval_v1.get_score(cur_eval_dict, squad_dev_v11['data'])
    #
    #                     # _, metrics = ext_hotpot_eval.eval(cur_results_dict, dev_list, verbose=False)
    #                     # print(metrics)
    #                     # print("---------------" * 3)
    #                     #
    #                     # logging_item = {
    #                     #     'label': 'ema',
    #                     #     'score': metrics,
    #                     # }
    #                     #
    #                     # joint_f1 = metrics['joint_f1']
    #                     # joint_em = metrics['joint_em']
    #                     #
    #                     # print(logging_item)
    #
    #                     if not debug:
    #                         save_file_name = f'ema_i({update_step})|e({epoch_i})|seed({seed})'
    #                         # print(save_file_name)
    #                         logging_agent.incorporate_results({}, save_file_name, logging_item)
    #                         logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")
    #
    #                         model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
    #                         output_model_file = Path(file_path_prefix) / save_file_name
    #                         torch.save(model_to_save.state_dict(), str(output_model_file))


def fine_tune_train_webq(model_path):
    seed = 12
    torch.manual_seed(seed)
    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    bert_model_name = "bert-base-uncased"
    lazy = False
    forward_size = 16
    batch_size = 32
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_rate = 0.1
    learning_rate = 5e-5
    num_train_epochs = 5
    eval_frequency = 100
    use_pre_trained = True

    do_lower_case = True

    # dataset = 'curatedtrec'
    dataset = 'webq'
    experiment_name = f'ft_dataset{dataset}_pret{use_pre_trained}'

    debug = False

    max_pre_context_length = 315
    max_query_length = 64
    doc_stride = 128
    qa_num_of_layer = 2
    do_ema = True
    ema_device_num = 1

    top_k = 5
    filter_value = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1

    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                              cache_dir=bert_pretrain_path)

    # Load dev data
    d_list = common.load_jsonl(config.OPEN_WEBQ_TEST_GT)
    dev_gt_list = common.load_jsonl(config.OPEN_WEBQ_TEST_GT)
    # d_list = common.load_jsonl(config.OPEN_CURATEDTERC_TEST_GT)
    # dev_gt_list = common.load_jsonl(config.OPEN_CURATEDTERC_TEST_GT)

    # upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|curatedtrec|top10(0.8069164265129684)|top20(0.8170028818443804)|seed(12)_eval_results.jsonl'
    upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|webq|top10(0.655511811023622)|top20(0.6756889763779528)|seed(12)_eval_results.jsonl'
    cur_eval_results_list = common.load_jsonl(upstream_p_file_name)

    match_type = 'regex'
    dev_fitems_dict, dev_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(d_list, cur_eval_results_list,
                                                                                    False,
                                                                                    tokenizer, max_pre_context_length,
                                                                                    max_query_length, doc_stride,
                                                                                    debug, top_k, filter_value,
                                                                                    match_type)

    # d_list = common.load_jsonl(config.OPEN_CURATEDTERC_TEST_GT)
    # d_list = common.load_jsonl(config.OPEN_WIKIM_TEST_GT)
    # upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-08:44:38_mtr_open_qa_p_level_(num_train_epochs:3)/i(2000)|e(2)|squad|top10(0.6909176915799432)|top20(0.7103122043519394)|seed(12)_eval_results.jsonl'
    # upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|curatedtrec|top10(0.8069164265129684)|top20(0.8170028818443804)|seed(12)_eval_results.jsonl'

    # upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|wikimovie|top10(0.8491760450160771)|top20(0.8512861736334405)|seed(12)_eval_results.jsonl'

    # match_type = 'string'
    # dev_gt_list = common.load_jsonl(config.OPEN_WIKIM_TEST_GT)
    # dev_gt_list = common.load_jsonl(config.OPEN_WEBQ_TEST_GT)

    # Load train data
    train_d_list = common.load_jsonl(config.OPEN_WEBQ_TRAIN_GT)
    # train_d_list = common.load_jsonl(config.OPEN_CURATEDTERC_TRAIN_GT)
    # upstream_train_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(3)_final_ema_model_curatedtrec_train_p_level_eval.jsonl'
    upstream_train_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(3)_final_ema_model_webq_train_p_level_eval.jsonl'
    cur_upstream_train_p_results_list = common.load_jsonl(upstream_train_p_file_name)

    train_fitems_dict, train_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(train_d_list,
                                                                                        cur_upstream_train_p_results_list,
                                                                                        True,
                                                                                        tokenizer,
                                                                                        max_pre_context_length,
                                                                                        max_query_length, doc_stride,
                                                                                        debug, top_k, filter_value,
                                                                                        match_type)

    # Train it
    est_datasize = len(train_fitems_list)

    span_pred_reader = BertPairedSpanPredReader(bert_tokenizer=tokenizer, lazy=lazy,
                                                example_filter=None)
    bert_encoder = BertModel.from_pretrained(bert_model_name, cache_dir=bert_pretrain_path)
    model = BertSpan(bert_encoder, qa_num_of_layer)

    if use_pre_trained:
        model.load_state_dict(torch.load(model_path))

    ema = None
    if do_ema:
        ema = EMA(model, model.named_parameters(), device_num=ema_device_num)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    iterator = BasicIterator(batch_size=forward_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    total_length = est_datasize
    print("Total train instances:", total_length)

    num_train_optimization_steps = int(est_datasize / forward_size / gradient_accumulate_step) * \
                                   num_train_epochs

    if debug:
        eval_frequency = 2
        num_train_optimization_steps = 100

    print("Estimated training size", est_datasize)
    print("Number of optimization steps:", num_train_optimization_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_rate,
                         t_total=num_train_optimization_steps)

    dev_instances = span_pred_reader.read(dev_fitems_list)

    forbackward_step = 0
    update_step = 0

    logging_agent = save_tool.ScoreLogger({})

    # ema_model = ema.get_inference_model()
    # ema_model.eval()
    # ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=ema_device_num)
    # ema_model = ema_model.to(ema_device_num)
    # ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)
    # dev_iter = iterator(dev_instances, num_epochs=1, shuffle=False)
    # cur_eitem_list, cur_eval_dict = span_eval(ema_model, dev_iter, do_lower_case,
    #                                           dev_fitems_dict,
    #                                           ema_device_num, pred_no_answer=False)
    # predict_dict = {
    #     'p_answer': cur_eval_dict
    # }
    #
    # open_domain_qa_eval.qa_eval(predict_dict, dev_gt_list, type=match_type)

    # # Create Log File
    file_path_prefix = None
    if not debug:
        file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
        # Save the source code.
        script_name = os.path.basename(__file__)
        with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
            out_f.write(it.read())
            out_f.flush()
    # # Log File end

    for epoch_i in range(num_train_epochs):
        print("Epoch:", epoch_i)

        print("Resampling:")

        all_train_fitem_list = train_fitems_list
        print("All train size:", len(all_train_fitem_list))
        random.shuffle(all_train_fitem_list)
        train_instances = span_pred_reader.read(all_train_fitem_list)
        train_iter = iterator(train_instances, num_epochs=1, shuffle=True)
        #
        for batch in tqdm(train_iter, desc="Batch Loop"):
            model.train()
            batch = allen_util.move_to_device(batch, device_num)
            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
            gt_span = batch['gt_span']

            loss = model(mode=BertSpan.ForwardMode.TRAIN,
                         input_ids=paired_sequence,
                         token_type_ids=paired_segments_ids,
                         attention_mask=att_mask,
                         gt_span=gt_span)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if gradient_accumulate_step > 1:
                loss = loss / gradient_accumulate_step

            loss.backward()
            forbackward_step += 1

            if forbackward_step % gradient_accumulate_step == 0:
                optimizer.step()
                if ema is not None and do_ema:
                    updated_model = model.module if hasattr(model, 'module') else model
                    ema(updated_model.named_parameters())
                optimizer.zero_grad()
                update_step += 1

                if update_step % eval_frequency == 0:
                    print("Non-EMA EVAL:")
                    #                 # eval_iter = iterator(dev_instances, num_epochs=1, shuffle=False)
                    #                 # cur_eitem_list, cur_eval_dict = span_eval(model, eval_iter, do_lower_case, dev_fitem_dict,
                    #                 #                                           device_num)
                    #                 # cur_results_dict = dict()
                    #                 # cur_results_dict['p_answer'] = cur_eval_dict
                    #                 # cur_results_dict['sp'] = dev_sp_results_dict
                    #                 #
                    #                 # _, metrics = ext_hotpot_eval.eval(cur_results_dict, dev_list, verbose=False)
                    #                 # # print(metrics)
                    #                 #
                    #                 logging_item = {
                    #                     'score': "None",
                    #                 }
                    #                 #
                    #                 # joint_f1 = metrics['joint_f1']
                    #                 # joint_em = metrics['joint_em']
                    #                 #
                    #                 # print(logging_item)
                    #
                    #                 # if not debug:
                    #                 #     save_file_name = f'i({update_step})|e({epoch_i})' \
                    #                 #         f'|j_f1({joint_f1})|j_em({joint_em})|seed({seed})'
                    #                 #
                    #                 #     # print(save_file_name)
                    #                 #     logging_agent.incorporate_results({}, save_file_name, logging_item)
                    #                 #     logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")
                    #                 #
                    #                 #     model_to_save = model.module if hasattr(model, 'module') else model
                    #                 #     output_model_file = Path(file_path_prefix) / save_file_name
                    #                 #     torch.save(model_to_save.state_dict(), str(output_model_file))
                    #
                    if do_ema and ema is not None:
                        eval_qa_task(ema, ema_device_num, iterator, dev_instances, dev_fitems_dict, do_lower_case,
                                     'curated', 'test', dev_gt_list, match_type, update_step, epoch_i, seed,
                                     file_path_prefix, debug,
                                     logging_agent)
                        # ema_model = ema.get_inference_model()
                        # ema_model.eval()
                        # ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=ema_device_num)
                        # ema_model = ema_model.to(ema_device_num)
                        # ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)
                        # dev_iter = iterator(dev_instances, num_epochs=1, shuffle=False)
                        # cur_eitem_list, cur_eval_dict = span_eval(ema_model, dev_iter, do_lower_case,
                        #                                           dev_fitems_dict,
                        #                                           ema_device_num, pred_no_answer=False)
                        # predict_dict = {
                        #     'p_answer': cur_eval_dict
                        # }
                        #
                        # _, metric = open_domain_qa_eval.qa_eval(predict_dict, dev_gt_list, type=match_type)
                        # f1 = metric['f1']
                        # em = metric['em']
                        #
                        # # print(f"EM/F1:{em}/{f1}")
                        # logging_item = {
                        #     'label': 'ema',
                        #     'score': metric,
                        # }
                        #
                        # print(logging_item)
                        #
                        # if not debug:
                        #     save_file_name = f'ema_i({update_step})|e({epoch_i})|em({em})|f1({f1})|seed({seed})'
                        #     # print(save_file_name)
                        #     logging_agent.incorporate_results({}, save_file_name, logging_item)
                        #     logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")
                        #
                        #     model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
                        #     output_model_file = Path(file_path_prefix) / save_file_name
                        #     torch.save(model_to_save.state_dict(), str(output_model_file))


def multitask_qa():
    seed = 12
    torch.manual_seed(seed)
    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    bert_model_name = "bert-base-uncased"
    lazy = False
    forward_size = 32
    batch_size = 32
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_rate = 0.1
    learning_rate = 5e-5
    num_train_epochs = 5
    eval_frequency = 5000
    use_pre_trained = False

    do_lower_case = True

    # dataset = 'curatedtrec'
    experiment_name = f'multitask_qa_pret{use_pre_trained}'

    debug = False

    max_pre_context_length = 315
    max_query_length = 64
    doc_stride = 128
    qa_num_of_layer = 2
    do_ema = True
    ema_device_num = 1

    top_k = 5
    filter_value = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1

    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                              cache_dir=bert_pretrain_path)

    # Load dev data
    # CuratedTREC curatedtrec
    curatedtrec_d_list = common.load_jsonl(config.OPEN_CURATEDTERC_TEST_GT)
    curatedtrec_dev_gt_list = common.load_jsonl(config.OPEN_CURATEDTERC_TEST_GT)

    curatedtrec_upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|curatedtrec|top10(0.8069164265129684)|top20(0.8170028818443804)|seed(12)_eval_results.jsonl'
    curatedtrec_cur_eval_results_list = common.load_jsonl(curatedtrec_upstream_p_file_name)

    curatedtrec_dev_fitems_dict, curatedtrec_dev_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(
        curatedtrec_d_list, curatedtrec_cur_eval_results_list,
        False,
        tokenizer,
        max_pre_context_length,
        max_query_length, doc_stride,
        debug, top_k, filter_value,
        'regex')

    # WikiM wikimovie
    wikimovie_d_list = common.load_jsonl(config.OPEN_WIKIM_TEST_GT)
    wikimovie_dev_gt_list = common.load_jsonl(config.OPEN_WIKIM_TEST_GT)

    wikimovie_upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|wikimovie|top10(0.8491760450160771)|top20(0.8512861736334405)|seed(12)_eval_results.jsonl'
    wikimovie_cur_eval_results_list = common.load_jsonl(wikimovie_upstream_p_file_name)

    wikimovie_dev_fitems_dict, wikimovie_dev_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(
        wikimovie_d_list, wikimovie_cur_eval_results_list,
        False,
        tokenizer,
        max_pre_context_length,
        max_query_length, doc_stride,
        debug, top_k, filter_value,
        'string')

    # SQuAD
    squad_d_list = common.load_jsonl(config.OPEN_SQUAD_DEV_GT)
    squad_dev_gt_list = common.load_jsonl(config.OPEN_SQUAD_DEV_GT)

    squad_upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|squad|top10(0.6892147587511825)|top20(0.7087038789025544)|seed(12)_eval_results.jsonl'
    squad_cur_eval_results_list = common.load_jsonl(squad_upstream_p_file_name)

    squad_dev_fitems_dict, squad_dev_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(
        squad_d_list, squad_cur_eval_results_list,
        False,
        tokenizer,
        max_pre_context_length,
        max_query_length, doc_stride,
        debug, top_k, filter_value,
        'string')

    # WebQ
    webq_d_list = common.load_jsonl(config.OPEN_WEBQ_TEST_GT)
    webq_dev_gt_list = common.load_jsonl(config.OPEN_WEBQ_TEST_GT)

    webq_upstream_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(2)|webq|top10(0.655511811023622)|top20(0.6756889763779528)|seed(12)_eval_results.jsonl'
    webq_cur_eval_results_list = common.load_jsonl(webq_upstream_p_file_name)

    webq_dev_fitems_dict, webq_dev_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(
        webq_d_list, webq_cur_eval_results_list,
        False,
        tokenizer,
        max_pre_context_length,
        max_query_length, doc_stride,
        debug, top_k, filter_value,
        'string')

    # Load train data
    # Curatedtrec
    curatedtrec_train_d_list = common.load_jsonl(config.OPEN_CURATEDTERC_TRAIN_GT)
    curatedtrec_upstream_train_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(3)_final_ema_model_curatedtrec_train_p_level_eval.jsonl'
    curatedtrec_cur_upstream_train_p_results_list = common.load_jsonl(curatedtrec_upstream_train_p_file_name)

    curatedtrec_train_fitems_dict, curatedtrec_train_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(
        curatedtrec_train_d_list,
        curatedtrec_cur_upstream_train_p_results_list,
        True,
        tokenizer,
        max_pre_context_length,
        max_query_length,
        doc_stride,
        debug, top_k, filter_value,
        'regex')

    # WikiM wikimovie
    wikimovie_train_d_list = common.load_jsonl(config.OPEN_WIKIM_TRAIN_GT)
    wikimovie_upstream_train_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(3)_final_ema_model_wikimovie_train_p_level_eval.jsonl'
    wikimovie_cur_upstream_train_p_results_list = common.load_jsonl(wikimovie_upstream_train_p_file_name)

    wikimovie_train_fitems_dict, wikimovie_train_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(
        wikimovie_train_d_list,
        wikimovie_cur_upstream_train_p_results_list,
        True,
        tokenizer,
        max_pre_context_length,
        max_query_length,
        doc_stride,
        debug, top_k, filter_value,
        'string')

    # SQuAD
    squad_train_d_list = common.load_jsonl(config.OPEN_SQUAD_TRAIN_GT)
    squad_upstream_train_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(3)_final_ema_model_squad_train_p_level_eval.jsonl'
    squad_cur_upstream_train_p_results_list = common.load_jsonl(squad_upstream_train_p_file_name)

    squad_train_fitems_dict, squad_train_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(
        squad_train_d_list,
        squad_cur_upstream_train_p_results_list,
        True,
        tokenizer,
        max_pre_context_length,
        max_query_length,
        doc_stride,
        debug, top_k, filter_value,
        'string')

    # WebQ
    webq_train_d_list = common.load_jsonl(config.OPEN_WEBQ_TRAIN_GT)
    webq_upstream_train_p_file_name = config.PRO_ROOT / 'saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(3)_final_ema_model_webq_train_p_level_eval.jsonl'
    webq_cur_upstream_train_p_results_list = common.load_jsonl(webq_upstream_train_p_file_name)

    webq_train_fitems_dict, webq_train_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(
        webq_train_d_list,
        webq_cur_upstream_train_p_results_list,
        True,
        tokenizer,
        max_pre_context_length,
        max_query_length,
        doc_stride,
        debug, top_k, filter_value,
        'string')

    # SQuAD v11
    squad_train_v11 = common.load_json(config.SQUAD_TRAIN_1_1)

    squad_train_eitem_list = preprocessing_squad(squad_train_v11)

    squad_v11_train_fitem_dict, squad_v11_train_fitem_list = eitems_to_fitems(squad_train_eitem_list, tokenizer,
                                                                              is_training=True,
                                                                              max_tokens_for_doc=max_pre_context_length,
                                                                              doc_stride=doc_stride,
                                                                              debug=debug)

    # Train it
    print(f"Train size Curated/WikiM/SQuAD/WebQ : "
          f"{len(curatedtrec_train_fitems_list)}/{len(wikimovie_train_fitems_list)}/{len(squad_train_fitems_list)}/{len(webq_train_fitems_list)}")

    print(f"Dev size Curated/WikiM/SQuAD/WebQ : "
          f"{len(curatedtrec_dev_fitems_list)}/{len(wikimovie_dev_fitems_list)}/{len(squad_dev_fitems_list)}/{len(webq_dev_fitems_list)}")

    print("SQuAD v11 size:", len(squad_v11_train_fitem_list))

    est_datasize = len(curatedtrec_train_fitems_list) + len(wikimovie_train_fitems_list) + \
                   len(squad_train_fitems_list) + len(webq_train_fitems_list) + len(squad_v11_train_fitem_list)

    span_pred_reader = BertPairedSpanPredReader(bert_tokenizer=tokenizer, lazy=lazy,
                                                example_filter=None)
    bert_encoder = BertModel.from_pretrained(bert_model_name, cache_dir=bert_pretrain_path)
    model = BertSpan(bert_encoder, qa_num_of_layer)

    # if use_pre_trained:
    #     model.load_state_dict(torch.load(model_path))

    ema = None
    if do_ema:
        ema = EMA(model, model.named_parameters(), device_num=ema_device_num)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    iterator = BasicIterator(batch_size=forward_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    total_length = est_datasize
    print("Total train instances:", total_length)

    num_train_optimization_steps = int(est_datasize / forward_size / gradient_accumulate_step) * \
                                   num_train_epochs

    if debug:
        eval_frequency = 2
        num_train_optimization_steps = 100

    print("Estimated training size", est_datasize)
    print("Number of optimization steps:", num_train_optimization_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_rate,
                         t_total=num_train_optimization_steps)

    curatedtrec_dev_instances = span_pred_reader.read(curatedtrec_dev_fitems_list)
    squad_dev_instances = span_pred_reader.read(squad_dev_fitems_list)
    wikimovie_dev_instances = span_pred_reader.read(wikimovie_dev_fitems_list)
    webq_dev_instances = span_pred_reader.read(webq_dev_fitems_list)

    forbackward_step = 0
    update_step = 0

    logging_agent = save_tool.ScoreLogger({})

    # # Create Log File
    file_path_prefix = "."
    if not debug:
        file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
        # Save the source code.
        script_name = os.path.basename(__file__)
        with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
            out_f.write(it.read())
            out_f.flush()
    # # Log File end

    for epoch_i in range(num_train_epochs):
        print("Epoch:", epoch_i)

        print("Resampling:")

        all_train_fitem_list = squad_v11_train_fitem_list + squad_train_fitems_list + \
                               webq_train_fitems_list + wikimovie_train_fitems_list + curatedtrec_train_fitems_list

        print("All train size:", len(all_train_fitem_list))
        random.shuffle(all_train_fitem_list)
        train_instances = span_pred_reader.read(all_train_fitem_list)
        train_iter = iterator(train_instances, num_epochs=1, shuffle=True)
        #
        for batch in tqdm(train_iter, desc="Batch Loop"):
            model.train()
            batch = allen_util.move_to_device(batch, device_num)
            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
            gt_span = batch['gt_span']

            loss = model(mode=BertSpan.ForwardMode.TRAIN,
                         input_ids=paired_sequence,
                         token_type_ids=paired_segments_ids,
                         attention_mask=att_mask,
                         gt_span=gt_span)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if gradient_accumulate_step > 1:
                loss = loss / gradient_accumulate_step

            loss.backward()
            forbackward_step += 1

            if forbackward_step % gradient_accumulate_step == 0:
                optimizer.step()
                if ema is not None and do_ema:
                    updated_model = model.module if hasattr(model, 'module') else model
                    ema(updated_model.named_parameters())
                optimizer.zero_grad()
                update_step += 1

                if update_step % eval_frequency == 0:
                    print("Non-EMA EVAL:")
                    if do_ema and ema is not None:
                        eval_qa_task(ema, ema_device_num, iterator, curatedtrec_dev_instances,
                                     curatedtrec_dev_fitems_dict, do_lower_case,
                                     'curatedtrec', 'test', curatedtrec_dev_gt_list, 'regex', update_step, epoch_i, seed,
                                     file_path_prefix, debug,
                                     logging_agent)

                        eval_qa_task(ema, ema_device_num, iterator, webq_dev_instances,
                                     webq_dev_fitems_dict, do_lower_case,
                                     'webq', 'test', webq_dev_gt_list, 'string', update_step, epoch_i,
                                     seed,
                                     file_path_prefix, debug,
                                     logging_agent)

                        eval_qa_task(ema, ema_device_num, iterator, squad_dev_instances,
                                     squad_dev_fitems_dict, do_lower_case,
                                     'squad', 'test', squad_dev_gt_list, 'string', update_step, epoch_i,
                                     seed,
                                     file_path_prefix, debug,
                                     logging_agent)

                        eval_qa_task(ema, ema_device_num, iterator, wikimovie_dev_instances,
                                     wikimovie_dev_fitems_dict, do_lower_case,
                                     'wikimovie', 'test', wikimovie_dev_gt_list, 'string', update_step, epoch_i,
                                     seed,
                                     file_path_prefix, debug,
                                     logging_agent)


def eval_qa_task(ema, ema_device_num, iterator, dev_instances, dev_fitems_dict, do_lower_case,
                 datasetname, tag, dev_gt_list, match_type, update_step, epoch_i, seed, file_path_prefix, debug,
                 logging_agent):
    ema_model = ema.get_inference_model()
    ema_model.eval()
    ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=ema_device_num)
    ema_model = ema_model.to(ema_device_num)
    ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)
    dev_iter = iterator(dev_instances, num_epochs=1, shuffle=False)

    result_save_file_name = f'ema_i({update_step})|e({epoch_i})|dataset({datasetname})|seed({seed})_qa_results.json'

    cur_eitem_list, cur_eval_dict = span_eval(ema_model, dev_iter, do_lower_case,
                                              dev_fitems_dict,
                                              ema_device_num, pred_no_answer=False,
                                              save_path=Path(file_path_prefix) / result_save_file_name)
    predict_dict = {
        'p_answer': cur_eval_dict
    }

    _, metric = open_domain_qa_eval.qa_eval(predict_dict, dev_gt_list, type=match_type, missing_ignore=True)
    f1 = metric['f1']
    em = metric['em']

    # print(f"EM/F1:{em}/{f1}")
    logging_item = {
        'label': 'ema',
        'dataset': datasetname,
        'score': metric,
    }

    print(logging_item)

    if not debug:
        save_file_name = f'ema_i({update_step})|e({epoch_i})|dataset({datasetname})|em({em})|f1({f1})|seed({seed})'
        # print(save_file_name)
        logging_agent.incorporate_results({}, save_file_name, logging_item)
        logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

        model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
        output_model_file = Path(file_path_prefix) / save_file_name
        torch.save(model_to_save.state_dict(), str(output_model_file))


if __name__ == '__main__':
    model_path = config.PRO_ROOT / "saved_models/05-13-10:32:40_model_transfer_(s_top_k:5,s_fv:0.5,qa_layer:2)/ema_i(6000)|e(2)|seed(12)"
    fine_tune_train_webq(model_path)
    # multitask_qa()
    # pure_transfer_eval(model_path)
    # model_transfer_go()
    # utest_on_squad_v2()
