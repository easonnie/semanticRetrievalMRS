import os
import random
from pathlib import Path
import math

from allennlp.data.iterators import BasicIterator
from allennlp.nn.util import move_to_device
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam

import config
from bert_model_variances.bert_maxout_clf import BertPairMaxOutMatcher
from bert_model_variances.bert_multilayer_output import BertMultiLayerSeqClassification
from data_utils.exvocab import ExVocabulary
from data_utils.readers.bert_reader_fever_sent_selection import BertContentSelectionReader
# from evaluation import ext_hotpot_eval
from data_utils.readers.bert_reader_nli import BertFeverNLIReader
from evaluation import fever_scorer
from fever_models.sentence_selection.bert_v1 import select_top_k_and_to_results_dict
from fever_sampler.nli_new_sampler import build_nli_forward_item
from fever_sampler.ss_sampler import build_full_wiki_document_forward_item, down_sample_neg
from fever_utils import fever_db
from flint import torch_util
# from hotpot_data_analysis.fullwiki_provided_upperbound import append_gt_downstream_to_get_upperbound_from_doc_retri

from utils import common, list_dict_data_tool
import torch
from tqdm import tqdm
import numpy as np
import copy
import allennlp
from utils import save_tool
import torch.nn.functional as F


def eval_model(model, data_iter, device_num, with_probs=False, make_int=False, show_progress=False,
               feed_input_span=False):
    id2label = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO"
    }

    print("Evaluating ...")
    tqdm_disable = not show_progress
    with torch.no_grad():
        model.eval()
        totoal_size = 0

        y_pred_list = []
        y_fid_list = []
        y_pid_list = []
        y_element_list = []

        y_logits_list = []
        y_probs_list = []

        for batch_idx, batch in tqdm(enumerate(data_iter), disable=tqdm_disable):
            batch = move_to_device(batch, device_num)

            eval_paired_sequence = batch['paired_sequence']
            eval_paired_segments_ids = batch['paired_segments_ids']
            eval_labels_ids = batch['label']
            eval_att_mask, _ = torch_util.get_length_and_mask(eval_paired_sequence)
            eval_s1_span = batch['bert_s1_span']
            eval_s2_span = batch['bert_s2_span']

            if not feed_input_span:
                out = model(eval_paired_sequence, token_type_ids=eval_paired_segments_ids, attention_mask=eval_att_mask,
                            mode=BertMultiLayerSeqClassification.ForwardMode.EVAL,
                            labels=eval_labels_ids)
            else:
                out = model(eval_paired_sequence, token_type_ids=eval_paired_segments_ids, attention_mask=eval_att_mask,
                            s1_span=eval_s1_span, s2_span=eval_s2_span,
                            mode=BertPairMaxOutMatcher.ForwardMode.EVAL,
                            labels=eval_labels_ids)

            y_pid_list.extend(list(batch['oid']))
            y_fid_list.extend(list(batch['fid']))
            y_element_list.extend(list(batch['item']))

            y_pred_list.extend(torch.max(out, 1)[1].view(out.size(0)).tolist())
            y_logits_list.extend(out.tolist())

            if with_probs:
                y_probs_list.extend(F.softmax(out, dim=1).tolist())

            totoal_size += out.size(0)

    result_items_list = []
    assert len(y_pred_list) == len(y_fid_list)
    assert len(y_pred_list) == len(y_pid_list)
    assert len(y_pred_list) == len(y_element_list)

    assert len(y_pred_list) == len(y_logits_list)

    if with_probs:
        assert len(y_pred_list) == len(y_probs_list)

    for i in range(len(y_pred_list)):
        r_item = dict()
        r_item['fid'] = y_fid_list[i]
        r_item['oid'] = y_pid_list[i] if not make_int else int(y_pid_list[i])
        r_item['logits'] = y_logits_list[i]
        # r_item['probs'] =
        r_item['element'] = y_element_list[i]
        r_item['predicted_label'] = id2label[y_pred_list[i]]

        if with_probs:
            r_item['prob'] = y_probs_list[i]

        result_items_list.append(r_item)

    return result_items_list


def get_inference_pair(tag, is_training, sent_result_path, debug_num=None, evidence_filtering_threshold=0.01):
    # sent_result_path = ""

    if tag == 'dev':
        d_list = common.load_jsonl(config.FEVER_DEV)
    elif tag == 'train':
        d_list = common.load_jsonl(config.FEVER_TRAIN)
    elif tag == 'test':
        d_list = common.load_jsonl(config.FEVER_TEST)
    else:
        raise ValueError(f"Tag:{tag} not supported.")

    if debug_num is not None:
        # d_list = d_list[:10]
        d_list = d_list[:50]
        # d_list = d_list[:200]

    d_dict = list_dict_data_tool.list_to_dict(d_list, 'id')

    threshold_value = evidence_filtering_threshold
    # sent_list = common.load_jsonl(
    #     config.RESULT_PATH / "doc_retri_results/fever_results/sent_results/4-14-sent_results_v0/train_sent_results.jsonl")
    # sent_list = common.load_jsonl(
    #     config.RESULT_PATH / "doc_retri_results/fever_results/sent_results/4-14-sent_results_v0/i(5000)|e(0)|s01(0.9170917091709171)|s05(0.8842384238423843)|seed(12)_dev_sent_results.json")

    # debug_num = None if not debug else 2971
    # debug_num = None

    if isinstance(sent_result_path, Path):
        sent_list = common.load_jsonl(sent_result_path, debug_num)
    elif isinstance(sent_result_path, list):
        sent_list = sent_result_path
    else:
        raise ValueError(f"{sent_result_path} is not of a valid argument type which should be [list, Path].")

    list_dict_data_tool.append_subfield_from_list_to_dict(sent_list, d_dict,
                                                          'oid', 'fid', check=True)

    filltered_sent_dict = select_top_k_and_to_results_dict(d_dict,
                                                           top_k=5, threshold=threshold_value)

    list_dict_data_tool.append_item_from_dict_to_list(d_list, filltered_sent_dict,
                                                      'id', ['predicted_evidence', 'predicted_scored_evidence'])
    fever_db_cursor = fever_db.get_cursor(config.FEVER_DB)
    forward_items = build_nli_forward_item(d_list, is_training=is_training, db_cursor=fever_db_cursor)

    return forward_items, d_list


def model_go():
    seed = 12
    torch.manual_seed(seed)
    # bert_model_name = 'bert-large-uncased'
    bert_model_name = 'bert-base-uncased'
    experiment_name = 'fever_v1_nli_maxout'
    lazy = False
    # lazy = True
    forward_size = 16
    # batch_size = 64
    # batch_size = 192
    batch_size = 32
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_proportion = 0.1
    learning_rate = 5e-5
    num_train_epochs = 3
    eval_frequency = 2000
    do_lower_case = True
    pair_order = 'cq'
    # debug_mode = True
    debug_mode = False
    maxout_model = True
    # est_datasize = 900_000

    num_class = 3
    # num_train_optimization_steps

    train_sent_filtering_prob = 0.35
    dev_sent_filtering_prob = 0.01
    
    dev_sent_results_file = config.RESULT_PATH / "doc_retri_results/fever_results/sent_results/4-14-sent_results_v0/i(5000)|e(0)|s01(0.9170917091709171)|s05(0.8842384238423843)|seed(12)_dev_sent_results.json"
    train_sent_results_file = config.RESULT_PATH / "doc_retri_results/fever_results/sent_results/4-14-sent_results_v0/train_sent_results.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1

    n_gpu = torch.cuda.device_count()

    unk_token_num = {'tokens': 1}  # work around for initiating vocabulary.
    vocab = ExVocabulary(unk_token_num=unk_token_num)
    vocab.add_token_to_namespace('SUPPORTS', namespace='labels')
    vocab.add_token_to_namespace('REFUTES', namespace='labels')
    vocab.add_token_to_namespace('NOT ENOUGH INFO', namespace='labels')
    vocab.add_token_to_namespace("hidden", namespace="labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='labels')

    # Load Dataset
    # train_fitems_list = get_inference_pair('train', True, train_sent_results_file, debug_mode, train_sent_filtering_prob)
    dev_debug_num = 2481 if debug_mode else None
    dev_fitems_list, dev_list = get_inference_pair('dev', False, dev_sent_results_file, dev_debug_num,
                                                   dev_sent_filtering_prob)
    # = common.load_jsonl(config.FEVER_DEV)

    if debug_mode:
        dev_list = dev_list[:50]
        eval_frequency = 1
        # print(dev_list[-1]['_id'])
        # exit(0)

    # sampled_train_list = down_sample_neg(train_fitems_list, ratio=pos_ratio)
    train_debug_num = 2971 if debug_mode else None
    train_fitems_list, _ = get_inference_pair('train', True, train_sent_results_file, train_debug_num,
                                              train_sent_filtering_prob)
    est_datasize = len(train_fitems_list)

    # dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, 'id')
    # print(dev_o_dict)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)
    bert_cs_reader = BertFeverNLIReader(bert_tokenizer, lazy, is_paired=True, query_l=64,
                                        example_filter=None, max_l=364, pair_order=pair_order)

    bert_encoder = BertModel.from_pretrained(bert_model_name)
    if not maxout_model:
        model = BertMultiLayerSeqClassification(bert_encoder, num_labels=num_class, num_of_pooling_layer=1,
                                                act_type='tanh', use_pretrained_pooler=True, use_sigmoid=False)
    else:
        model = BertPairMaxOutMatcher(bert_encoder, num_of_class=num_class, act_type="gelu", num_of_out_layers=2)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(est_datasize / forward_size / gradient_accumulate_step) * \
                                   num_train_epochs

    if debug_mode:
        num_train_optimization_steps = 100

    print("Estimated training size", est_datasize)
    print("Number of optimization steps:", num_train_optimization_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=num_train_optimization_steps)

    dev_instances = bert_cs_reader.read(dev_fitems_list)

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    forbackward_step = 0
    update_step = 0

    logging_agent = save_tool.ScoreLogger({})

    file_path_prefix = '.'
    if not debug_mode:
        file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
        # # # Create Log File
        # Save the source code.
        script_name = os.path.basename(__file__)
        with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
            out_f.write(it.read())
            out_f.flush()
        # # # Log File end

    for epoch_i in range(num_train_epochs):
        print("Epoch:", epoch_i)

        train_fitems_list, _ = get_inference_pair('train', True, train_sent_results_file, train_debug_num,
                                                  train_sent_filtering_prob)
        random.shuffle(train_fitems_list)
        train_instance = bert_cs_reader.read(train_fitems_list)
        train_iter = biterator(train_instance, num_epochs=1, shuffle=True)

        for batch in tqdm(train_iter):
            model.train()
            batch = move_to_device(batch, device_num)

            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            labels_ids = batch['label']
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
            s1_span = batch['bert_s1_span']
            s2_span = batch['bert_s2_span']

            if not maxout_model:
                loss = model(paired_sequence, token_type_ids=paired_segments_ids, attention_mask=att_mask,
                             mode=BertMultiLayerSeqClassification.ForwardMode.TRAIN,
                             labels=labels_ids)
            else:
                loss = model(paired_sequence, token_type_ids=paired_segments_ids, attention_mask=att_mask,
                             s1_span=s1_span, s2_span=s2_span,
                             mode=BertPairMaxOutMatcher.ForwardMode.TRAIN,
                             labels=labels_ids)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if gradient_accumulate_step > 1:
                loss = loss / gradient_accumulate_step

            loss.backward()
            forbackward_step += 1

            if forbackward_step % gradient_accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                update_step += 1

                if update_step % eval_frequency == 0:
                    print("Update steps:", update_step)
                    dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)

                    cur_eval_results_list = eval_model(model, dev_iter, device_num, with_probs=True, make_int=True,
                                                       feed_input_span=maxout_model)

                    results_dict = list_dict_data_tool.list_to_dict(cur_eval_results_list, 'oid')
                    copied_dev_list = copy.deepcopy(dev_list)
                    list_dict_data_tool.append_item_from_dict_to_list(copied_dev_list, results_dict,
                                                                      'id', 'predicted_label')

                    mode = {'standard': True}
                    strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(copied_dev_list, dev_fitems_list,
                                                                                    mode=mode, max_evidence=5)
                    logging_item = {
                        'ss': strict_score, 'ac': acc_score,
                        'pr': pr, 'rec': rec, 'f1': f1,
                    }

                    save_file_name = f'i({update_step})|e({epoch_i})' \
                        f'|ss({strict_score})|ac({acc_score})|pr({pr})|rec({rec})|f1({f1})' \
                        f'|seed({seed})'

                    common.save_jsonl(copied_dev_list, Path(file_path_prefix) /
                                      f"{save_file_name}_dev_nli_results.json")

                    # print(save_file_name)
                    logging_agent.incorporate_results({}, save_file_name, logging_item)
                    logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = Path(file_path_prefix) / save_file_name
                    torch.save(model_to_save.state_dict(), str(output_model_file))

                    # print(logging_agent.logging_item_list)


def model_go_with_old_data():
    seed = 12
    torch.manual_seed(seed)
    # bert_model_name = 'bert-large-uncased'
    bert_model_name = 'bert-base-uncased'
    experiment_name = 'fever_v1_nli'
    lazy = False
    # lazy = True
    forward_size = 16
    # batch_size = 64
    # batch_size = 192
    batch_size = 32
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_proportion = 0.1
    learning_rate = 5e-5
    num_train_epochs = 3
    eval_frequency = 2000
    do_lower_case = True
    pair_order = 'cq'
    # debug_mode = True
    debug_mode = False
    # est_datasize = 900_000

    num_class = 3
    # num_train_optimization_steps

    train_sent_filtering_prob = 0.35
    dev_sent_filtering_prob = 0.1

    # dev_sent_results_file = config.RESULT_PATH / "doc_retri_results/fever_results/sent_results/4-14-sent_results_v0/i(5000)|e(0)|s01(0.9170917091709171)|s05(0.8842384238423843)|seed(12)_dev_sent_results.json"
    # train_sent_results_file = config.RESULT_PATH / "doc_retri_results/fever_results/sent_results/4-14-sent_results_v0/train_sent_results.jsonl"
    from utest.utest_format_converter_for_old_sent.tool import format_convert
    dev_sent_results_file = format_convert(config.PRO_ROOT / "results/doc_retri_results/fever_results/sent_results/old_sent_data_by_NSMN/4-15-dev_sent_pred_scores_old_format.jsonl")
    train_sent_results_file = format_convert(config.PRO_ROOT / "results/doc_retri_results/fever_results/sent_results/old_sent_data_by_NSMN/train_sent_scores_old_format.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1

    n_gpu = torch.cuda.device_count()

    unk_token_num = {'tokens': 1}  # work around for initiating vocabulary.
    vocab = ExVocabulary(unk_token_num=unk_token_num)
    vocab.add_token_to_namespace('SUPPORTS', namespace='labels')
    vocab.add_token_to_namespace('REFUTES', namespace='labels')
    vocab.add_token_to_namespace('NOT ENOUGH INFO', namespace='labels')
    vocab.add_token_to_namespace("hidden", namespace="labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='labels')

    # Load Dataset
    # train_fitems_list = get_inference_pair('train', True, train_sent_results_file, debug_mode, train_sent_filtering_prob)
    dev_debug_num = 2481 if debug_mode else None
    dev_fitems_list, dev_list = get_inference_pair('dev', False, dev_sent_results_file, dev_debug_num,
                                                   dev_sent_filtering_prob)
    # = common.load_jsonl(config.FEVER_DEV)

    if debug_mode:
        dev_list = dev_list[:50]
        eval_frequency = 1
        # print(dev_list[-1]['_id'])
        # exit(0)

    # sampled_train_list = down_sample_neg(train_fitems_list, ratio=pos_ratio)
    train_debug_num = 2971 if debug_mode else None
    train_fitems_list, _ = get_inference_pair('train', True, train_sent_results_file, train_debug_num,
                                              train_sent_filtering_prob)
    est_datasize = len(train_fitems_list)

    # dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, 'id')
    # print(dev_o_dict)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)
    bert_cs_reader = BertFeverNLIReader(bert_tokenizer, lazy, is_paired=True, query_l=64,
                                        example_filter=None, max_l=364, pair_order=pair_order)

    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertMultiLayerSeqClassification(bert_encoder, num_labels=num_class, num_of_pooling_layer=1,
                                            act_type='tanh', use_pretrained_pooler=True, use_sigmoid=False)
    #
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(est_datasize / forward_size / gradient_accumulate_step) * \
                                   num_train_epochs

    if debug_mode:
        num_train_optimization_steps = 100

    print("Estimated training size", est_datasize)
    print("Number of optimization steps:", num_train_optimization_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=num_train_optimization_steps)

    dev_instances = bert_cs_reader.read(dev_fitems_list)

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    forbackward_step = 0
    update_step = 0

    logging_agent = save_tool.ScoreLogger({})

    file_path_prefix = '.'
    if not debug_mode:
        file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
        # # # Create Log File
        # Save the source code.
        script_name = os.path.basename(__file__)
        with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
            out_f.write(it.read())
            out_f.flush()
        # # # Log File end

    for epoch_i in range(num_train_epochs):
        print("Epoch:", epoch_i)

        train_fitems_list, _ = get_inference_pair('train', True, train_sent_results_file, train_debug_num,
                                                  train_sent_filtering_prob)
        random.shuffle(train_fitems_list)
        train_instance = bert_cs_reader.read(train_fitems_list)
        train_iter = biterator(train_instance, num_epochs=1, shuffle=True)

        for batch in tqdm(train_iter):
            model.train()
            batch = move_to_device(batch, device_num)

            paired_sequence = batch['paired_sequence']
            paired_segments_ids = batch['paired_segments_ids']
            labels_ids = batch['label']
            att_mask, _ = torch_util.get_length_and_mask(paired_sequence)
            s1_span = batch['bert_s1_span']
            s2_span = batch['bert_s2_span']

            loss = model(paired_sequence, token_type_ids=paired_segments_ids, attention_mask=att_mask,
                         mode=BertMultiLayerSeqClassification.ForwardMode.TRAIN,
                         labels=labels_ids)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if gradient_accumulate_step > 1:
                loss = loss / gradient_accumulate_step

            loss.backward()
            forbackward_step += 1

            if forbackward_step % gradient_accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                update_step += 1

                if update_step % eval_frequency == 0:
                    print("Update steps:", update_step)
                    dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)

                    cur_eval_results_list = eval_model(model, dev_iter, device_num, with_probs=True, make_int=True)

                    results_dict = list_dict_data_tool.list_to_dict(cur_eval_results_list, 'oid')
                    copied_dev_list = copy.deepcopy(dev_list)
                    list_dict_data_tool.append_item_from_dict_to_list(copied_dev_list, results_dict,
                                                                      'id', 'predicted_label')

                    mode = {'standard': True}
                    strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(copied_dev_list, dev_fitems_list,
                                                                                    mode=mode, max_evidence=5)
                    logging_item = {
                        'ss': strict_score, 'ac': acc_score,
                        'pr': pr, 'rec': rec, 'f1': f1,
                    }

                    save_file_name = f'i({update_step})|e({epoch_i})' \
                        f'|ss({strict_score})|ac({acc_score})|pr({pr})|rec({rec})|f1({f1})' \
                        f'|seed({seed})'

                    common.save_jsonl(copied_dev_list, Path(file_path_prefix) /
                                      f"{save_file_name}_dev_nli_results.json")

                    # print(save_file_name)
                    logging_agent.incorporate_results({}, save_file_name, logging_item)
                    logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = Path(file_path_prefix) / save_file_name
                    torch.save(model_to_save.state_dict(), str(output_model_file))

                    # print(logging_agent.logging_item_list)


if __name__ == '__main__':
    # model_go_with_old_data()
    model_go()
