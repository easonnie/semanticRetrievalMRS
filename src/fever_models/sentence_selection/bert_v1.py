import os
from pathlib import Path

from allennlp.data.iterators import BasicIterator
from allennlp.nn.util import move_to_device
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam

import config
from bert_model_variances.bert_multilayer_output import BertMultiLayerSeqClassification
from data_utils.exvocab import ExVocabulary
from data_utils.readers.bert_reader_fever_sent_selection import BertContentSelectionReader
# from evaluation import ext_hotpot_eval
from evaluation import fever_scorer
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


def eval_model(model, data_iter, device_num, with_probs=False, make_int=False, show_progress=False):
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
            s1_span = batch['bert_s1_span']
            s2_span = batch['bert_s2_span']

            out = model(eval_paired_sequence, token_type_ids=eval_paired_segments_ids, attention_mask=eval_att_mask,
                        mode=BertMultiLayerSeqClassification.ForwardMode.EVAL,
                        labels=eval_labels_ids)

            y_pid_list.extend(list(batch['oid']))
            y_fid_list.extend(list(batch['fid']))
            y_element_list.extend(list(batch['item']))

            y_pred_list.extend(torch.max(out, 1)[1].view(out.size(0)).tolist())

            y_logits_list.extend(out.view(out.size(0)).tolist())

            if with_probs:
                y_probs_list.extend(torch.sigmoid(out).view(out.size(0)).tolist())

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
        r_item['score'] = y_logits_list[i]
        r_item['element'] = y_element_list[i]

        if with_probs:
            r_item['prob'] = y_probs_list[i]

        result_items_list.append(r_item)

    return result_items_list


def select_top_k_and_to_results_dict(scored_dict, merged_field_name='merged_field',
                                     score_field_name='prob', item_field_name='element',
                                     top_k=5, threshold=None):
    results_dict = dict()

    for key, value in scored_dict.items():
        if key not in results_dict:
            results_dict[key] = dict()

        # if merged_field_name not in value:
        #     results_dict[key]['scored_results'] = []
        #     results_dict[key]['predicated_evidence'] = []
        #     continue

        fitems_dict = value[merged_field_name]
        scored_element_list = []
        for item in fitems_dict.values():
            score = item[score_field_name]
            element = item[item_field_name]
            scored_element_list.append((score, element))  # score is index 0.

        results_dict[key]['scored_results'] = scored_element_list
        sorted_e_list = sorted(scored_element_list, key=lambda x: x[0], reverse=True)
        evidence_sid = []
        scored_evidence_sid = []
        for s, e in sorted_e_list:
            if threshold is not None:
                if s >= threshold:
                    evidence_sid.append(e)
                    scored_evidence_sid.append([s, e])
            else:
                evidence_sid.append(e)
                scored_evidence_sid.append([s, e])
        evidence_sid = evidence_sid[:top_k]
        scored_evidence_sid = scored_evidence_sid[:top_k]

        assert len(evidence_sid) == len(scored_evidence_sid)

        results_dict[key]['predicted_evidence'] = []
        for sid in evidence_sid:
            doc_id, ln = sid.split('(-.-)')[0], int(sid.split('(-.-)')[1])
            results_dict[key]['predicted_evidence'].append([doc_id, ln])

        results_dict[key]['predicted_scored_evidence'] = []
        for score, sid in scored_evidence_sid:
            doc_id, ln = sid.split('(-.-)')[0], int(sid.split('(-.-)')[1])
            results_dict[key]['predicted_scored_evidence'].append((score, [doc_id, ln]))

        # predicted_sentids
        # results_dict[key]['predicted_sentids'] = results_dict[key]['predicted_evidence']

    return results_dict


def get_sentences(tag, is_training, debug=False):
    if tag == 'dev':
        d_list = common.load_jsonl(config.FEVER_DEV)
    elif tag == 'train':
        d_list = common.load_jsonl(config.FEVER_TRAIN)
    elif tag == 'test':
        d_list = common.load_jsonl(config.FEVER_TEST)
    else:
        raise ValueError(f"Tag:{tag} not supported.")

    if debug:
        # d_list = d_list[:10]
        d_list = d_list[:50]
        # d_list = d_list[:200]

    doc_results = common.load_jsonl(
        config.RESULT_PATH / f"doc_retri_results/fever_results/merged_doc_results/m_doc_{tag}.jsonl")
    doc_results_dict = list_dict_data_tool.list_to_dict(doc_results, 'id')
    fever_db_cursor = fever_db.get_cursor(config.FEVER_DB)
    forward_items = build_full_wiki_document_forward_item(doc_results_dict, d_list, is_training=is_training,
                                                          db_cursor=fever_db_cursor)
    return forward_items


def set_gt_nli_label(d_list, delete_label=False):
    for item in d_list:
        item['predicted_label'] = item['label']
        if delete_label:
            del item['label']
    return d_list


def model_go():
    seed = 12
    torch.manual_seed(seed)
    # bert_model_name = 'bert-large-uncased'
    bert_model_name = 'bert-base-uncased'
    experiment_name = 'fever_v0_cs_ratio_001'
    # lazy = False
    lazy = True
    forward_size = 128
    # batch_size = 64
    # batch_size = 192
    batch_size = 128
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_proportion = 0.1
    learning_rate = 5e-5
    num_train_epochs = 5
    eval_frequency = 20000
    pos_ratio = 0.01
    do_lower_case = True

    # debug_mode = True
    debug_mode = False
    # est_datasize = 900_000

    num_class = 1
    # num_train_optimization_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1

    n_gpu = torch.cuda.device_count()

    unk_token_num = {'tokens': 1}  # work around for initiating vocabulary.
    vocab = ExVocabulary(unk_token_num=unk_token_num)
    vocab.add_token_to_namespace("false", namespace="labels")  # 0
    vocab.add_token_to_namespace("true", namespace="labels")  # 1
    vocab.add_token_to_namespace("hidden", namespace="labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='labels')

    # Load Dataset
    # train_list = common.load_jsonl(config.FEVER_TRAIN)
    dev_list = common.load_jsonl(config.FEVER_DEV)
    set_gt_nli_label(dev_list)

    # dev_fitems_list = common.load_jsonl(
    #     config.PDATA_ROOT / "content_selection_forward" / "hotpot_dev_p_level_unlabeled.jsonl")
    # train_fitems_list = common.load_jsonl(
    #     config.PDATA_ROOT / "content_selection_forward" / "hotpot_train_p_level_labeled.jsonl")

    dev_fitems_list = get_sentences('dev', is_training=False, debug=debug_mode)
    train_fitems_list = get_sentences('train', is_training=True, debug=debug_mode)

    if debug_mode:
        dev_list = dev_list[:50]
        eval_frequency = 1
        # print(dev_list[-1]['_id'])
        # exit(0)

    sampled_train_list = down_sample_neg(train_fitems_list, ratio=pos_ratio)
    est_datasize = len(sampled_train_list)

    dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, 'id')
    # print(dev_o_dict)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)
    bert_cs_reader = BertContentSelectionReader(bert_tokenizer, lazy, is_paired=True,
                                                example_filter=lambda x: len(x['context']) == 0, max_l=128)

    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertMultiLayerSeqClassification(bert_encoder, num_labels=num_class, num_of_pooling_layer=1,
                                            act_type='tanh', use_pretrained_pooler=True, use_sigmoid=True)
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
        sampled_train_list = down_sample_neg(train_fitems_list, ratio=pos_ratio)
        train_instance = bert_cs_reader.read(sampled_train_list)
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
                    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
                    copied_dev_d_list = copy.deepcopy(dev_list)
                    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                                          'oid', 'fid', check=True)

                    print("Threshold 0.5:")
                    cur_results_dict_th0_5 = select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                              top_k=5, threshold=0.5)
                    list_dict_data_tool.append_item_from_dict_to_list(copied_dev_d_list, cur_results_dict_th0_5,
                                                                      'id', 'predicted_evidence')
                    mode = {'standard': True, 'check_sent_id_correct': True}
                    strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(copied_dev_d_list, dev_list,
                                                                                    mode=mode, max_evidence=5)
                    score_05 = {
                        'ss': strict_score, 'as': acc_score,
                        'pr': pr, 'rec': rec, 'f1': f1,
                    }

                    print("Threshold 0.1:")
                    cur_results_dict_th0_1 = select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                              top_k=5, threshold=0.1)
                    list_dict_data_tool.append_item_from_dict_to_list(copied_dev_d_list, cur_results_dict_th0_1,
                                                                      'id', 'predicted_evidence')
                    mode = {'standard': True, 'check_sent_id_correct': True}
                    strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(copied_dev_d_list, dev_list,
                                                                                    mode=mode, max_evidence=5)
                    score_01 = {
                        'ss': strict_score, 'as': acc_score,
                        'pr': pr, 'rec': rec, 'f1': f1,
                    }

                    logging_item = {
                        'score_01': score_01,
                        'score_05': score_05,
                    }

                    print(logging_item)

                    s01_ss_score = score_01['ss']
                    s05_ss_score = score_05['ss']
                    #
                    # exit(0)

                    # print(logging_item)
                    save_file_name = f'i({update_step})|e({epoch_i})' \
                        f'|s01({s01_ss_score})|s05({s05_ss_score})' \
                        f'|seed({seed})'

                    common.save_jsonl(cur_eval_results_list, Path(file_path_prefix) /
                                      f"{save_file_name}_dev_sent_results.json")

                    # print(save_file_name)
                    logging_agent.incorporate_results({}, save_file_name, logging_item)
                    logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = Path(file_path_prefix) / save_file_name
                    torch.save(model_to_save.state_dict(), str(output_model_file))

                    # print(logging_agent.logging_item_list)

        # Epoch eval:
        print("Update steps:", update_step)
        dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)

        cur_eval_results_list = eval_model(model, dev_iter, device_num, with_probs=True, make_int=True)
        copied_dev_o_dict = copy.deepcopy(dev_o_dict)
        copied_dev_d_list = copy.deepcopy(dev_list)
        list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                              'oid', 'fid', check=True)

        print("Threshold 0.5:")
        cur_results_dict_th0_5 = select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                  top_k=5, threshold=0.5)
        list_dict_data_tool.append_item_from_dict_to_list(copied_dev_d_list, cur_results_dict_th0_5,
                                                          'id', 'predicted_evidence')
        mode = {'standard': True, 'check_sent_id_correct': True}
        strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(copied_dev_d_list, dev_list,
                                                                        mode=mode, max_evidence=5)
        score_05 = {
            'ss': strict_score, 'as': acc_score,
            'pr': pr, 'rec': rec, 'f1': f1,
        }

        print("Threshold 0.1:")
        cur_results_dict_th0_1 = select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                  top_k=5, threshold=0.1)
        list_dict_data_tool.append_item_from_dict_to_list(copied_dev_d_list, cur_results_dict_th0_1,
                                                          'id', 'predicted_evidence')
        mode = {'standard': True, 'check_sent_id_correct': True}
        strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(copied_dev_d_list, dev_list,
                                                                        mode=mode, max_evidence=5)
        score_01 = {
            'ss': strict_score, 'as': acc_score,
            'pr': pr, 'rec': rec, 'f1': f1,
        }

        logging_item = {
            'score_01': score_01,
            'score_05': score_05,
        }

        print(logging_item)

        s01_ss_score = score_01['ss']
        s05_ss_score = score_05['ss']
        #
        # exit(0)

        # print(logging_item)
        save_file_name = f'i({update_step})|e({epoch_i})' \
            f'|s01({s01_ss_score})|s05({s05_ss_score})' \
            f'|seed({seed})'

        common.save_jsonl(cur_eval_results_list, Path(file_path_prefix) /
                          f"{save_file_name}_dev_sent_results.jsonl")

        # print(save_file_name)
        logging_agent.incorporate_results({}, save_file_name, logging_item)
        logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = Path(file_path_prefix) / save_file_name
        torch.save(model_to_save.state_dict(), str(output_model_file))


def eval_trainset_for_train_nli(model_path):
    tag = 'test'
    is_training = False

    seed = 12
    torch.manual_seed(seed)
    bert_model_name = 'bert-base-uncased'
    lazy = False
    # lazy = True
    forward_size = 128
    # batch_size = 64
    # batch_size = 192
    batch_size = 128

    do_lower_case = True

    debug_mode = False
    # debug_mode = True

    num_class = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1

    n_gpu = torch.cuda.device_count()

    unk_token_num = {'tokens': 1}  # work around for initiating vocabulary.
    vocab = ExVocabulary(unk_token_num=unk_token_num)
    vocab.add_token_to_namespace("false", namespace="labels")  # 0
    vocab.add_token_to_namespace("true", namespace="labels")  # 1
    vocab.add_token_to_namespace("hidden", namespace="labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='labels')

    # Load Dataset

    train_fitems_list = get_sentences(tag, is_training=is_training, debug=debug_mode)
    est_datasize = len(train_fitems_list)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)
    bert_cs_reader = BertContentSelectionReader(bert_tokenizer, lazy, is_paired=True,
                                                example_filter=lambda x: len(x['context']) == 0, max_l=128)

    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertMultiLayerSeqClassification(bert_encoder, num_labels=num_class, num_of_pooling_layer=1,
                                            act_type='tanh', use_pretrained_pooler=True, use_sigmoid=True)

    model.load_state_dict(torch.load(model_path))

    print("Estimated training size", est_datasize)
    print("Estimated forward steps:", est_datasize / forward_size)

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    train_instance = bert_cs_reader.read(train_fitems_list)
    train_iter = biterator(train_instance, num_epochs=1, shuffle=False)

    cur_eval_results_list = eval_model(model, train_iter, device_num, with_probs=True, make_int=True, show_progress=True)

    if debug_mode:
        train_list = common.load_jsonl(config.FEVER_TRAIN)
        train_list = train_list[:50]
        set_gt_nli_label(train_list)
        train_o_dict = list_dict_data_tool.list_to_dict(train_list, 'id')

        copied_dev_o_dict = copy.deepcopy(train_o_dict)
        copied_dev_d_list = copy.deepcopy(train_list)
        list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                              'oid', 'fid', check=True)

        print("Threshold 0.5:")
        cur_results_dict_th0_5 = select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                  top_k=5, threshold=0.1)
        list_dict_data_tool.append_item_from_dict_to_list(copied_dev_d_list, cur_results_dict_th0_5,
                                                          'id', 'predicted_evidence')
        mode = {'standard': True, 'check_sent_id_correct': True}
        strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(copied_dev_d_list, train_list,
                                                                        mode=mode, max_evidence=5)
        print(strict_score, acc_score, pr, rec, f1)

    common.save_jsonl(cur_eval_results_list, f'{tag}_sent_results_labeled:{is_training}.jsonl')


if __name__ == '__main__':
    model_go()
    # model_path = config.PRO_ROOT / "saved_models/04-13-16:37:29_fever_v0_cs/i(5000)|e(0)|s01(0.9170917091709171)|s05(0.8842384238423843)|seed(12)"
    #
    # model_path = config.PRO_ROOT / "saved_models/04-13-16:37:29_fever_v0_cs/i(15000)|e(1)|s01(0.9013901390139014)|s05(0.8517851785178517)|seed(12)"
    # eval_trainset_for_train_nli(model_path)

    # dev_sent_list = get_sentences('dev', is_training=False)
    # print(len(dev_sent_list))
    #
    # train_sent_list = get_sentences('dev', is_training=True)
    # sampled_sent_list = down_sample_neg(train_sent_list, ratio=0.2)
    # print(len(sampled_sent_list))
