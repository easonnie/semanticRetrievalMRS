import os
import random
from pathlib import Path

from allennlp.data.iterators import BasicIterator
from allennlp.nn.util import move_to_device
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam

import config
from bert_model_variances.bert_multilayer_output import BertMultiLayerSeqClassification
from data_utils.exvocab import ExVocabulary
from data_utils.readers.bert_reader_content_selection import BertContentSelectionReader
from evaluation import ext_hotpot_eval
from flint import torch_util
from hotpot_data_analysis.fullwiki_provided_upperbound import append_gt_downstream_to_get_upperbound_from_doc_retri
from hotpot_fact_selection_sampler.sampler_full_wiki import down_sample_neg
from hotpot_fact_selection_sampler.sampler_utils import select_top_k_and_to_results_dict
from hotpot_fact_selection_sampler.sentence_level_sampler import get_sentence_pair
from neural_modules.model_EMA import EMA, get_ema_gpu_id_list
from utils import common, list_dict_data_tool
import torch
from tqdm import tqdm
import numpy as np
import copy
import allennlp
from utils import save_tool
import torch.nn.functional as F
from tqdm import tqdm
from hotpot_fact_selection_sampler import sentence_level_sampler


def eval_model(model, data_iter, device_num, with_probs=False, show_progress=False):
    print("Evaluating ...")
    with torch.no_grad():
        model.eval()
        totoal_size = 0

        y_pred_list = []
        y_fid_list = []
        y_pid_list = []
        y_element_list = []

        y_logits_list = []
        y_probs_list = []

        for batch_idx, batch in tqdm(enumerate(data_iter), disable=(not show_progress)):
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

            y_pid_list.extend(list(batch['qid']))
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
        r_item['qid'] = y_pid_list[i]
        r_item['score'] = y_logits_list[i]
        r_item['element'] = y_element_list[i]

        if with_probs:
            r_item['prob'] = y_probs_list[i]

        result_items_list.append(r_item)

    return result_items_list


# def select_top_k_and_to_results_dict(scored_dict, merged_field_name='merged_field',
#                                      score_field_name='score', item_field_name='element',
#                                      top_k=5):
#
#     results_dict = {'sp_doc': dict(), 'scored_results': dict()}
#     for key, value in scored_dict.items():
#         fitems_dict = value[merged_field_name]
#         scored_element_list = []
#         for item in fitems_dict.values():
#             score = item[score_field_name]
#             element = item[item_field_name]
#             scored_element_list.append((score, element))  # score is index 0.
#
#         results_dict['scored_results'][key] = scored_element_list
#         sorted_e_list = sorted(scored_element_list, key=lambda x: x[0], reverse=True)
#         results_dict['sp_doc'][key] = [e for s, e in sorted_e_list[:top_k]]
#
#     return results_dict


def model_go():
    seed = 12
    torch.manual_seed(seed)
    # bert_model_name = 'bert-large-uncased'
    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    bert_model_name = 'bert-base-uncased'
    lazy = False
    # lazy = True
    forward_size = 128
    # batch_size = 64
    batch_size = 128
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_proportion = 0.1
    learning_rate = 5e-5
    num_train_epochs = 5
    eval_frequency = 2000
    pos_ratio = 0.2
    do_lower_case = True
    document_top_k = 2
    experiment_name = f'hotpot_v0_slevel_retri_(doc_top_k:{document_top_k})'

    debug_mode = False
    do_ema = True
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
    train_list = common.load_json(config.TRAIN_FILE)
    dev_list = common.load_json(config.DEV_FULLWIKI_FILE)

    # train_fitems = sentence_level_sampler.get_train_sentence_pair(document_top_k, True, debug_mode)
    # dev_fitems = sentence_level_sampler.get_dev_sentence_pair(document_top_k, False, debug_mode)

    # Load train eval results list
    cur_train_eval_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/"
                          "i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/train_p_level_bert_v1_results.jsonl")

    cur_dev_eval_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/"
                          "i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/dev_p_level_bert_v1_results.jsonl")

    train_fitems = get_sentence_pair(document_top_k, train_list, cur_train_eval_results_list, is_training=True,
                                     debug_mode=debug_mode)

    dev_fitems = get_sentence_pair(document_top_k, dev_list, cur_dev_eval_results_list, is_training=False,
                                   debug_mode=debug_mode)

    if debug_mode:
        dev_list = dev_list[:100]
        eval_frequency = 2
        # print(dev_list[-1]['_id'])
        # exit(0)

    # sampled_train_list = down_sample_neg(train_fitems_list, ratio=pos_ratio)
    est_datasize = len(train_fitems)

    dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, '_id')
    # print(dev_o_dict)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                                   cache_dir=bert_pretrain_path)
    bert_cs_reader = BertContentSelectionReader(bert_tokenizer, lazy, is_paired=True,
                                                example_filter=lambda x: len(x['context']) == 0, max_l=128,
                                                element_fieldname='element')

    bert_encoder = BertModel.from_pretrained(bert_model_name, cache_dir=bert_pretrain_path)
    model = BertMultiLayerSeqClassification(bert_encoder, num_labels=num_class, num_of_pooling_layer=1,
                                            act_type='tanh', use_pretrained_pooler=True, use_sigmoid=True)

    ema = None
    if do_ema:
        ema = EMA(model, model.named_parameters(), device_num=1)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
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

    dev_instances = bert_cs_reader.read(dev_fitems)

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    forbackward_step = 0
    update_step = 0

    logging_agent = save_tool.ScoreLogger({})

    # # # Create Log File
    file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
    # Save the source code.
    script_name = os.path.basename(__file__)
    with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
        out_f.write(it.read())
        out_f.flush()
    # # # Log File end

    for epoch_i in range(num_train_epochs):
        print("Epoch:", epoch_i)
        # sampled_train_list = down_sample_neg(train_fitems_list, ratio=pos_ratio)
        random.shuffle(train_fitems)
        train_instance = bert_cs_reader.read(train_fitems)
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
                if ema is not None and do_ema:
                    updated_model = model.module if hasattr(model, 'module') else model
                    ema(updated_model.named_parameters())
                optimizer.zero_grad()
                update_step += 1

                if update_step % eval_frequency == 0:
                    print("Update steps:", update_step)
                    dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)

                    cur_eval_results_list = eval_model(model, dev_iter, device_num, with_probs=True)
                    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
                    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                                          'qid', 'fid', check=True)
                    # 0.5
                    cur_results_dict_v05 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
                                                                            score_field_name='prob',
                                                                            filter_value=0.5,
                                                                            result_field='sp')

                    cur_results_dict_v02 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
                                                                            score_field_name='prob',
                                                                            filter_value=0.2,
                                                                            result_field='sp')

                    _, metrics_v5 = ext_hotpot_eval.eval(cur_results_dict_v05, dev_list, verbose=False)

                    _, metrics_v2 = ext_hotpot_eval.eval(cur_results_dict_v02, dev_list, verbose=False)

                    v02_sp_f1 = metrics_v2['sp_f1']
                    v02_sp_recall = metrics_v2['sp_recall']
                    v02_sp_prec = metrics_v2['sp_prec']

                    v05_sp_f1 = metrics_v5['sp_f1']
                    v05_sp_recall = metrics_v5['sp_recall']
                    v05_sp_prec = metrics_v5['sp_prec']

                    logging_item = {
                        'v02': metrics_v2,
                        'v05': metrics_v5,
                    }

                    print(logging_item)

                    # print(logging_item)
                    if not debug_mode:
                        save_file_name = f'i({update_step})|e({epoch_i})' \
                            f'|v02_f1({v02_sp_f1})|v02_recall({v02_sp_recall})' \
                            f'|v05_f1({v05_sp_f1})|v05_recall({v05_sp_recall})|seed({seed})'

                        # print(save_file_name)
                        logging_agent.incorporate_results({}, save_file_name, logging_item)
                        logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = Path(file_path_prefix) / save_file_name
                        torch.save(model_to_save.state_dict(), str(output_model_file))

                    if do_ema and ema is not None:
                        ema_model = ema.get_inference_model()
                        master_device_num = 1
                        ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=master_device_num)
                        ema_model = ema_model.to(master_device_num)
                        ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)
                        dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)

                        cur_eval_results_list = eval_model(ema_model, dev_iter, master_device_num, with_probs=True)
                        copied_dev_o_dict = copy.deepcopy(dev_o_dict)
                        list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                                              'qid', 'fid', check=True)
                        # 0.5
                        cur_results_dict_v05 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
                                                                                score_field_name='prob',
                                                                                filter_value=0.5,
                                                                                result_field='sp')

                        cur_results_dict_v02 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
                                                                                score_field_name='prob',
                                                                                filter_value=0.2,
                                                                                result_field='sp')

                        _, metrics_v5 = ext_hotpot_eval.eval(cur_results_dict_v05, dev_list, verbose=False)

                        _, metrics_v2 = ext_hotpot_eval.eval(cur_results_dict_v02, dev_list, verbose=False)

                        v02_sp_f1 = metrics_v2['sp_f1']
                        v02_sp_recall = metrics_v2['sp_recall']
                        v02_sp_prec = metrics_v2['sp_prec']

                        v05_sp_f1 = metrics_v5['sp_f1']
                        v05_sp_recall = metrics_v5['sp_recall']
                        v05_sp_prec = metrics_v5['sp_prec']

                        logging_item = {
                            'label': 'ema',
                            'v02': metrics_v2,
                            'v05': metrics_v5,
                        }

                        print(logging_item)

                        if not debug_mode:
                            save_file_name = f'ema_i({update_step})|e({epoch_i})' \
                                f'|v02_f1({v02_sp_f1})|v02_recall({v02_sp_recall})' \
                                f'|v05_f1({v05_sp_f1})|v05_recall({v05_sp_recall})|seed({seed})'

                            # print(save_file_name)
                            logging_agent.incorporate_results({}, save_file_name, logging_item)
                            logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

                            model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
                            output_model_file = Path(file_path_prefix) / save_file_name
                            torch.save(model_to_save.state_dict(), str(output_model_file))


def eval_model_for_downstream(model_saved_path, doc_top_k=2, tag='dev'):
    seed = 12
    torch.manual_seed(seed)
    bert_model_name = 'bert-base-uncased'
    # lazy = False
    lazy = True
    # forward_size = 256
    forward_size = 128
    # batch_size = 64
    batch_size = 128
    do_lower_case = True
    document_top_k = doc_top_k

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
    train_list = common.load_json(config.TRAIN_FILE)
    dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    test_list = common.load_json(config.TEST_FULLWIKI_FILE)

    # Load train eval results list
    cur_train_eval_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/"
                          "i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/train_p_level_bert_v1_results.jsonl")

    cur_dev_eval_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/"
                          "i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/dev_p_level_bert_v1_results.jsonl")

    cur_test_eval_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/"
                          "i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/test_p_level_bert_v1_results.jsonl")

    if tag == 'train':
        train_fitems = get_sentence_pair(document_top_k, train_list, cur_train_eval_results_list, is_training=True,
                                         debug_mode=debug_mode)
    elif tag == 'dev':
        dev_fitems = get_sentence_pair(document_top_k, dev_list, cur_dev_eval_results_list, is_training=False,
                                       debug_mode=debug_mode)

    elif tag == 'test':
        test_fitems = get_sentence_pair(document_top_k, test_list, cur_test_eval_results_list, is_training=False,
                                        debug_mode=debug_mode)

    if debug_mode:
        eval_frequency = 2

    #     dev_list = dev_list[:10]
    #     dev_fitems_list = dev_fitems_list[:296]
    #     train_fitems_list = train_fitems_list[:300]
    # print(dev_list[-1]['_id'])
    # exit(0)

    dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, '_id')
    train_o_dict = list_dict_data_tool.list_to_dict(train_list, '_id')

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)
    bert_cs_reader = BertContentSelectionReader(bert_tokenizer, lazy, is_paired=True,
                                                example_filter=lambda x: len(x['context']) == 0, max_l=128,
                                                element_fieldname='element')

    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertMultiLayerSeqClassification(bert_encoder, num_labels=num_class, num_of_pooling_layer=1,
                                            act_type='tanh', use_pretrained_pooler=True, use_sigmoid=True)

    model.load_state_dict(torch.load(model_saved_path))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    #
    if tag == 'train':
        train_instance = bert_cs_reader.read(train_fitems)
    elif tag == 'dev':
        dev_instances = bert_cs_reader.read(dev_fitems)
    elif tag == 'test':
        test_instances = bert_cs_reader.read(test_fitems)

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    if tag == 'train':
        train_iter = biterator(train_instance, num_epochs=1, shuffle=False)
        print(len(train_fitems))
    elif tag == 'dev':
        dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)
        print(len(dev_fitems))
    elif tag == 'test':
        test_iter = biterator(test_instances, num_epochs=1, shuffle=False)
        print(len(test_fitems))

    print("Forward size:", forward_size)

    if tag == 'train':
        cur_train_eval_results_list_out = eval_model(model, train_iter, device_num, with_probs=True, show_progress=True)
        common.save_jsonl(cur_train_eval_results_list_out,
                          config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/train_s_level_bert_v1_results.jsonl")
    elif tag == 'dev':
        cur_dev_eval_results_list_out = eval_model(model, dev_iter, device_num, with_probs=True, show_progress=True)
        common.save_jsonl(cur_dev_eval_results_list_out,
                          config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/dev_s_level_bert_v1_results.jsonl")

    elif tag == 'test':
        cur_test_eval_results_list_out = eval_model(model, test_iter, device_num, with_probs=True, show_progress=True)
        common.save_jsonl(cur_test_eval_results_list_out,
                          config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/test_s_level_bert_v1_results.jsonl")

    if tag == 'train' or tag == 'test':
        exit(0)

    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    list_dict_data_tool.append_subfield_from_list_to_dict(cur_dev_eval_results_list, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)
    # 0.5
    cur_results_dict_v05 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
                                                            score_field_name='prob',
                                                            filter_value=0.5,
                                                            result_field='sp')

    cur_results_dict_v02 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
                                                            score_field_name='prob',
                                                            filter_value=0.2,
                                                            result_field='sp')

    _, metrics_v5 = ext_hotpot_eval.eval(cur_results_dict_v05, dev_list, verbose=False)

    _, metrics_v2 = ext_hotpot_eval.eval(cur_results_dict_v02, dev_list, verbose=False)

    logging_item = {
        'v02': metrics_v2,
        'v05': metrics_v5,
    }

    print(logging_item)


def eval_model_for_downstream_ablation(model_saved_path, doc_top_k=2, tag='dev'):
    print(f"Run doc_top_k:{doc_top_k}")
    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    seed = 12
    torch.manual_seed(seed)
    bert_model_name = 'bert-base-uncased'
    # lazy = False
    lazy = True
    # forward_size = 256
    forward_size = 256
    # batch_size = 64
    batch_size = 128
    do_lower_case = True
    document_top_k = doc_top_k

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
    train_list = common.load_json(config.TRAIN_FILE)
    dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    test_list = common.load_json(config.TEST_FULLWIKI_FILE)

    # Load train eval results list
    # cur_train_eval_results_list = common.load_jsonl(
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/"
    #                       "i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/train_p_level_bert_v1_results.jsonl")

    cur_dev_eval_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/"
                          "i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/dev_p_level_bert_v1_results.jsonl")

    # cur_test_eval_results_list = common.load_jsonl(
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/"
    #                       "i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/test_p_level_bert_v1_results.jsonl")

    # if tag == 'train':
    #     train_fitems = get_sentence_pair(document_top_k, train_list, cur_train_eval_results_list, is_training=True,
    #                                      debug_mode=debug_mode)
    if tag == 'dev':
        dev_fitems = get_sentence_pair(document_top_k, dev_list, cur_dev_eval_results_list, is_training=False,
                                       debug_mode=debug_mode)

    # elif tag == 'test':
    #     test_fitems = get_sentence_pair(document_top_k, test_list, cur_test_eval_results_list, is_training=False,
    #                                     debug_mode=debug_mode)

    if debug_mode:
        eval_frequency = 2

    #     dev_list = dev_list[:10]
    #     dev_fitems_list = dev_fitems_list[:296]
    #     train_fitems_list = train_fitems_list[:300]
    # print(dev_list[-1]['_id'])
    # exit(0)

    dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, '_id')
    train_o_dict = list_dict_data_tool.list_to_dict(train_list, '_id')

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                                   cache_dir=bert_pretrain_path)
    bert_cs_reader = BertContentSelectionReader(bert_tokenizer, lazy, is_paired=True,
                                                example_filter=lambda x: len(x['context']) == 0, max_l=128,
                                                element_fieldname='element')

    bert_encoder = BertModel.from_pretrained(bert_model_name, cache_dir=bert_pretrain_path)
    model = BertMultiLayerSeqClassification(bert_encoder, num_labels=num_class, num_of_pooling_layer=1,
                                            act_type='tanh', use_pretrained_pooler=True, use_sigmoid=True)

    model.load_state_dict(torch.load(model_saved_path))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    #
    if tag == 'train':
        train_instance = bert_cs_reader.read(train_fitems)
    elif tag == 'dev':
        dev_instances = bert_cs_reader.read(dev_fitems)
    elif tag == 'test':
        test_instances = bert_cs_reader.read(test_fitems)

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    if tag == 'train':
        train_iter = biterator(train_instance, num_epochs=1, shuffle=False)
        print(len(train_fitems))
    elif tag == 'dev':
        dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)
        print(len(dev_fitems))
    elif tag == 'test':
        test_iter = biterator(test_instances, num_epochs=1, shuffle=False)
        print(len(test_fitems))

    print("Forward size:", forward_size)

    if tag == 'train':
        cur_train_eval_results_list_out = eval_model(model, train_iter, device_num, with_probs=True,
                                                     show_progress=True)
        common.save_jsonl(cur_train_eval_results_list_out,
                          config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/train_s_level_bert_v1_results.jsonl")
    elif tag == 'dev':
        cur_dev_eval_results_list_out = eval_model(model, dev_iter, device_num, with_probs=True, show_progress=True)
        common.save_jsonl(cur_dev_eval_results_list_out, f"hotpot_s_level_{tag}_results_top_k_doc_{document_top_k}.jsonl")

    elif tag == 'test':
        cur_test_eval_results_list_out = eval_model(model, test_iter, device_num, with_probs=True,
                                                    show_progress=True)
        common.save_jsonl(cur_test_eval_results_list_out,
                          config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/test_s_level_bert_v1_results.jsonl")

    if tag == 'train' or tag == 'test':
        exit(0)

    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    list_dict_data_tool.append_subfield_from_list_to_dict(cur_dev_eval_results_list_out, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)
    # 0.5
    cur_results_dict_v05 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
                                                            score_field_name='prob',
                                                            filter_value=0.5,
                                                            result_field='sp')

    cur_results_dict_v02 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
                                                            score_field_name='prob',
                                                            filter_value=0.2,
                                                            result_field='sp')

    _, metrics_v5 = ext_hotpot_eval.eval(cur_results_dict_v05, dev_list, verbose=False)

    _, metrics_v2 = ext_hotpot_eval.eval(cur_results_dict_v02, dev_list, verbose=False)

    logging_item = {
        'v02': metrics_v2,
        'v05': metrics_v5,
    }

    print(logging_item)
    f1 = metrics_v5['sp_f1']
    em = metrics_v5['sp_em']
    pr = metrics_v5['sp_prec']
    rec = metrics_v5['sp_recall']
    common.save_json(logging_item, f"top_k_doc:{document_top_k}_em:{em}_pr:{pr}_rec:{rec}_f1:{f1}")

    # common.save_jsonl(cur_train_eval_results_list, "train_p_level_bert_v1_results.jsonl")


if __name__ == '__main__':
    # model_go()
    model_saved_path = config.PRO_ROOT / "saved_models/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)"
    # eval_model_for_downstream(model_saved_path, tag='train')
    for doc_top_k in [1, 3, 5, 7, 9, 10, 11, 12]:
        eval_model_for_downstream_ablation(model_saved_path, doc_top_k, tag='dev')
    # eval_model_for_downstream_ablation(model_saved_path, 100, tag='dev')
    # eval_model_for_downstream(model_saved_path, tag='test')
