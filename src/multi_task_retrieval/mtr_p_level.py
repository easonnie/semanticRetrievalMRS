import copy
import os
import random
from pathlib import Path

import torch
from allennlp.data.iterators import BasicIterator
from allennlp.nn.util import move_to_device
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
from tqdm import tqdm

from data_utils.readers.bert_reader_content_selection import BertContentSelectionReader
import datetime

import config
from bert_model_variances.bert_multilayer_output import BertMultiLayerSeqClassification
from evaluation import ext_hotpot_eval, fever_scorer
from fever_sampler import fever_p_level_sampler
from fever_sampler import fever_sampler_utils
from flint import torch_util
from hotpot_data_analysis.fullwiki_provided_upperbound import append_gt_downstream_to_get_upperbound_from_doc_retri
from hotpot_fact_selection_sampler import sampler_utils as hotpot_sampler_utils
from hotpot_fact_selection_sampler.sampler_full_wiki import down_sample_neg
from neural_modules.model_EMA import get_ema_gpu_id_list, EMA

from utils import common, list_dict_data_tool, save_tool

from data_utils.exvocab import ExVocabulary


def select_top_k_and_to_results_dict(scored_dict, merged_field_name='merged_field',
                                     score_field_name='score', item_field_name='element',
                                     top_k=5):
    results_dict = {'sp_doc': dict(), 'scored_results': dict()}
    for key, value in scored_dict.items():
        fitems_dict = value[merged_field_name]
        scored_element_list = []
        for item in fitems_dict.values():
            score = item[score_field_name]
            element = item[item_field_name]
            scored_element_list.append((score, element))  # score is index 0.

        results_dict['scored_results'][key] = scored_element_list
        sorted_e_list = sorted(scored_element_list, key=lambda x: x[0], reverse=True)
        results_dict['sp_doc'][key] = [e for s, e in sorted_e_list[:top_k]]

    return results_dict


def eval_model(model, data_iter, device_num, with_probs=False, make_int=False, show_progress=False):
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
        r_item['qid'] = y_pid_list[i] if not make_int else int(y_pid_list[i])
        r_item['score'] = y_logits_list[i]
        r_item['element'] = y_element_list[i]

        if with_probs:
            r_item['prob'] = y_probs_list[i]

        result_items_list.append(r_item)

    return result_items_list


def eval_fever_procedure(biterator, dev_instances, model, device_num, ema_device_num,
                         dev_list, dev_o_dict, debug_mode, logging_agent, update_step, epoch_i, file_path_prefix,
                         do_ema, ema, seed):
    print("Eval FEVER!")
    dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)

    cur_eval_results_list = eval_model(model, dev_iter, device_num, make_int=True, with_probs=True)
    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    copied_dev_d_list = copy.deepcopy(dev_list)
    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)

    cur_results_dict_th0_5 = fever_sampler_utils.select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                                  score_field_name='prob',
                                                                                  top_k=5, filter_value=0.5)

    list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
                                                                   cur_results_dict_th0_5,
                                                                   'id', 'predicted_docids')
    # mode = {'standard': False, 'check_doc_id_correct': True}
    strict_score, pr, rec, f1 = fever_scorer.fever_doc_only(copied_dev_d_list, dev_list,
                                                            max_evidence=5)
    score_05 = {
        'ss': strict_score,
        'pr': pr, 'rec': rec, 'f1': f1,
    }

    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)

    cur_results_dict_th0_2 = fever_sampler_utils.select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                                  score_field_name='prob',
                                                                                  top_k=5, filter_value=0.2)

    list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
                                                                   cur_results_dict_th0_2,
                                                                   'id', 'predicted_docids')
    # mode = {'standard': False, 'check_doc_id_correct': True}
    strict_score, pr, rec, f1 = fever_scorer.fever_doc_only(copied_dev_d_list, dev_list,
                                                            max_evidence=5)
    score_02 = {
        'ss': strict_score,
        'pr': pr, 'rec': rec, 'f1': f1,
    }

    logging_item = {
        'step:': update_step,
        'epoch': epoch_i,
        'score_02': score_02,
        'score_05': score_05,
        'time': str(datetime.datetime.now())
    }

    print(logging_item)

    s02_ss_score = score_02['ss']
    s05_ss_score = score_05['ss']

    if not debug_mode:
        save_file_name = f'i({update_step})|e({epoch_i})' \
            f'|v02_ofever({s02_ss_score})' \
            f'|v05_ofever({s05_ss_score})|seed({seed})'

        # print(save_file_name)
        logging_agent.incorporate_results({}, save_file_name, logging_item)
        logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = Path(file_path_prefix) / save_file_name
        torch.save(model_to_save.state_dict(), str(output_model_file))

    if do_ema and ema is not None:
        ema_model = ema.get_inference_model()
        master_device_num = ema_device_num
        ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=master_device_num)
        ema_model = ema_model.to(master_device_num)
        ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)

        dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)
        cur_eval_results_list = eval_model(ema_model, dev_iter, master_device_num, make_int=True, with_probs=True)
        copied_dev_o_dict = copy.deepcopy(dev_o_dict)
        copied_dev_d_list = copy.deepcopy(dev_list)
        list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                              'qid', 'fid', check=True)

        cur_results_dict_th0_5 = fever_sampler_utils.select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                                      score_field_name='prob',
                                                                                      top_k=5, filter_value=0.5)

        list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
                                                                       cur_results_dict_th0_5,
                                                                       'id', 'predicted_docids')
        # mode = {'standard': False, 'check_doc_id_correct': True}
        strict_score, pr, rec, f1 = fever_scorer.fever_doc_only(copied_dev_d_list, dev_list,
                                                                max_evidence=5)
        score_05 = {
            'ss': strict_score,
            'pr': pr, 'rec': rec, 'f1': f1,
        }

        list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                              'qid', 'fid', check=True)

        cur_results_dict_th0_2 = fever_sampler_utils.select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                                      score_field_name='prob',
                                                                                      top_k=5, filter_value=0.2)

        list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
                                                                       cur_results_dict_th0_2,
                                                                       'id', 'predicted_docids')

        strict_score, pr, rec, f1 = fever_scorer.fever_doc_only(copied_dev_d_list, dev_list,
                                                                max_evidence=5)
        score_02 = {
            'ss': strict_score,
            'pr': pr, 'rec': rec, 'f1': f1,
        }

        logging_item = {
            'label': 'ema',
            'step:': update_step,
            'epoch': epoch_i,
            'score_02': score_02,
            'score_05': score_05,
            'time': str(datetime.datetime.now())
        }

        print(logging_item)

        s02_ss_score = score_02['ss']
        s05_ss_score = score_05['ss']

        if not debug_mode:
            save_file_name = f'i({update_step})|e({epoch_i})' \
                f'|v02_ofever({s02_ss_score})' \
                f'|v05_ofever({s05_ss_score})|seed({seed})'

            # print(save_file_name)
            logging_agent.incorporate_results({}, save_file_name, logging_item)
            logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

            model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
            output_model_file = Path(file_path_prefix) / save_file_name
            torch.save(model_to_save.state_dict(), str(output_model_file))


def eval_hotpot_procedure(biterator, dev_instances, model, device_num, ema_device_num,
                          dev_list, dev_o_dict, debug_mode, logging_agent, update_step, epoch_i, file_path_prefix,
                          do_ema, ema, seed):
    print("Eval HOTPOT!")
    dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)
    cur_eval_results_list = eval_model(model, dev_iter, device_num, with_probs=True)

    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)
    # Top_5
    cur_results_dict_top5 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5)
    upperbound_results_dict_top5 = append_gt_downstream_to_get_upperbound_from_doc_retri(
        cur_results_dict_top5,
        dev_list)

    cur_results_dict_top10 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=10)
    upperbound_results_dict_top10 = append_gt_downstream_to_get_upperbound_from_doc_retri(
        cur_results_dict_top10,
        dev_list)

    _, metrics_top5 = ext_hotpot_eval.eval(cur_results_dict_top5, dev_list, verbose=False)
    _, metrics_top5_UB = ext_hotpot_eval.eval(upperbound_results_dict_top5, dev_list, verbose=False)

    _, metrics_top10 = ext_hotpot_eval.eval(cur_results_dict_top10, dev_list, verbose=False)
    _, metrics_top10_UB = ext_hotpot_eval.eval(upperbound_results_dict_top10, dev_list, verbose=False)

    top5_doc_recall = metrics_top5['doc_recall']
    top5_UB_sp_recall = metrics_top5_UB['sp_recall']
    top10_doc_recall = metrics_top10['doc_recall']
    top10_Ub_sp_recall = metrics_top10_UB['sp_recall']

    logging_item = {
        'step:': update_step,
        'epoch': epoch_i,
        'top5': metrics_top5,
        'top5_UB': metrics_top5_UB,
        'top10': metrics_top10,
        'top10_UB': metrics_top10_UB,
        'time': str(datetime.datetime.now())
    }

    print(logging_item)
    if not debug_mode:
        save_file_name = f'i({update_step})|e({epoch_i})' \
            f'|t5_doc_recall({top5_doc_recall})|t5_sp_recall({top5_UB_sp_recall})' \
            f'|t10_doc_recall({top10_doc_recall})|t5_sp_recall({top10_Ub_sp_recall})|seed({seed})'

        # print(save_file_name)
        logging_agent.incorporate_results({}, save_file_name, logging_item)
        logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = Path(file_path_prefix) / save_file_name
        torch.save(model_to_save.state_dict(), str(output_model_file))

    if do_ema and ema is not None:
        ema_model = ema.get_inference_model()
        master_device_num = ema_device_num
        ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=master_device_num)
        ema_model = ema_model.to(master_device_num)
        ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)

        dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)

        cur_eval_results_list = eval_model(ema_model, dev_iter, master_device_num, with_probs=True)

        copied_dev_o_dict = copy.deepcopy(dev_o_dict)
        list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                              'qid', 'fid', check=True)
        # Top_5
        cur_results_dict_top5 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5)
        upperbound_results_dict_top5 = append_gt_downstream_to_get_upperbound_from_doc_retri(
            cur_results_dict_top5,
            dev_list)

        cur_results_dict_top10 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=10)
        upperbound_results_dict_top10 = append_gt_downstream_to_get_upperbound_from_doc_retri(
            cur_results_dict_top10,
            dev_list)

        _, metrics_top5 = ext_hotpot_eval.eval(cur_results_dict_top5, dev_list, verbose=False)
        _, metrics_top5_UB = ext_hotpot_eval.eval(upperbound_results_dict_top5, dev_list, verbose=False)

        _, metrics_top10 = ext_hotpot_eval.eval(cur_results_dict_top10, dev_list, verbose=False)
        _, metrics_top10_UB = ext_hotpot_eval.eval(upperbound_results_dict_top10, dev_list, verbose=False)

        top5_doc_recall = metrics_top5['doc_recall']
        top5_UB_sp_recall = metrics_top5_UB['sp_recall']
        top10_doc_recall = metrics_top10['doc_recall']
        top10_Ub_sp_recall = metrics_top10_UB['sp_recall']

        logging_item = {
            'label': 'ema',
            'step:': update_step,
            'epoch': epoch_i,
            'top5': metrics_top5,
            'top5_UB': metrics_top5_UB,
            'top10': metrics_top10,
            'top10_UB': metrics_top10_UB,
            'time': str(datetime.datetime.now())
        }

        print(logging_item)
        if not debug_mode:
            save_file_name = f'ema_i({update_step})|e({epoch_i})' \
                f'|t5_doc_recall({top5_doc_recall})|t5_sp_recall({top5_UB_sp_recall})' \
                f'|t10_doc_recall({top10_doc_recall})|t5_sp_recall({top10_Ub_sp_recall})|seed({seed})'

            # print(save_file_name)
            logging_agent.incorporate_results({}, save_file_name, logging_item)
            logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

            model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
            output_model_file = Path(file_path_prefix) / save_file_name
            torch.save(model_to_save.state_dict(), str(output_model_file))


def multitask_model_go():
    seed = 12
    torch.manual_seed(seed)
    # bert_model_name = 'bert-large-uncased'
    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    bert_model_name = 'bert-base-uncased'
    lazy = False
    # lazy = True
    forward_size = 64
    # batch_size = 64
    batch_size = 128
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_proportion = 0.1
    learning_rate = 5e-5
    num_train_epochs = 1
    eval_frequency = 5000
    hotpot_pos_ratio = 0.2
    do_lower_case = True
    max_l = 264

    experiment_name = f'mtr_p_level_(num_train_epochs:{num_train_epochs})'

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

    # Load Hotpot Dataset
    hotpot_train_list = common.load_json(config.TRAIN_FILE)
    hotpot_dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    hotpot_dev_o_dict = list_dict_data_tool.list_to_dict(hotpot_dev_list, '_id')

    # Load Hotpot upstream paragraph forward item
    hotpot_dev_fitems_list = common.load_jsonl(
        config.PDATA_ROOT / "content_selection_forward" / "hotpot_dev_p_level_unlabeled.jsonl")
    hotpot_train_fitems_list = common.load_jsonl(
        config.PDATA_ROOT / "content_selection_forward" / "hotpot_train_p_level_labeled.jsonl")

    hotpot_train_fitems_list = hotpot_sampler_utils.field_name_convert(hotpot_train_fitems_list, 'doc_t', 'element')
    hotpot_dev_fitems_list = hotpot_sampler_utils.field_name_convert(hotpot_dev_fitems_list, 'doc_t', 'element')

    # Load FEVER Dataset
    # fever_train_list = common.load_json(config.FEVER_TRAIN)
    fever_dev_list = common.load_jsonl(config.FEVER_DEV)
    fever_dev_o_dict = list_dict_data_tool.list_to_dict(fever_dev_list, 'id')

    train_ruleterm_doc_results = common.load_jsonl(
        config.PRO_ROOT / "results/doc_retri_results/fever_results/merged_doc_results/m_doc_train.jsonl")
    dev_ruleterm_doc_results = common.load_jsonl(
        config.PRO_ROOT / "results/doc_retri_results/fever_results/merged_doc_results/m_doc_dev.jsonl")

    fever_train_fitems_list = fever_p_level_sampler.get_paragraph_forward_pair('train', train_ruleterm_doc_results,
                                                                               is_training=True, debug=debug_mode,
                                                                               ignore_non_verifiable=True)
    fever_dev_fitems_list = fever_p_level_sampler.get_paragraph_forward_pair('dev', dev_ruleterm_doc_results,
                                                                             is_training=False, debug=debug_mode,
                                                                             ignore_non_verifiable=False)
    if debug_mode:
        hotpot_dev_list = hotpot_dev_list[:10]
        hotpot_dev_fitems_list = hotpot_dev_fitems_list[:296]
        hotpot_train_fitems_list = hotpot_train_fitems_list[:300]

        fever_dev_list = fever_dev_list[:100]
        eval_frequency = 2

    # Down_sample for hotpot.
    hotpot_sampled_train_list = down_sample_neg(hotpot_train_fitems_list, ratio=hotpot_pos_ratio)
    hotpot_est_datasize = len(hotpot_sampled_train_list)
    fever_est_datasize = len(fever_train_fitems_list)

    print("Hotpot Train Size:", hotpot_est_datasize)
    print("Fever Train Size:", fever_est_datasize)

    est_datasize = hotpot_est_datasize + fever_est_datasize

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                                   cache_dir=bert_pretrain_path)
    bert_cs_reader = BertContentSelectionReader(bert_tokenizer, lazy, is_paired=True,
                                                example_filter=lambda x: len(x['context']) == 0, max_l=max_l,
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

    hotpot_dev_instances = bert_cs_reader.read(hotpot_dev_fitems_list)
    fever_dev_instances = bert_cs_reader.read(fever_dev_fitems_list)

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    forbackward_step = 0
    update_step = 0

    logging_agent = save_tool.ScoreLogger({})

    file_path_prefix = '.'
    if not debug_mode:
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
        hotpot_sampled_train_list = down_sample_neg(hotpot_train_fitems_list, ratio=hotpot_pos_ratio)
        all_train_data = hotpot_sampled_train_list + fever_train_fitems_list
        random.shuffle(all_train_data)
        train_instance = bert_cs_reader.read(all_train_data)
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
                    # Eval FEVER
                    eval_fever_procedure(biterator, fever_dev_instances, model, device_num, 1, fever_dev_list,
                                         fever_dev_o_dict, debug_mode, logging_agent, update_step, epoch_i,
                                         file_path_prefix,
                                         do_ema, ema, seed)
                    eval_hotpot_procedure(biterator, hotpot_dev_instances, model, device_num, 1, hotpot_dev_list,
                                          hotpot_dev_o_dict, debug_mode, logging_agent, update_step, epoch_i,
                                          file_path_prefix, do_ema, ema, seed)

    if not debug_mode:
        print("Final Saving.")
        save_file_name = f'i({update_step})|e({num_train_epochs})_final_model'
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = Path(file_path_prefix) / save_file_name
        torch.save(model_to_save.state_dict(), str(output_model_file))

        if do_ema and ema is not None:
            print("Final EMA Saving")
            ema_model = ema.get_inference_model()
            save_file_name = f'i({update_step})|e({num_train_epochs})_final_ema_model'
            model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
            output_model_file = Path(file_path_prefix) / save_file_name
            torch.save(model_to_save.state_dict(), str(output_model_file))


if __name__ == '__main__':
    multitask_model_go()
