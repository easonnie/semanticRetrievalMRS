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
from open_domain_sampler import p_sampler as open_domain_p_sampler
from open_domain_sampler import od_sample_utils
from evaluation import open_domain_qa_eval
from span_prediction_task_utils.squad_utils import get_squad_question_selection_forward_list

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


def eval_open_qa_procedure(biterator, dev_instances, model, device_num, ema_device_num,
                           dev_list, dev_o_dict, debug_mode, logging_agent, update_step, epoch_i, file_path_prefix,
                           do_ema, ema, seed, dataset_name):
    print(f"Eval {dataset_name}!")
    # dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)
    #
    # cur_eval_results_list = eval_model(model, dev_iter, device_num, make_int=True, with_probs=True)
    # copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    # copied_dev_d_list = copy.deepcopy(dev_list)
    # list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
    #                                                       'qid', 'fid', check=True)
    #
    # cur_results_dict_th0_5 = od_sample_utils.select_top_k_and_to_results_dict(copied_dev_o_dict,
    #                                                                               score_field_name='prob',
    #                                                                               top_k=5, filter_value=0.5)
    #
    # list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
    #                                                                cur_results_dict_th0_5,
    #                                                                'id', 'predicted_docids')
    # # mode = {'standard': False, 'check_doc_id_correct': True}
    # strict_score, pr, rec, f1 = fever_scorer.fever_doc_only(copied_dev_d_list, dev_list,
    #                                                         max_evidence=5)
    # score_05 = {
    #     'ss': strict_score,
    #     'pr': pr, 'rec': rec, 'f1': f1,
    # }
    #
    # list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
    #                                                       'qid', 'fid', check=True)
    #
    # cur_results_dict_th0_2 = fever_sampler_utils.select_top_k_and_to_results_dict(copied_dev_o_dict,
    #                                                                               score_field_name='prob',
    #                                                                               top_k=5, filter_value=0.2)
    #
    # list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
    #                                                                cur_results_dict_th0_2,
    #                                                                'id', 'predicted_docids')
    # # mode = {'standard': False, 'check_doc_id_correct': True}
    # strict_score, pr, rec, f1 = fever_scorer.fever_doc_only(copied_dev_d_list, dev_list,
    #                                                         max_evidence=5)
    # score_02 = {
    #     'ss': strict_score,
    #     'pr': pr, 'rec': rec, 'f1': f1,
    # }
    #
    # logging_item = {
    #     'step:': update_step,
    #     'epoch': epoch_i,
    #     'score_02': score_02,
    #     'score_05': score_05,
    #     'time': str(datetime.datetime.now())
    # }
    #
    # print(logging_item)
    #
    # s02_ss_score = score_02['ss']
    # s05_ss_score = score_05['ss']
    #
    # if not debug_mode:
    #     save_file_name = f'i({update_step})|e({epoch_i})' \
    #         f'|v02_ofever({s02_ss_score})' \
    #         f'|v05_ofever({s05_ss_score})|seed({seed})'
    #
    #     # print(save_file_name)
    #     logging_agent.incorporate_results({}, save_file_name, logging_item)
    #     logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")
    #
    #     model_to_save = model.module if hasattr(model, 'module') else model
    #     output_model_file = Path(file_path_prefix) / save_file_name
    #     torch.save(model_to_save.state_dict(), str(output_model_file))

    if do_ema and ema is not None:
        ema_model = ema.get_inference_model()
        master_device_num = ema_device_num
        ema_inference_device_ids = get_ema_gpu_id_list(master_device_num=master_device_num)
        ema_model = ema_model.to(master_device_num)
        ema_model = torch.nn.DataParallel(ema_model, device_ids=ema_inference_device_ids)

        dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)
        cur_eval_results_list = eval_model(ema_model, dev_iter, master_device_num, make_int=False, with_probs=True)

        copied_dev_o_dict = copy.deepcopy(dev_o_dict)
        copied_dev_d_list = copy.deepcopy(dev_list)
        list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                              'qid', 'fid', check=True)

        cur_results_dict_top10 = od_sample_utils.select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                                  score_field_name='prob',
                                                                                  top_k=10, filter_value=0.01)

        list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
                                                                       cur_results_dict_top10,
                                                                       'qid', 'pred_p_list')

        t10_recall = open_domain_qa_eval.qa_paragraph_eval_v1(copied_dev_d_list, dev_list)

        top_10_recall = {
            'recall': t10_recall,
        }

        list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                              'qid', 'fid', check=True)

        cur_results_dict_top20 = od_sample_utils.select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                                  score_field_name='prob',
                                                                                  top_k=20, filter_value=0.01)

        list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
                                                                       cur_results_dict_top20,
                                                                       'qid', 'pred_p_list')

        t20_recall = open_domain_qa_eval.qa_paragraph_eval_v1(copied_dev_d_list, dev_list)

        top_20_recall = {
            'top_20_recall': t20_recall,
        }

        logging_item = {
            'label': 'ema',
            'step:': update_step,
            'epoch': epoch_i,
            'dataset_name': dataset_name,
            'top10': top_10_recall,
            'top20': top_20_recall,
            'time': str(datetime.datetime.now())
        }

        print(logging_item)

        common.save_jsonl(cur_eval_results_list, Path(file_path_prefix) / Path(
            f"i({update_step})|e({epoch_i})|{dataset_name}|top10({t10_recall})|top20({t20_recall})|seed({seed})_eval_results.jsonl"))

        if not debug_mode:
            save_file_name = f'i({update_step})|e({epoch_i})|{dataset_name}' \
                f'|top10({t10_recall})' \
                f'|top20({t20_recall})|seed({seed})'

            # print(save_file_name)
            logging_agent.incorporate_results({}, save_file_name, logging_item)
            logging_agent.logging_to_file(Path(file_path_prefix) / "log.json")

            model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
            output_model_file = Path(file_path_prefix) / save_file_name
            torch.save(model_to_save.state_dict(), str(output_model_file))


def separate_eval_open_qa_procedure(biterator, dev_instances, model, device_num,
                                    dev_list, dev_o_dict, dataset_name, save_path=None, tag=""):
    print(f"Eval {dataset_name}!")

    dev_iter = biterator(dev_instances, num_epochs=1, shuffle=False)
    cur_eval_results_list = eval_model(model, dev_iter, device_num, make_int=False, with_probs=True, show_progress=True)

    if save_path is not None:
        model_file = str(Path(model_path).stem)
        save_filename = Path(save_path) / f"{model_file}_{dataset_name}_{tag}_p_level_eval.jsonl"
        common.save_jsonl(cur_eval_results_list, Path(save_filename))
    else:
        pass

    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    copied_dev_d_list = copy.deepcopy(dev_list)
    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)

    cur_results_dict_top10 = od_sample_utils.select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                              score_field_name='prob',
                                                                              top_k=10, filter_value=0.01)

    list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
                                                                   cur_results_dict_top10,
                                                                   'qid', 'pred_p_list')

    t10_recall = open_domain_qa_eval.qa_paragraph_eval_v1(copied_dev_d_list, dev_list)

    top_10_recall = {
        'recall': t10_recall,
    }

    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)

    cur_results_dict_top20 = od_sample_utils.select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                                              score_field_name='prob',
                                                                              top_k=20, filter_value=0.01)

    list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
                                                                   cur_results_dict_top20,
                                                                   'qid', 'pred_p_list')

    t20_recall = open_domain_qa_eval.qa_paragraph_eval_v1(copied_dev_d_list, dev_list)

    top_20_recall = {
        'top_20_recall': t20_recall,
    }

    logging_item = {
        'label': 'ema',
        'dataset_name': dataset_name,
        'top10': top_10_recall,
        'top20': top_20_recall,
        'time': str(datetime.datetime.now())
    }

    print(logging_item)


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


def multitask_open_qa_model_go():
    seed = 12
    torch.manual_seed(seed)
    # bert_model_name = 'bert-large-uncased'
    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    bert_model_name = 'bert-base-uncased'
    lazy = True
    # lazy = True
    forward_size = 64
    # batch_size = 64
    batch_size = 128
    gradient_accumulate_step = int(batch_size / forward_size)
    warmup_proportion = 0.1
    learning_rate = 3e-5
    num_train_epochs = 3
    eval_frequency = 10000
    hotpot_pos_ratio = 0.25
    do_lower_case = True
    max_l = 254

    hotpot_train_size = None
    fever_train_size = None
    curatedtrec_train_size = 0
    webq_train_size = 0
    squad_train_size = 0
    wikimovie_train_size = 0

    squad_v11_pos_size = None

    # hotpot_train_size = 0
    # fever_train_size = 0
    # squad_train_size = 80_000
    # squad_v11_pos_size = 0

    experiment_name = f'mtr_open_qa_p_level_(num_train_epochs:{num_train_epochs})'

    debug_mode = False
    do_ema = True

    open_qa_paras = {
        'webq': {'upstream_top_k': 40, 'distant_gt_top_k': 2, 'down_sample_ratio': 0.25},
        'curatedtrec': {'upstream_top_k': 40, 'distant_gt_top_k': 2, 'down_sample_ratio': 0.25},
        'squad': {'upstream_top_k': 40, 'distant_gt_top_k': 1, 'down_sample_ratio': 0.25},
        'wikimovie': {'upstream_top_k': 40, 'distant_gt_top_k': 2, 'down_sample_ratio': 0.25},
    }

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
    # hotpot_train_list = common.load_json(config.TRAIN_FILE)
    # hotpot_dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    # hotpot_dev_o_dict = list_dict_data_tool.list_to_dict(hotpot_dev_list, '_id')

    # Load Hotpot upstream paragraph forward item
    hotpot_dev_fitems_list = common.load_jsonl(
        config.PDATA_ROOT / "content_selection_forward" / "hotpot_dev_p_level_unlabeled.jsonl")
    hotpot_train_fitems_list = common.load_jsonl(
        config.PDATA_ROOT / "content_selection_forward" / "hotpot_train_p_level_labeled.jsonl")

    hotpot_train_fitems_list = hotpot_sampler_utils.field_name_convert(hotpot_train_fitems_list, 'doc_t', 'element')
    hotpot_dev_fitems_list = hotpot_sampler_utils.field_name_convert(hotpot_dev_fitems_list, 'doc_t', 'element')

    # Load FEVER Dataset
    # fever_train_list = common.load_json(config.FEVER_TRAIN)
    # fever_dev_list = common.load_jsonl(config.FEVER_DEV)
    # fever_dev_o_dict = list_dict_data_tool.list_to_dict(fever_dev_list, 'id')

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

    # Load Open QA Dataset.
    webq_test_fitem_list = open_domain_p_sampler.prepare_forward_data('webq', 'test', False, debug=debug_mode,
                                                                      upstream_top_k=40)
    webq_train_fitem_list = open_domain_p_sampler.prepare_forward_data('webq', 'train', True,
                                                                       open_qa_paras['webq']['upstream_top_k'],
                                                                       open_qa_paras['webq']['distant_gt_top_k'],
                                                                       open_qa_paras['webq']['down_sample_ratio'],
                                                                       debug=debug_mode)
    webq_test_gt_list = common.load_jsonl(config.OPEN_WEBQ_TEST_GT)

    curatedtrec_test_fitem_list = open_domain_p_sampler.prepare_forward_data('curatedtrec', 'test', False,
                                                                             upstream_top_k=40,
                                                                             debug=debug_mode)
    curatedtrec_train_fitem_list = open_domain_p_sampler.prepare_forward_data('curatedtrec', 'train', True,
                                                                              open_qa_paras['curatedtrec'][
                                                                                  'upstream_top_k'],
                                                                              open_qa_paras['curatedtrec'][
                                                                                  'distant_gt_top_k'],
                                                                              open_qa_paras['curatedtrec'][
                                                                                  'down_sample_ratio'],
                                                                              debug=debug_mode)
    curatedtrec_test_gt_list = common.load_jsonl(config.OPEN_CURATEDTERC_TEST_GT)

    squad_dev_fitem_list = open_domain_p_sampler.prepare_forward_data('squad', 'dev', False,
                                                                      upstream_top_k=40,
                                                                      debug=debug_mode)
    squad_train_fitem_list = open_domain_p_sampler.prepare_forward_data('squad', 'train', True,
                                                                        open_qa_paras['squad'][
                                                                            'upstream_top_k'],
                                                                        open_qa_paras['squad'][
                                                                            'distant_gt_top_k'],
                                                                        open_qa_paras['squad'][
                                                                            'down_sample_ratio'],
                                                                        debug=debug_mode)
    squad_dev_gt_list = common.load_jsonl(config.OPEN_SQUAD_DEV_GT)

    wikimovie_test_fitem_list = open_domain_p_sampler.prepare_forward_data('wikimovie', 'test', False,
                                                                           upstream_top_k=40,
                                                                           debug=debug_mode)
    wikimovie_train_fitem_list = open_domain_p_sampler.prepare_forward_data('wikimovie', 'train', True,
                                                                            open_qa_paras['wikimovie'][
                                                                                'upstream_top_k'],
                                                                            open_qa_paras['wikimovie'][
                                                                                'distant_gt_top_k'],
                                                                            open_qa_paras['wikimovie'][
                                                                                'down_sample_ratio'],
                                                                            debug=debug_mode)
    wikimovie_test_gt_list = common.load_jsonl(config.OPEN_WIKIM_TEST_GT)

    # Load squadv11 forward:
    squad_v11_pos_fitems = get_squad_question_selection_forward_list(common.load_json(config.SQUAD_TRAIN_1_1))

    if debug_mode:
        webq_test_gt_list = webq_test_gt_list[:50]
        curatedtrec_test_gt_list = curatedtrec_test_gt_list[:50]
        squad_dev_gt_list = squad_dev_gt_list[:50]
        wikimovie_test_gt_list = wikimovie_test_gt_list[:50]

        # hotpot_dev_list = hotpot_dev_list[:10]
        hotpot_dev_fitems_list = hotpot_dev_fitems_list[:296]
        hotpot_train_fitems_list = hotpot_train_fitems_list[:300]

        # fever_dev_list = fever_dev_list[:100]
        eval_frequency = 2

    webq_test_gt_dict = list_dict_data_tool.list_to_dict(webq_test_gt_list, 'question')
    curatedtrec_test_gt_dict = list_dict_data_tool.list_to_dict(curatedtrec_test_gt_list, 'question')
    squad_dev_gt_dict = list_dict_data_tool.list_to_dict(squad_dev_gt_list, 'question')
    wikimovie_test_gt_dict = list_dict_data_tool.list_to_dict(wikimovie_test_gt_list, 'question')

    # Down_sample for hotpot.
    hotpot_sampled_train_list = down_sample_neg(hotpot_train_fitems_list, ratio=hotpot_pos_ratio)
    if hotpot_train_size is None:
        hotpot_est_datasize = len(hotpot_sampled_train_list)
    else:
        hotpot_est_datasize = hotpot_train_size

    if fever_train_size is None:
        fever_est_datasize = len(fever_train_fitems_list)
    else:
        fever_est_datasize = fever_train_size

    sampled_squad_v11_pos_fitems = squad_v11_pos_fitems[:squad_v11_pos_size]

    webq_est_datasize = len(webq_train_fitem_list[:webq_train_size])
    curatedtrec_est_datasize = len(curatedtrec_train_fitem_list[:curatedtrec_train_size])
    squad_est_datasize = len(squad_train_fitem_list[:squad_train_size])
    wikimovie_est_datasize = len(wikimovie_train_fitem_list[:wikimovie_train_size])

    print("Hotpot Train Size:", hotpot_est_datasize)
    print("Fever Train Size:", fever_est_datasize)
    print("WebQ Train Size:", webq_est_datasize)
    print("TREC Train Size:", curatedtrec_est_datasize)
    print("SQuAD Train Size:", squad_est_datasize)
    print("WikiMovie Train Size:", wikimovie_est_datasize)

    print("SQuADv11 pos size:", len(sampled_squad_v11_pos_fitems))

    est_datasize = hotpot_est_datasize + fever_est_datasize + webq_est_datasize + curatedtrec_est_datasize + \
                   len(sampled_squad_v11_pos_fitems) + squad_est_datasize + wikimovie_est_datasize

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

    # hotpot_dev_instances = bert_cs_reader.read(hotpot_dev_fitems_list)
    # fever_dev_instances = bert_cs_reader.read(fever_dev_fitems_list)
    webq_test_instance = bert_cs_reader.read(webq_test_fitem_list)
    curatedtrec_test_instance = bert_cs_reader.read(curatedtrec_test_fitem_list)
    squad_dev_instance = bert_cs_reader.read(squad_dev_fitem_list)
    wikimovie_test_instance = bert_cs_reader.read(wikimovie_test_fitem_list)

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
        random.shuffle(hotpot_sampled_train_list)
        hotpot_sampled_train_list = hotpot_sampled_train_list[:hotpot_train_size]

        random.shuffle(fever_train_fitems_list)
        fever_train_fitems_list = fever_train_fitems_list[:fever_train_size]

        random.shuffle(squad_v11_pos_fitems)
        sampled_squad_v11_pos_fitems = squad_v11_pos_fitems[:squad_v11_pos_size]

        all_train_data = hotpot_sampled_train_list + fever_train_fitems_list + sampled_squad_v11_pos_fitems
        # all_train_data = []

        webq_train_fitem_list = open_domain_p_sampler.prepare_forward_data('webq', 'train', True,
                                                                           open_qa_paras['webq']['upstream_top_k'],
                                                                           open_qa_paras['webq']['distant_gt_top_k'],
                                                                           open_qa_paras['webq']['down_sample_ratio'],
                                                                           debug=debug_mode)
        curatedtrec_train_fitem_list = open_domain_p_sampler.prepare_forward_data('curatedtrec', 'train', True,
                                                                                  open_qa_paras['curatedtrec'][
                                                                                      'upstream_top_k'],
                                                                                  open_qa_paras['curatedtrec'][
                                                                                      'distant_gt_top_k'],
                                                                                  open_qa_paras['curatedtrec'][
                                                                                      'down_sample_ratio'],
                                                                                  debug=debug_mode)
        squad_train_fitem_list = open_domain_p_sampler.prepare_forward_data('squad', 'train', True,
                                                                            open_qa_paras['squad'][
                                                                                'upstream_top_k'],
                                                                            open_qa_paras['squad'][
                                                                                'distant_gt_top_k'],
                                                                            open_qa_paras['squad'][
                                                                                'down_sample_ratio'],
                                                                            debug=debug_mode)
        wikimovie_train_fitem_list = open_domain_p_sampler.prepare_forward_data('wikimovie', 'train', True,
                                                                                open_qa_paras['wikimovie'][
                                                                                    'upstream_top_k'],
                                                                                open_qa_paras['wikimovie'][
                                                                                    'distant_gt_top_k'],
                                                                                open_qa_paras['wikimovie'][
                                                                                    'down_sample_ratio'],
                                                                                debug=debug_mode)

        random.shuffle(squad_train_fitem_list)
        squad_train_fitem_list = squad_train_fitem_list[:squad_train_size]

        random.shuffle(wikimovie_train_fitem_list)
        wikimovie_train_fitem_list = wikimovie_train_fitem_list[:wikimovie_train_size]

        random.shuffle(curatedtrec_train_fitem_list)
        curatedtrec_train_fitem_list = curatedtrec_train_fitem_list[:curatedtrec_train_size]

        random.shuffle(webq_train_fitem_list)
        webq_train_fitem_list = webq_train_fitem_list[:webq_train_size]

        all_train_data = all_train_data + webq_train_fitem_list + curatedtrec_train_fitem_list + \
                         squad_train_fitem_list + wikimovie_train_fitem_list

        print("Current all train size:", len(all_train_data))

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
                    eval_open_qa_procedure(biterator, webq_test_instance, model, device_num, 1, webq_test_gt_list,
                                           webq_test_gt_dict, debug_mode, logging_agent, update_step, epoch_i,
                                           file_path_prefix,
                                           do_ema, ema, seed, 'webq')

                    eval_open_qa_procedure(biterator, curatedtrec_test_instance, model, device_num, 1,
                                           curatedtrec_test_gt_list,
                                           curatedtrec_test_gt_dict, debug_mode, logging_agent, update_step, epoch_i,
                                           file_path_prefix,
                                           do_ema, ema, seed, 'curatedtrec')

                    eval_open_qa_procedure(biterator, squad_dev_instance, model, device_num, 1,
                                           squad_dev_gt_list,
                                           squad_dev_gt_dict, debug_mode, logging_agent, update_step, epoch_i,
                                           file_path_prefix,
                                           do_ema, ema, seed, 'squad')

                    eval_open_qa_procedure(biterator, wikimovie_test_instance, model, device_num, 1,
                                           wikimovie_test_gt_list,
                                           wikimovie_test_gt_dict, debug_mode, logging_agent, update_step, epoch_i,
                                           file_path_prefix,
                                           do_ema, ema, seed, 'wikimovie')
                    # Eval FEVER
                    # eval_fever_procedure(biterator, fever_dev_instances, model, device_num, 1, fever_dev_list,
                    #                      fever_dev_o_dict, debug_mode, logging_agent, update_step, epoch_i,
                    #                      file_path_prefix,
                    #                      do_ema, ema, seed)
                    # eval_hotpot_procedure(biterator, hotpot_dev_instances, model, device_num, 1, hotpot_dev_list,
                    #                       hotpot_dev_o_dict, debug_mode, logging_agent, update_step, epoch_i,
                    #                       file_path_prefix, do_ema, ema, seed)
    epoch_i = num_train_epochs - 1
    eval_open_qa_procedure(biterator, webq_test_instance, model, device_num, 1, webq_test_gt_list,
                           webq_test_gt_dict, debug_mode, logging_agent, update_step, epoch_i,
                           file_path_prefix,
                           do_ema, ema, seed, 'webq')

    eval_open_qa_procedure(biterator, curatedtrec_test_instance, model, device_num, 1,
                           curatedtrec_test_gt_list,
                           curatedtrec_test_gt_dict, debug_mode, logging_agent, update_step, epoch_i,
                           file_path_prefix,
                           do_ema, ema, seed, 'curatedtrec')

    eval_open_qa_procedure(biterator, squad_dev_instance, model, device_num, 1,
                           squad_dev_gt_list,
                           squad_dev_gt_dict, debug_mode, logging_agent, update_step, epoch_i,
                           file_path_prefix,
                           do_ema, ema, seed, 'squad')

    eval_open_qa_procedure(biterator, wikimovie_test_instance, model, device_num, 1,
                           wikimovie_test_gt_list,
                           wikimovie_test_gt_dict, debug_mode, logging_agent, update_step, epoch_i,
                           file_path_prefix,
                           do_ema, ema, seed, 'wikimovie')

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


def selective_eval(model_path):
    seed = 12
    torch.manual_seed(seed)
    # bert_model_name = 'bert-large-uncased'
    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    bert_model_name = 'bert-base-uncased'
    lazy = True
    # lazy = True
    forward_size = 128
    do_lower_case = True
    max_l = 264

    debug_mode = False

    open_qa_paras = {
        'webq': {'upstream_top_k': 40, 'distant_gt_top_k': 2, 'down_sample_ratio': None},
        'curatedtrec': {'upstream_top_k': 40, 'distant_gt_top_k': 2, 'down_sample_ratio': None},
        'squad': {'upstream_top_k': 30, 'distant_gt_top_k': 1, 'down_sample_ratio': None},
        'wikimovie': {'upstream_top_k': 40, 'distant_gt_top_k': 2, 'down_sample_ratio': None},
    }

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

    # Load Open QA Dataset.
    webq_test_fitem_list = open_domain_p_sampler.prepare_forward_data('webq', 'test', False, debug=debug_mode,
                                                                      upstream_top_k=40)
    webq_train_fitem_list = open_domain_p_sampler.prepare_forward_data('webq', 'train', True,
                                                                       open_qa_paras['webq']['upstream_top_k'],
                                                                       open_qa_paras['webq']['distant_gt_top_k'],
                                                                       open_qa_paras['webq']['down_sample_ratio'],
                                                                       debug=debug_mode)
    webq_test_gt_list = common.load_jsonl(config.OPEN_WEBQ_TEST_GT)
    webq_train_gt_list = common.load_jsonl(config.OPEN_WEBQ_TRAIN_GT)

    curatedtrec_test_fitem_list = open_domain_p_sampler.prepare_forward_data('curatedtrec', 'test', False,
                                                                             upstream_top_k=40,
                                                                             debug=debug_mode)
    curatedtrec_train_fitem_list = open_domain_p_sampler.prepare_forward_data('curatedtrec', 'train', True,
                                                                              open_qa_paras['curatedtrec'][
                                                                                  'upstream_top_k'],
                                                                              open_qa_paras['curatedtrec'][
                                                                                  'distant_gt_top_k'],
                                                                              open_qa_paras['curatedtrec'][
                                                                                  'down_sample_ratio'],
                                                                              debug=debug_mode)
    curatedtrec_test_gt_list = common.load_jsonl(config.OPEN_CURATEDTERC_TEST_GT)
    curatedtrec_train_gt_list = common.load_jsonl(config.OPEN_CURATEDTERC_TRAIN_GT)

    squad_dev_fitem_list = open_domain_p_sampler.prepare_forward_data('squad', 'dev', False,
                                                                      upstream_top_k=40,
                                                                      debug=debug_mode)
    squad_train_fitem_list = open_domain_p_sampler.prepare_forward_data('squad', 'train', True,
                                                                        open_qa_paras['squad'][
                                                                            'upstream_top_k'],
                                                                        open_qa_paras['squad'][
                                                                            'distant_gt_top_k'],
                                                                        open_qa_paras['squad'][
                                                                            'down_sample_ratio'],
                                                                        debug=debug_mode)
    squad_dev_gt_list = common.load_jsonl(config.OPEN_SQUAD_DEV_GT)
    squad_train_gt_list = common.load_jsonl(config.OPEN_SQUAD_TRAIN_GT)

    wikimovie_test_fitem_list = open_domain_p_sampler.prepare_forward_data('wikimovie', 'test', False,
                                                                           upstream_top_k=40,
                                                                           debug=debug_mode)
    wikimovie_train_fitem_list = open_domain_p_sampler.prepare_forward_data('wikimovie', 'train', True,
                                                                            open_qa_paras['wikimovie'][
                                                                                'upstream_top_k'],
                                                                            open_qa_paras['wikimovie'][
                                                                                'distant_gt_top_k'],
                                                                            open_qa_paras['wikimovie'][
                                                                                'down_sample_ratio'],
                                                                            debug=debug_mode)
    wikimovie_test_gt_list = common.load_jsonl(config.OPEN_WIKIM_TEST_GT)
    wikimovie_train_gt_list = common.load_jsonl(config.OPEN_WIKIM_TRAIN_GT)

    # Load squadv11 forward:

    webq_test_gt_dict = list_dict_data_tool.list_to_dict(webq_test_gt_list, 'question')
    curatedtrec_test_gt_dict = list_dict_data_tool.list_to_dict(curatedtrec_test_gt_list, 'question')
    squad_dev_gt_dict = list_dict_data_tool.list_to_dict(squad_dev_gt_list, 'question')
    wikimovie_test_gt_dict = list_dict_data_tool.list_to_dict(wikimovie_test_gt_list, 'question')

    webq_train_gt_dict = list_dict_data_tool.list_to_dict(webq_train_gt_list, 'question')
    curatedtrec_train_gt_dict = list_dict_data_tool.list_to_dict(curatedtrec_train_gt_list, 'question')
    squad_train_gt_dict = list_dict_data_tool.list_to_dict(squad_train_gt_list, 'question')
    wikimovie_train_gt_dict = list_dict_data_tool.list_to_dict(wikimovie_train_gt_list, 'question')

    webq_est_datasize = len(webq_train_fitem_list)
    curatedtrec_est_datasize = len(curatedtrec_train_fitem_list)

    print("WebQ Train Size:", webq_est_datasize)
    print("TREC Train Size:", curatedtrec_est_datasize)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                                   cache_dir=bert_pretrain_path)
    bert_cs_reader = BertContentSelectionReader(bert_tokenizer, lazy, is_paired=True,
                                                example_filter=lambda x: len(x['context']) == 0, max_l=max_l,
                                                element_fieldname='element')

    bert_encoder = BertModel.from_pretrained(bert_model_name, cache_dir=bert_pretrain_path)
    model = BertMultiLayerSeqClassification(bert_encoder, num_labels=num_class, num_of_pooling_layer=1,
                                            act_type='tanh', use_pretrained_pooler=True, use_sigmoid=True)

    model.load_state_dict(torch.load(model_path))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    #

    webq_test_instance = bert_cs_reader.read(webq_test_fitem_list)
    curatedtrec_test_instance = bert_cs_reader.read(curatedtrec_test_fitem_list)
    squad_dev_instance = bert_cs_reader.read(squad_dev_fitem_list)
    wikimovie_test_instance = bert_cs_reader.read(wikimovie_test_fitem_list)

    webq_train_instance = bert_cs_reader.read(webq_train_fitem_list)
    curatedtrec_train_instance = bert_cs_reader.read(curatedtrec_train_fitem_list)
    squad_train_instance = bert_cs_reader.read(squad_train_fitem_list)
    wikimovie_train_instance = bert_cs_reader.read(wikimovie_train_fitem_list)

    print('webq:', len(webq_train_fitem_list))
    print('curatedtrec:', len(curatedtrec_train_fitem_list))
    print('squad:', len(squad_train_fitem_list))
    print('wikimovie:', len(wikimovie_train_fitem_list))

    biterator = BasicIterator(batch_size=forward_size)
    biterator.index_with(vocab)

    # separate_eval_open_qa_procedure(biterator, curatedtrec_test_instance, model, 0,
    #                                 curatedtrec_test_gt_list, curatedtrec_test_gt_dict, 'curatedtrec',
    #                                 save_path=".", tag='test')
    #
    # separate_eval_open_qa_procedure(biterator, curatedtrec_train_instance, model, 0,
    #                                 curatedtrec_train_gt_list, curatedtrec_train_gt_dict, 'curatedtrec',
    #                                 save_path=".", tag='train')

    # separate_eval_open_qa_procedure(biterator, squad_train_instance, model, 0,
    #                                 squad_train_gt_list, squad_train_gt_dict, 'squad',
    #                                 save_path=".", tag='train')

    # separate_eval_open_qa_procedure(biterator, wikimovie_train_instance, model, 0,
    #                                 wikimovie_train_gt_list, wikimovie_train_gt_dict, 'wikimovie',
    #                                 save_path=".", tag='train')

    separate_eval_open_qa_procedure(biterator, webq_train_instance, model, 0,
                                    webq_train_gt_list, webq_train_gt_dict, 'webq',
                                    save_path=".", tag='train')


if __name__ == '__main__':
    multitask_open_qa_model_go()
    # model_path = config.PRO_ROOT / "saved_models/05-12-20:32:15_mtr_open_qa_p_level_(num_train_epochs:3)/i(3837)|e(3)_final_ema_model"
    # selective_eval(model_path)