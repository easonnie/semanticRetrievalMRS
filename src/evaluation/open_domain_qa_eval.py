import sys
# import ujson as json
import json
import re
import string
from collections import Counter
import pickle
import config
import numpy as np

import evaluation
from evaluation.squad_eval_v1 import get_score
from span_prediction_task_utils.squad_utils import get_squad_question_answer_list

empty_metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def regex_match_score(prediction, pattern):
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE
        )
    except BaseException:
        print('Regular expression failed to compile: %s' % pattern)
        return False
    return compiled.match(prediction) is not None


def update_answer(metrics, prediction, gold, match_type='string'):
    if not isinstance(gold, list):
        gold = [gold]
    if match_type == 'string':
        max_em, max_f1, max_prec, max_recall = -1, -1, -1, -1
        for g_anw in gold:
            em = exact_match_score(prediction, g_anw)
            f1, prec, recall = f1_score(prediction, g_anw)
            max_em = max(float(em), max_em)
            max_f1 = max(f1, max_f1)
            max_prec = max(prec, max_prec)
            max_recall = max(recall, max_recall)
        metrics['em'] += float(max_em)
        metrics['f1'] += max_f1
        metrics['prec'] += max_prec
        metrics['recall'] += max_recall
    elif match_type == 'regex':
        max_em, max_f1, max_prec, max_recall = -1, 0, 0, 0
        for g_anw in gold:
            em = regex_match_score(prediction, g_anw)
            # f1, prec, recall = f1_score(prediction, g_anw)
            f1, prec, recall = 0, 0, 0
            max_em = max(float(em), max_em)
            max_f1 = max(f1, max_f1)
            max_prec = max(prec, max_prec)
            max_recall = max(recall, max_recall)

        metrics['em'] += float(max_em)
        metrics['f1'] += max_f1
        metrics['prec'] += max_prec
        metrics['recall'] += max_recall
    else:
        raise ValueError(f"Invalid Match Type: {match_type}")

    return max_em, max_f1, max_prec, max_recall


def update_document(metrics, prediction, gold):
    cur_sp_pred = set(prediction)

    gold_sp_pred = set()
    for doc, ln in map(tuple, gold):
        gold_sp_pred.add(doc)

    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['doc_em'] += em
    metrics['doc_f1'] += f1
    metrics['doc_prec'] += prec
    metrics['doc_recall'] += recall
    return em, prec, recall, f1


def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall, f1


# Important
# The metric for each example has the following:
# metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
#                'doc_em': 0, 'doc_f1': 0, 'doc_prec': 0, 'doc_recall': 0,
#                'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
#                'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
# global_score track will contain useful information for error analysis
# question, answer, supporting_facts, type, level are provided as groundturth annotated information
#                       sp_doc, sp, p_answer are predictions.

def qa_paragraph_eval(pred_list, gt_list, top_k=100):
    assert len(pred_list) == len(gt_list)
    total = len(pred_list)
    hit = 0
    for p_item, gt_item in zip(pred_list, gt_list):
        gt_set = set()
        for item in gt_item["gt_p_list"]:
            gt_set.add((item[0], item[1]))

        for item, score in p_item['score_list'][:top_k]:
            cur_item = (item[0], item[1])
            if cur_item in gt_set:
                hit += 1
                break

    print(total, hit, hit/total)


def qa_paragraph_eval_v1(pred_list, gt_list):
    assert len(pred_list) == len(gt_list)
    total = len(pred_list)
    hit = 0
    pred_len_list = []
    for p_item, gt_item in zip(pred_list, gt_list):
        gt_set = set()
        for item in gt_item["gt_p_list"]:
            gt_set.add((item[0], item[1]))

        pred_len_list.append(len(p_item['pred_p_list']))
        for item in p_item['pred_p_list']:
            cur_item = (item[0], item[1])
            if cur_item in gt_set:
                hit += 1
                break

    len_counter = Counter(pred_len_list)
    print(len_counter.most_common())
    print(np.mean(pred_len_list))
    print(np.std(pred_len_list))
    recall = hit / total
    return recall


def qa_eval(prediction, gold, verbose=True, type='string', missing_ignore=True):
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}

    global_score_tracker = {}

    eval_answer = True

    if 'p_answer' not in prediction:
        print('Key \'p_answer\' not in prediction, do not evaluation answer performance.')
        eval_answer = False

    for dp in gold:
        cur_id = dp['qid']
        global_score_tracker[cur_id] = {}

        # More subtype can go here
        global_score_tracker[cur_id].update(empty_metrics)
        global_score_tracker[cur_id].update(dp)

        if eval_answer:
            if cur_id not in prediction['p_answer']:
                if not missing_ignore:
                    print('missing answer {}'.format(cur_id))
                # global_score_tracker[cur_id].update({'p_answer': ""})
            else:
                global_score_tracker[cur_id].update({'p_answer': prediction['p_answer'][cur_id]})

                em, prec, recall, f1 = update_answer(
                    metrics, prediction['p_answer'][cur_id], dp['answers'], match_type=type)
                global_score_tracker[cur_id]['em'] = em
                global_score_tracker[cur_id]['prec'] = prec
                global_score_tracker[cur_id]['recall'] = recall
                global_score_tracker[cur_id]['f1'] = f1

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    if verbose:
        print("Total examples:", N)
        print(metrics)

    scorer_all = {}
    scorer_all.update(empty_metrics)
    total_count = 0
    for k, value in global_score_tracker.items():
        for metrices_name in scorer_all.keys():
            scorer_all[metrices_name] += global_score_tracker[k][metrices_name]
        total_count += 1

    # for k in scorer_all.keys():
    #     print(k, scorer_all[k] / total_count)
    #

    if total_count != N:
        print(f"Potential Duplicate Question, {total_count}, {N}")

    if len(global_score_tracker) != len(gold):
        print("Same issue above.")

    return global_score_tracker, metrics


if __name__ == '__main__':

    predict_dict = {'p_answer': {
        '133': 'go w on',
        # '133': 'what'
    }}
    #
    target = [{'qid': '133', 'answers': ['go\\s\\w+\\s\w*']},
              {'qid': '133', 'answers': ['go\\s\\w+\\s\w*']}]
    qa_eval(predict_dict, target, type='regex')

    # from utils import common
    # length = 0
    # gt_list = common.load_jsonl(config.OPEN_SQUAD_TRAIN_GT)
    # print(len(gt_list))
    # question_set = dict()
    # for item in gt_list:
    #     question = item['question']
    #     if question in question_set:
    #         if question_set[question] != item['answers']:
    #             print(question, question_set[question], item['answers'])
    #             length += 1
    #     else:
    #         question_set[question] = item['answers']
    #
    # print(length)

    # squad_v11 = common.load_json(config.SQUAD_DEV_1_1)
    # squad_v11_list, random_dict = get_squad_question_answer_list(squad_v11)
    #
    # predict_dict = {'p_answer':
    #     random_dict}
    # # target = [{'qid': '133', 'answers': ['go on']}]
    # qa_eval(predict_dict, squad_v11_list, type='regex')
    # get_score(predict_dict, target)

    # from utils import common
    # pred_list = common.load_jsonl(config.PRO_ROOT / "data/p_webq/tf_idf_p_level/webq_train_para_tfidf.jsonl")
    # gt_list = common.load_jsonl(config.OPEN_WEBQ_TRAIN_GT)
    # qa_paragraph_eval(pred_list, gt_list, top_k=80)
    #
    # pred_list = common.load_jsonl(config.PRO_ROOT / "data/p_webq/tf_idf_p_level/webq_test_para_tfidf.jsonl")
    # gt_list = common.load_jsonl(config.OPEN_WEBQ_TEST_GT)
    # qa_paragraph_eval(pred_list, gt_list, top_k=40)
    #
    # pred_list = common.load_jsonl(
    #     config.PRO_ROOT / "data/p_curatedtrec/tf_idf_p_level/curatedtrec_train_para_tfidf.jsonl")
    # gt_list = common.load_jsonl(config.OPEN_CURATEDTERC_TRAIN_GT)
    # qa_paragraph_eval(pred_list, gt_list, top_k=60)
    #
    # pred_list = common.load_jsonl(config.PRO_ROOT / "data/p_curatedtrec/tf_idf_p_level/curatedtrec_test_para_tfidf.jsonl")
    # gt_list = common.load_jsonl(config.OPEN_CURATEDTERC_TEST_GT)
    # qa_paragraph_eval(pred_list, gt_list, top_k=10)
    #
    # pred_list = common.load_jsonl(
    #     config.PRO_ROOT / "data/p_squad/tf_idf_p_level/squad_train_para_tfidf.jsonl")
    # gt_list = common.load_jsonl(config.OPEN_SQUAD_TRAIN_GT)
    # qa_paragraph_eval(pred_list, gt_list, top_k=60)
    # #
    # pred_list = common.load_jsonl(
    #     config.PRO_ROOT / "data/p_squad/tf_idf_p_level/squad_dev_para_tfidf.jsonl")
    # gt_list = common.load_jsonl(config.OPEN_SQUAD_DEV_GT)
    # qa_paragraph_eval(pred_list, gt_list, top_k=60)

    # pred_list = common.load_jsonl(
    #     config.PRO_ROOT / "data/p_wikimovie/kwm_p_level/wikimovie_train_kwm_tfidf.jsonl")
    # gt_list = common.load_jsonl(config.OPEN_WIKIM_TRAIN_GT)
    # qa_paragraph_eval(pred_list, gt_list, top_k=60)
    #
    # pred_list = common.load_jsonl(
    #     config.PRO_ROOT / "data/p_wikimovie/kwm_p_level/wikimovie_test_kwm_tfidf.jsonl")
    # gt_list = common.load_jsonl(config.OPEN_WIKIM_TEST_GT)
    # qa_paragraph_eval(pred_list, gt_list, top_k=60)

    # pred_list = common.load_jsonl(
    #     config.PRO_ROOT / "data/p_wikimovie/tf_idf_p_level/wikimovie_test_para_tfidf.txt")
    # gt_list = common.load_jsonl(config.OPEN_WIKIM_TEST_GT)
    # qa_paragraph_eval(pred_list, gt_list, top_k=40)

