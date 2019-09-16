import sys
# import ujson as json
import json
import re
import string
from collections import Counter
import pickle
import config

empty_metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
                 'doc_em': 0, 'doc_f1': 0, 'doc_prec': 0, 'doc_recall': 0,
                 'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
                 'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}


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


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall, f1


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
def eval(prediction, gold, verbose=True):
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
               'doc_em': 0, 'doc_f1': 0, 'doc_prec': 0, 'doc_recall': 0,
               'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
               'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    global_score_tracker = {}

    eval_doc = True
    eval_answer = True
    eval_sp = True

    if 'sp_doc' not in prediction:
        print('Key \'sp_doc\' not in prediction, do not evaluation document retrieval performance.')
        eval_doc = False

    if 'p_answer' not in prediction:
        print('Key \'p_answer\' not in prediction, do not evaluation answer performance.')
        eval_answer = False

    if 'sp' not in prediction:
        print('Key \'sp\' not in prediction, do not evaluation sp retrieval performance.')
        eval_sp = False

    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        global_score_tracker[cur_id] = {}

        global_score_tracker[cur_id]['type'] = dp['type']
        # More subtype can go here
        global_score_tracker[cur_id].update(empty_metrics)
        global_score_tracker[cur_id].update(dp)

        # First build gold document
        if eval_doc:
            if cur_id not in prediction['sp_doc']:
                print('missing sp doc {}'.format(cur_id))
                # can_eval_joint = False
            else:
                global_score_tracker[cur_id].update({'sp_doc': prediction['sp_doc'][cur_id]})

                doc_em, doc_prec, doc_recall, doc_f1 = update_document(
                    metrics, prediction['sp_doc'][cur_id], dp['supporting_facts'])
                # Supporting facts will not be used later.
                global_score_tracker[cur_id]['doc_em'] = doc_em
                global_score_tracker[cur_id]['doc_prec'] = doc_prec
                global_score_tracker[cur_id]['doc_recall'] = doc_recall
                global_score_tracker[cur_id]['doc_f1'] = doc_f1
        if eval_answer:
            if cur_id not in prediction['p_answer']:
                print('missing answer {}'.format(cur_id))
                can_eval_joint = False
            else:
                global_score_tracker[cur_id].update({'p_answer': prediction['p_answer'][cur_id]})

                em, prec, recall, f1 = update_answer(
                    metrics, prediction['p_answer'][cur_id], dp['answer'])
                global_score_tracker[cur_id]['em'] = em
                global_score_tracker[cur_id]['prec'] = prec
                global_score_tracker[cur_id]['recall'] = recall
                global_score_tracker[cur_id]['f1'] = f1

        if eval_sp:
            if cur_id not in prediction['sp']:
                print('missing sp fact {}'.format(cur_id))
                can_eval_joint = False
            else:
                global_score_tracker[cur_id].update({'sp': prediction['sp'][cur_id]})

                sp_em, sp_prec, sp_recall, sp_f1 = update_sp(
                    metrics, prediction['sp'][cur_id], dp['supporting_facts'])
                global_score_tracker[cur_id]['sp_em'] = sp_em
                global_score_tracker[cur_id]['sp_prec'] = sp_prec
                global_score_tracker[cur_id]['sp_recall'] = sp_recall
                global_score_tracker[cur_id]['sp_f1'] = sp_f1

        if can_eval_joint and eval_sp and eval_answer:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

            global_score_tracker[cur_id]['joint_em'] = joint_em
            global_score_tracker[cur_id]['joint_f1'] = joint_f1
            global_score_tracker[cur_id]['joint_prec'] = joint_prec
            global_score_tracker[cur_id]['joint_recall'] = joint_recall

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
    assert total_count == N

    assert len(global_score_tracker) == len(gold)

    return global_score_tracker, metrics


if __name__ == '__main__':
    with open(config.RESULT_PATH / "samples/sample_dev_pred.json", 'r') as in_f:
        sample_dev_pred = json.load(in_f)

    with open(config.DEV_FULLWIKI_FILE, 'r') as in_f:
        dev_full_wiki = json.load(in_f)

    dev_doc_prediction = {}

    for k, v in sample_dev_pred['sp'].items():
        sp_doc_pred = set()
        for doc, ln in map(tuple, v):
            sp_doc_pred.add(doc)
            dev_doc_prediction[k] = sp_doc_pred

    sample_dev_pred['sp_doc'] = dev_doc_prediction

    eval(sample_dev_pred, dev_full_wiki)

    # {'em': 0.3099257258609048, 'f1': 0.42141167006470304, 'prec': 0.4486902961244628, 'recall': 0.4265058855717989, 'sp_em': 0.07629979743416611, 'sp_f1': 0.44457187043384133, 'sp_prec': 0.47724637242867995, 'sp_recall': 0.4910771357834174, 'joint_em': 0.03187035786630655, 'joint_f1': 0.21004137978705564, 'joint_prec': 0.23764676005965701, 'joint_recall': 0.2341920575474705}
    # {'em': 0.3099257258609048, 'f1': 0.42141167006470304, 'prec': 0.4486902961244628, 'recall': 0.4265058855717989, 'sp_em': 0.07629979743416611, 'sp_f1': 0.44457187043384133, 'sp_prec': 0.47724637242867995, 'sp_recall': 0.4910771357834174, 'joint_em': 0.03187035786630655, 'joint_f1': 0.21004137978705564, 'joint_prec': 0.23764676005965701, 'joint_recall': 0.2341920575474705}
    # {'em': 0.3099257258609048, 'f1': 0.42141167006470304, 'prec': 0.4486902961244628, 'recall': 0.4265058855717989, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.07629979743416611, 'sp_f1': 0.44457187043384133, 'sp_prec': 0.47724637242867995, 'sp_recall': 0.4910771357834174, 'joint_em': 0.03187035786630655, 'joint_f1': 0.21004137978705564, 'joint_prec': 0.23764676005965701, 'joint_recall': 0.2341920575474705}
