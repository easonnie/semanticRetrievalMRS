import json
import config
from evaluation import fever_scorer
import copy

from fever_sampler.nli_new_sampler import get_nli_pair
from utils import common, list_dict_data_tool
import numpy as np


def ensemble_nli_results(nli_r_list):
    id2label = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO",
    }

    r_len = len(nli_r_list[0])
    for nli_r in nli_r_list:
        assert len(nli_r) == r_len

    new_list = copy.deepcopy(nli_r_list[0])
    logits_list = []
    for i in range(r_len):
        logits_current_logits_list = []
        for nli_r in nli_r_list:
            assert nli_r[i]['oid'] == new_list[i]['oid']
            logits_current_logits_list.append(np.asarray(nli_r[i]['logits'], dtype=np.float32))  # [(3)]
        logits_current_logits = np.stack(logits_current_logits_list, axis=0)  # [num, 3]
        current_mean_logits = np.mean(logits_current_logits, axis=0)  # [3]
        logits_list.append(current_mean_logits)

    logits = np.stack(logits_list, axis=0)  # (len, 3)
    y_ = np.argmax(logits, axis=1)  # (len)
    assert y_.shape[0] == len(new_list)

    for i in range(r_len):
        new_list[i]['predicted_label'] = id2label[y_[i]]

    return new_list


def build_submission_file(d_list, filename):
    with open(filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            instance_item = dict()
            instance_item['id'] = int(item['id'])
            instance_item['claim'] = item['claim']
            instance_item['predicted_label'] = item['predicted_label']
            instance_item['predicted_evidence'] = item['predicted_evidence']
            out_f.write(json.dumps(instance_item) + "\n")


def delete_unused_evidence(d_list):
    for item in d_list:
        if item['predicted_label'] == 'NOT ENOUGH INFO':
            item['predicted_evidence'] = []


def evidence_adjustment(tag, sent_file, label_file, filter_prob=0.2, top_k=5):
    dev_sent_filtering_prob = filter_prob

    # dev_list = common.load_jsonl(config.FEVER_DEV)
    dev_sent_results_list = common.load_jsonl(sent_file)

    dev_fitems, dev_list = get_nli_pair(tag, is_training=False,
                                        sent_level_results_list=dev_sent_results_list, debug=False,
                                        sent_top_k=top_k, sent_filter_value=dev_sent_filtering_prob)

    cur_eval_results_list = common.load_jsonl(label_file)

    ema_results_dict = list_dict_data_tool.list_to_dict(cur_eval_results_list, 'oid')
    copied_dev_list = copy.deepcopy(dev_list)
    list_dict_data_tool.append_item_from_dict_to_list(copied_dev_list, ema_results_dict,
                                                      'id', 'predicted_label')

    mode = {'standard': True}
    # delete_unused_evidence(copied_dev_list)
    strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(copied_dev_list, dev_list,
                                                                    mode=mode, max_evidence=5)
    logging_item = {
        'ss': strict_score, 'ac': acc_score,
        'pr': pr, 'rec': rec, 'f1': f1,
    }

    print(logging_item)


def eval_ensemble():
    sent_file = config.PRO_ROOT / "data/p_fever/fever_sentence_level/04-24-00-11-19_fever_v0_slevel_retri_(ignore_non_verifiable-True)/fever_s_level_dev_results.jsonl"
    dev_sent_filtering_prob = 0.01
    tag = 'dev'
    top_k = 5

    # dev_list = common.load_jsonl(config.FEVER_DEV)
    dev_sent_results_list = common.load_jsonl(sent_file)

    dev_fitems, dev_list = get_nli_pair(tag, is_training=False,
                                        sent_level_results_list=dev_sent_results_list, debug=False,
                                        sent_top_k=top_k, sent_filter_value=dev_sent_filtering_prob)

    pred_file_list = [
        config.PRO_ROOT / "data/p_fever/fever_nli/04-25-22:02:53_fever_v2_nli_th0.2/ema_i(20000)|e(3)|ss(0.7002700270027002)|ac(0.746024602460246)|pr(0.6141389138913633)|rec(0.8627362736273627)|f1(0.7175148212089147)|seed(12)/nli_dev_label_results_th0.2.jsonl",
        config.PRO_ROOT / "data/p_fever/fever_nli/04-26-10:15:39_fever_v2_nli_th0.2/ema_i(14000)|e(2)|ss(0.6991199119911992)|ac(0.7492249224922493)|pr(0.7129412941294097)|rec(0.8338583858385838)|f1(0.7686736484619933)|seed(12)/nli_dev_label_results_th0.2.jsonl",
        config.PRO_ROOT / "data/p_fever/fever_nli/04-27-10:03:27_fever_v2_nli_th0.2/ema_i(26000)|e(3)|ss(0.6958695869586958)|ac(0.7447744774477447)|pr(0.7129412941294097)|rec(0.8338583858385838)|f1(0.7686736484619933)|seed(12)/nli_dev_label_results_th0.2.jsonl",
    ]
    pred_d_list = [common.load_jsonl(file) for file in pred_file_list]
    final_list = ensemble_nli_results(pred_d_list)
    pred_list = final_list

    ema_results_dict = list_dict_data_tool.list_to_dict(pred_list, 'oid')
    copied_dev_list = copy.deepcopy(dev_list)
    list_dict_data_tool.append_item_from_dict_to_list(copied_dev_list, ema_results_dict,
                                                      'id', 'predicted_label')

    dev_list = common.load_jsonl(config.FEVER_DEV)
    mode = {'standard': True}
    strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(copied_dev_list, dev_list,
                                                                    mode=mode, max_evidence=5)
    logging_item = {
        'ss': strict_score, 'ac': acc_score,
        'pr': pr, 'rec': rec, 'f1': f1,
    }

    print(logging_item)


if __name__ == '__main__':
    eval_ensemble()
    # Get sentence file:
    # evidence_adjustment('dev',
    #     config.PRO_ROOT / "data/p_fever/fever_sentence_level/04-24-00-11-19_fever_v0_slevel_retri_(ignore_non_verifiable-True)/fever_s_level_dev_results.jsonl",
    #     config.PRO_ROOT / "data/p_fever/fever_nli/04-25-22:02:53_fever_v2_nli_th0.2/ema_i(20000)|e(3)|ss(0.7002700270027002)|ac(0.746024602460246)|pr(0.6141389138913633)|rec(0.8627362736273627)|f1(0.7175148212089147)|seed(12)/nli_dev_label_results_th0.2.jsonl",
    # )

    # dev_list = common.load_jsonl(config.FEVER_DEV)
    # prediction_file = config.PRO_ROOT / "data/p_fever/fever_nli/04-25-22:02:53_fever_v2_nli_th0.2/ema_i(20000)|e(3)|ss(0.7002700270027002)|ac(0.746024602460246)|pr(0.6141389138913633)|rec(0.8627362736273627)|f1(0.7175148212089147)|seed(12)/nli_dev_cp_results_th0.2.jsonl"
    # pred_list = common.load_jsonl(prediction_file)
    # mode = {'standard': True}
    # strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(pred_list, dev_list,
    #                                                                 mode=mode, max_evidence=5)
    # logging_item = {
    #     'ss': strict_score, 'ac': acc_score,
    #     'pr': pr, 'rec': rec, 'f1': f1,
    # }
    #
    # print(logging_item)


    # build_submission_file(
    #     common.load_jsonl(config.PRO_ROOT / "data/p_fever/fever_nli/04-25-22:02:53_fever_v2_nli_th0.2/ema_i(20000)|e(3)|ss(0.7002700270027002)|ac(0.746024602460246)|pr(0.6141389138913633)|rec(0.8627362736273627)|f1(0.7175148212089147)|seed(12)/nli_test_cp_results_th0.2.jsonl"),
    #     "pred.jsonl",
    # )