from collections import Counter

import config
from evaluation import ext_hotpot_eval
from utils import common
import numpy as np
from hotpot_doc_retri.retrieval_utils import RetrievedItem, RetrievedSet

# metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
#                'doc_em': 0, 'doc_f1': 0, 'doc_prec': 0, 'doc_recall': 0,
#                'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
#                'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}


def sp_doc_analysis(item):
    # if item['type'] == 'comparison':
    # if item['type'] == 'bridge':
    #     return True

    if item['doc_recall'] != 1.0 and item['type'] == 'bridge':
        return True
    # if item['doc_recall'] == 0.0 and item['type'] == 'comparison':
    # if item['doc_prec'] == 0.0 and item['type'] == 'comparison':
    # if item['doc_recall'] == 0.0 and item['type'] == 'bridge':
    # if item['doc_prec'] == 0.0 and item['type'] == 'bridge':
    # if item['doc_recall'] != 1.0 and item['type'] == 'bridge':
    # if 1 >= item['doc_recall'] >= 0.0 and item['type'] == 'comparison':
    #     return True
    return False


def sp_position_analysis(item, counter):
    supoorting_fact = item['supporting_facts']
    sent_numbers = []
    for doc, sent_number in supoorting_fact:
        sent_numbers.append(sent_number)
    counter.update(sent_numbers)


def filter_analysis(score_tracker, filter_func, max_count=None, show_info=None, additional_item=None):
    count = 0
    total_count = 0
    for k, v in score_tracker.items():
        if filter_func(v):
            count += 1
            if show_info is None:
                print(v)
            else:
                # print out information that is needed
                pinfo = dict()
                for key in show_info:
                    if key in v:
                        pinfo[key] = v[key]
                    elif additional_item is not None and key in additional_item:
                        pinfo[key] = additional_item[key][k]
                    else:
                        raise ValueError(f"Key Value {key} is not presented in both score tracker and additaional_item.")
                print(pinfo)

        total_count += 1

        if max_count is not None and count == max_count:
            break

    print(count, total_count, count / total_count)


def counter_analysis(score_tracker):
    c = Counter()
    for k, v in score_tracker.items():
        sp_position_analysis(v, c)
    print(c)


def get_sp_position_count():
    train_list = common.load_json(config.TRAIN_FILE)
    c = Counter()
    for item in train_list:
        sp_position_analysis(item, c)

    print(c)


if __name__ == '__main__':
    # pred_dev = common.load_json(
        # config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_uncased_v3_file.json")
        # config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_v5_file.json")
    # pred_dev = common.load_json(config.RESULT_PATH / "doc_retri_results/doc_raw_matching_file.json")
    # pred_dev = common.load_json("/Users/yixin/projects/extinguishHotpot/src/doc_retri/doc_raw_matching_with_disamb_withiout_hyperlinked_v6_file_debug_4.json")
    # pred_dev = common.load_json("/Users/yixin/projects/extinguishHotpot/src/doc_retri/doc_raw_matching_with_disamb_withiout_hyperlinked_v7_file_debug_top3.json")
    # pred_dev = common.load_json("/Users/yixin/projects/extinguishHotpot/src/doc_retri/doc_raw_matching_with_disamb_withiout_hyperlinked_v7_file_debug_top2.json")
    pred_dev = common.load_json("/Users/yixin/projects/extinguishHotpot/src/doc_retri/doc_raw_matching_with_disamb_with_hyperlinked_v7_file_pipeline_top_none.json")
    # pred_dev = common.load_json("/Users/yixin/projects/extinguishHotpot/src/doc_retri/doc_raw_matching_with_disamb_with_hyperlinked_v7_file_pipeline_top_none_redo_0.json")



    print(pred_dev['raw_retrieval_set']['5a8e3ea95542995a26add48d'])
    # pred_dev = common.load_json(config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_withiout_hyperlinked_v5_file.json")

    print(pred_dev.keys())

    # len_counter = Counter()
    len_list = []

    # for rset in pred_dev['raw_retrieval_set'].values():
    #     len_list.append(len(rset))

    for rset in pred_dev['sp_doc'].values():
        len_list.append(len(rset))

    print(Counter(len_list).most_common(10000))

    # exit(0)
    #     print()
    #     print(len(rset))
    # pred_dev = common.load_json(config.RESULT_PATH / "doc_retri_results/toy_doc_rm_stopword_pred_file.json")
    # pred_dev = common.load_json(config.RESULT_PATH / "doc_retriesults/toy_doc_rm_stopword_pred_file.json")

    print(len(pred_dev))
    print(np.mean(len_list))
    print(np.std(len_list))
    print(np.max(len_list))
    print(np.min(len_list))

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    global_score_tracker, metric = ext_hotpot_eval.eval(pred_dev, dev_fullwiki_list)

    print(metric)

    filter_analysis(global_score_tracker, sp_doc_analysis, max_count=25,
                    show_info=['question', 'answer', 'sp_doc', 'supporting_facts', 'doc_recall',
                               'doc_prec', 'type', 'raw_retrieval_set'], additional_item=pred_dev)

    # counter_analysis(global_score_tracker)

    # for key, value in global_score_tracker.items():
    #     print(key)
    #     print(value)
    #     print(value.keys())
    #     break

