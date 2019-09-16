from collections import Counter
from hotpot_doc_retri.retrieval_utils import RetrievedItem, RetrievedSet
import config
from evaluation import ext_hotpot_eval
from utils import common
import numpy as np

if __name__ == '__main__':
    pred_dev_a = common.load_json(
        config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_v2_file.json")

    pred_dev_b = common.load_json(
        config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_v3_file.json")

    all_ids = pred_dev_a['sp_doc'].keys()

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    global_score_tracker_a, metric = ext_hotpot_eval.eval(pred_dev_a, dev_fullwiki_list)
    global_score_tracker_b, metric = ext_hotpot_eval.eval(pred_dev_b, dev_fullwiki_list)

    print(global_score_tracker_a.keys())
    for key in all_ids:
        scored_item_a = global_score_tracker_a[key]
        scored_item_b = global_score_tracker_b[key]
        # print(scored_item_a.keys())
        if scored_item_a['doc_recall'] != scored_item_b['doc_recall']:
            print(scored_item_a['question'])
            print(scored_item_a['doc_recall'], scored_item_b['doc_recall'])
            print(pred_dev_a['raw_retrieval_set'][key])
            print(pred_dev_b['raw_retrieval_set'][key])
            print(scored_item_a['supporting_facts'])
            print(pred_dev_a['sp_doc'][key])
            print(pred_dev_b['sp_doc'][key])
            break