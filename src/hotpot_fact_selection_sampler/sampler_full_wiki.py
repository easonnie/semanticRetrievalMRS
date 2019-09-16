import collections
import numpy as np

from hotpot_doc_retri.hotpot_doc_retri_v0 import results_multihop_filtering
from evaluation import ext_hotpot_eval
from utils import common
from hotpot_doc_retri.retrieval_utils import RetrievedSet, RetrievedItem
import config
import uuid
from tqdm import tqdm
from wiki_util import wiki_db_tool
import random


def append_baseline_context(doc_results, baseline_data_list):
    data_list = baseline_data_list
    for item in data_list:
        key = item['_id']
        contexts = item['context']
        provided_title = []
        for title, paragraph in contexts:
            provided_title.append(title)

        doc_results['sp_doc'][key] = list(set.union(set(doc_results['sp_doc'][key]), set(provided_title)))

    return doc_results

# Deprecated method.
def append_additional_scored_title(doc_results, top_k, terms_based_results_list):
    # terms_based_results_list = common.load_jsonl(
    #     config.RESULT_PATH / "doc_retri_results/term_based_methods_results/hotpot_tf_idf_dev.jsonl")
    terms_based_results_dict = dict()
    for item in terms_based_results_list:
        terms_based_results_dict[item['qid']] = item

    for item in data_list:
        key = item['_id']
        contexts = item['context']
        provided_title = []
        for title, paragraph in contexts:
            provided_title.append(title)

        top_term_base = sorted(terms_based_results_dict[key]['doc_list'], key=lambda x: x[0], reverse=True)[:top_k]
        top_term_base = [e[1] for e in top_term_base]
        doc_results['sp_doc'][key] = list(set.union(set(doc_results['sp_doc'][key]), set(top_term_base)))

    return doc_results


def results_analysis():
    doc_results = common.load_json(
        # config.PRO_ROOT / "results/doc_retri_results/doc_retrieval_final_v8/hotpot_train_doc_retrieval_v8_before_multihop_filtering.json")
        config.PRO_ROOT / "results/doc_retri_results/doc_retrieval_final_v8/hotpot_dev_doc_retrieval_v8_before_multihop_filtering.json")
    doc_results = results_multihop_filtering(doc_results, multihop_retrieval_top_k=3, strict_mode=True)

    # terms_based_results_list = common.load_jsonl(
    #     config.RESULT_PATH / "doc_retri_results/term_based_methods_results/hotpot_tf_idf_dev.jsonl")

    data_list = common.load_json(config.DEV_FULLWIKI_FILE)
    # data_list = common.load_json(config.TRAIN_FILE)

    append_baseline_context(doc_results, data_list)

    len_list = []
    for rset in doc_results['sp_doc'].values():
        len_list.append(len(rset))

    print("Results with filtering:")

    print(collections.Counter(len_list).most_common(10000))
    print(len(len_list))
    print("Mean:\t", np.mean(len_list))
    print("Std:\t", np.std(len_list))
    print("Max:\t", np.max(len_list))
    print("Min:\t", np.min(len_list))

    ext_hotpot_eval.eval(doc_results, data_list)


def build_full_wiki_document_forward_item(doc_results, data_list, is_training,
                                          db_cursor=None, to_text=True):
    forward_item_list = []
    # Forward item:
        # qid, fid, query, context, doc_t, s_labels.

    print("Build forward items")
    for item in tqdm(data_list):
        qid = item['_id']
        question = item['question']
        selected_doc = doc_results['sp_doc'][qid]

        if is_training:
            gt_doc = list(set([fact[0] for fact in item['supporting_facts']]))
        else:
            gt_doc = []

        all_doc = list(set.union(set(selected_doc), set(gt_doc)))

        fitem_list = []
        for doc in all_doc:
            fitem = dict()
            fitem['qid'] = qid
            fid = str(uuid.uuid4())
            fitem['fid'] = fid

            fitem['query'] = question
            fitem['doc_t'] = doc
            # print(doc)
            # print(doc_results['raw_retrieval_set'][qid])

            if to_text:
                text_item = wiki_db_tool.get_item_by_key(db_cursor, key=doc)
                context = wiki_db_tool.get_first_paragraph_from_clean_text_item(text_item)
            else:
                context = ""

            fitem['context'] = " ".join(context)    # Join the context

            if is_training:
                if doc in gt_doc:
                    fitem['s_labels'] = 'true'
                else:
                    fitem['s_labels'] = 'false'
            else:
                fitem['s_labels'] = 'hidden'

            # print(fitem)
            fitem_list.append(fitem)

        forward_item_list.extend(fitem_list)

    return forward_item_list


def down_sample_neg(fitems, ratio=None):
    if ratio is None:
        return fitems

    pos_count = 0
    neg_count = 0
    other_count = 0

    pos_items = []
    neg_items = []
    other_items = []

    for item in fitems:
        if item['s_labels'] == 'true':
            pos_count += 1
            pos_items.append(item)
        elif item['s_labels'] == 'false':
            neg_count += 1
            neg_items.append(item)
        else:
            other_count += 1
            other_items.append(item)

    if other_count != 0:
        print("Potential Error! We have labels that are not true or false:", other_count)

    print(f"Before Sampling, we have {pos_count}/{neg_count} (pos/neg).")

    random.shuffle(pos_items)
    random.shuffle(neg_items)
    neg_sample_count = int(pos_count / ratio)

    sampled_neg = neg_items[:neg_sample_count]

    print(f"After Sampling, we have {pos_count}/{len(sampled_neg)} (pos/neg).")

    sampled_list = sampled_neg + pos_items
    random.shuffle(sampled_list)

    return sampled_list


def precompute_forward_items_and_cache():
    # 3 places need to switch from dev to train !!!

    is_training = False
    doc_results = common.load_json(
        # config.PRO_ROOT / "results/doc_retri_results/doc_retrieval_final_v8/hotpot_train_doc_retrieval_v8_before_multihop_filtering.json")
        # config.PRO_ROOT / "results/doc_retri_results/doc_retrieval_final_v8/hotpot_dev_doc_retrieval_v8_before_multihop_filtering.json")
        config.PRO_ROOT / "results/doc_retri_results/doc_retrieval_final_v8/hotpot_test_doc_retrieval_v8_before_multihop_filtering.json")
    doc_results = results_multihop_filtering(doc_results, multihop_retrieval_top_k=3, strict_mode=True)

    # db_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_DB)

    t_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)

    # data_list = common.load_json(config.DEV_FULLWIKI_FILE)
    data_list = common.load_json(config.TEST_FULLWIKI_FILE)
    # data_list = common.load_json(config.TRAIN_FILE)
    append_baseline_context(doc_results, data_list)

    fitem_list = build_full_wiki_document_forward_item(doc_results, data_list, is_training, t_db_cursor, True)

    print(len(fitem_list))
    common.save_jsonl(fitem_list, config.PDATA_ROOT / "content_selection_forward" / "hotpot_test_p_level_unlabeled.jsonl")
    # common.save_jsonl(fitem_list, config.PDATA_ROOT / "content_selection_forward" / "hotpot_dev_p_level_unlabeled.jsonl")
    # common.save_jsonl(fitem_list, config.PDATA_ROOT / "content_selection_forward" / "hotpot_train_p_level.jsonl")


if __name__ == '__main__':
    precompute_forward_items_and_cache()
    # pass
    # train_list = common.load_jsonl(config.PDATA_ROOT / "content_selection_forward/hotpot_train_p_level.jsonl")
    # train_list = common.load_jsonl(config.PDATA_ROOT / "content_selection_forward/hotpot_dev_p_level_labeled.jsonl")  # 227983
    # train_list = common.load_jsonl(config.PDATA_ROOT / "content_selection_forward/hotpot_dev_p_level_unlabeled.jsonl")    # 226335
    # train_list = down_sample_neg(train_list, ratio=0.2)
    # print(len(train_list))
