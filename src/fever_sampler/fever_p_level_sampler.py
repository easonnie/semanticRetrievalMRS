import collections
import itertools
import random

from tqdm import tqdm
import uuid

import config
import fever_utils.check_sentences
from evaluation import fever_scorer
from fever_utils import fever_db
from utils import common
from utils import list_dict_data_tool


def build_full_wiki_document_forward_item(doc_results, data_list, is_training,
                                          db_cursor=None, ignore_non_verifiable=False):
    forward_item_list = []
    # Forward item:
        # qid, fid, query, context, doc_t, s_labels.

    print("Build forward items")
    for item in tqdm(data_list):
        cur_id = int(item['id'])
        query = item['claim']
        selected_doc = doc_results[cur_id]['predicted_docids']
        if 'verifiable' in item.keys():
            verifiable = item['verifiable'] == "VERIFIABLE"
        else:
            verifiable = None

        if not verifiable and is_training and ignore_non_verifiable:
            continue

        all_id_list = []

        if is_training:
            e_list = fever_utils.check_sentences.check_and_clean_evidence(item)
            all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))

            # Here, we retrieval all the evidence sentence
            gt_doc_id = []
            gt_doc_set = set()

            for doc_id, ln in all_evidence_set:
                gt_doc_set.add(doc_id)

            for doc_id in gt_doc_set:
                gt_doc_id.append(doc_id)
                all_id_list.append(doc_id)
        else:
            gt_doc_id = []

        for doc_id in selected_doc:
            if doc_id not in all_id_list:
                all_id_list.append(doc_id)

        # assert len(all_texts_list) == len(all_id_list)
        fitem_list = []

        for doc_id in all_id_list:
            fitem = dict()
            fitem['qid'] = str(cur_id)  # query id
            fid = str(uuid.uuid4())
            fitem['fid'] = fid          # forward id

            fitem['query'] = query

            cur_text = get_paragraph_text(doc_id, db_cursor)

            fitem['context'] = ' '.join(cur_text)
            fitem['element'] = doc_id   # the element is just the doc_id

            if is_training:
                if doc_id in gt_doc_id:
                    fitem['s_labels'] = 'true'
                else:
                    fitem['s_labels'] = 'false'
            else:
                fitem['s_labels'] = 'hidden'

            fitem_list.append(fitem)

        forward_item_list.extend(fitem_list)

    return forward_item_list


def get_paragraph_text(doc_id, db_cursor):
    text_list, _ = fever_db.get_all_sent_by_doc_id(db_cursor, doc_id)
    all_text = []
    for sent_text in text_list:
        natural_formatted_text = fever_db.convert_brc(sent_text)
        all_text.append(natural_formatted_text)

    return all_text


def get_paragraph_forward_pair(tag, ruleterm_doc_results, is_training, debug=False, ignore_non_verifiable=False):
    if tag == 'dev':
        d_list = common.load_jsonl(config.FEVER_DEV)
    elif tag == 'train':
        d_list = common.load_jsonl(config.FEVER_TRAIN)
    elif tag == 'test':
        d_list = common.load_jsonl(config.FEVER_TEST)
    else:
        raise ValueError(f"Tag:{tag} not supported.")

    if debug:
        d_list = d_list[:100]
        ruleterm_doc_results = ruleterm_doc_results[:100]

    ruleterm_doc_results_dict = list_dict_data_tool.list_to_dict(ruleterm_doc_results, 'id')
    db_cursor = fever_db.get_cursor()
    fitems = build_full_wiki_document_forward_item(ruleterm_doc_results_dict, d_list, is_training, db_cursor,
                                                   ignore_non_verifiable)

    return fitems


def down_sample_neg(fitems, ratio=None):
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

    if ratio is None:
        return fitems

    random.shuffle(pos_items)
    random.shuffle(neg_items)
    neg_sample_count = int(pos_count / ratio)

    sampled_neg = neg_items[:neg_sample_count]

    print(f"After Sampling, we have {pos_count}/{len(sampled_neg)} (pos/neg).")

    sampled_list = sampled_neg + pos_items
    random.shuffle(sampled_list)

    return sampled_list


if __name__ == '__main__':
    d_list = common.load_jsonl(config.FEVER_TRAIN)
    ruleterm_doc_results = common.load_jsonl(config.PRO_ROOT / "results/doc_retri_results/fever_results/merged_doc_results/m_doc_train.jsonl")
    mode = {'standard': False, 'check_doc_id_correct': True}
    fever_scorer.fever_score_analysis(ruleterm_doc_results, d_list, mode=mode, max_evidence=None)

    # fitems = get_paragraph_forward_pair('dev', ruleterm_doc_results, is_training=False, debug=False,
    #                                     ignore_non_verifiable=False)
    # down_sample_neg(fitems, None)
