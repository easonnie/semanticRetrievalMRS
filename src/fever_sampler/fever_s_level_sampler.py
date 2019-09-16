import itertools
import uuid

import fever_utils.check_sentences
from fever_sampler.fever_p_level_sampler import down_sample_neg
from fever_sampler.fever_sampler_utils import select_top_k_and_to_results_dict
from fever_utils import fever_db
from utils import common, list_dict_data_tool
from tqdm import tqdm
from evaluation import fever_scorer
from typing import Dict, List, Tuple
import config
import copy


def build_full_wiki_sentence_forward_item(doc_results, data_list, is_training,
                                          db_cursor=None, ignore_non_verifiable=False):
    forward_item_list = []

    print("Build forward items")
    for item in tqdm(data_list):
        cur_id = int(item['id'])
        query = item['claim']
        selected_doc = doc_results['predicted_docids'][cur_id]
        if 'verifiable' in item.keys():
            verifiable = item['verifiable'] == "VERIFIABLE"
        else:
            verifiable = None

        if not verifiable and is_training and ignore_non_verifiable:
            continue

        all_id_list: List[List[str, int]] = []
        all_texts_list = []

        if is_training:
            e_list = fever_utils.check_sentences.check_and_clean_evidence(item)
            all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))

            # Here, we retrieval all the evidence sentence
            # gt_evidence_texts = []
            gt_evidence_id = []
            for doc_id, ln in all_evidence_set:
                _, text, _ = fever_db.get_evidence(db_cursor, doc_id, ln)

                # gt_evidence_texts.append(text)
                all_texts_list.append(text)

                gt_evidence_id.append([doc_id, ln])
                all_id_list.append([doc_id, ln])
        else:
            # gt_evidence_texts = []
            gt_evidence_id = []

        for doc_id in selected_doc:
            cur_text_list, cur_id_list = fever_db.get_all_sent_by_doc_id(db_cursor, doc_id, with_h_links=False)
            # Merging to data list and removing duplicate
            for i in range(len(cur_text_list)):
                sid = cur_id_list[i]
                doc_id, ln = sid.split('(-.-)')[0], int(sid.split('(-.-)')[1])
                if [doc_id, ln] not in all_id_list:
                    all_texts_list.append(cur_text_list[i])
                    all_id_list.append([doc_id, ln])

        assert len(all_texts_list) == len(all_id_list)
        fitem_list = []

        for text, sid in zip(all_texts_list, all_id_list):
            fitem = dict()
            fitem['qid'] = str(cur_id)
            fid = str(uuid.uuid4())
            fitem['fid'] = fid
            fitem['element'] = sid

            fitem['query'] = query

            cur_text = convert_to_formatted_sent(text, sid, contain_head=True)
            fitem['context'] = cur_text

            if is_training:
                if sid in gt_evidence_id:
                    fitem['s_labels'] = 'true'
                else:
                    fitem['s_labels'] = 'false'
            else:
                fitem['s_labels'] = 'hidden'

            fitem_list.append(fitem)

        forward_item_list.extend(fitem_list)

    return forward_item_list


def convert_to_formatted_sent(text, sid, contain_head=False):
    sent = text
    doc_id, ln = sid
    doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
    if contain_head and ln != 0 and doc_id_natural_format.lower() not in sent.lower():
        cur_sent = f"{doc_id_natural_format} . " + sent
    else:
        cur_sent = sent
    cur_sent = fever_db.convert_brc(cur_sent)

    return cur_sent


def get_sentence_forward_pair(tag, ruleterm_doc_results, is_training,
                              debug=False, ignore_non_verifiable=False,
                              top_k=5, filter_value=0.005):
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

    # ruleterm_doc_results_dict = list_dict_data_tool.list_to_dict(ruleterm_doc_results, 'id')
    d_o_dict = list_dict_data_tool.list_to_dict(d_list, 'id')
    copied_d_o_dict = copy.deepcopy(d_o_dict)
    # copied_d_list = copy.deepcopy(d_list)
    list_dict_data_tool.append_subfield_from_list_to_dict(ruleterm_doc_results, copied_d_o_dict,
                                                          'qid', 'fid', check=True)

    cur_results_dict_filtered = select_top_k_and_to_results_dict(copied_d_o_dict,
                                                                 score_field_name='prob',
                                                                 top_k=top_k, filter_value=filter_value)

    db_cursor = fever_db.get_cursor()
    fitems = build_full_wiki_sentence_forward_item(cur_results_dict_filtered, d_list, is_training, db_cursor,
                                                   ignore_non_verifiable)

    return fitems


if __name__ == '__main__':
    ruleterm_doc_results = common.load_jsonl(
        config.PRO_ROOT / "data/p_fever/fever_paragraph_level/04-22-15:05:45_fever_v0_plevel_retri_(ignore_non_verifiable:True)/"
                          "i(5000)|e(0)|v02_ofever(0.8947894789478947)|v05_ofever(0.8555355535553555)|seed(12)/fever_p_level_dev_results.jsonl")
    fitem = get_sentence_forward_pair('dev', ruleterm_doc_results, is_training=True, debug=False,
                                      ignore_non_verifiable=True)
    down_sample_neg(fitem, None)
