# 2019-04-12
# This is a new sampler than its brother "sentence selection sampler".
# The reason to write this new one is to
import itertools
import random
from tqdm import tqdm
import uuid
import fever_utils.check_sentences
from fever_utils import fever_db
from utils import common
import config
from utils import list_dict_data_tool


SMILE_SEPARATOR = '(-.-)'


def build_full_wiki_document_forward_item(doc_results, data_list, is_training,
                                          db_cursor=None):
    forward_item_list = []
    # Forward item:
        # qid, fid, query, context, doc_t, s_labels.

    print("Build forward items")
    for item in tqdm(data_list):
        cur_id = int(item['id'])
        query = item['claim']
        selected_doc = doc_results[cur_id]['predicted_docids']

        # selected_evidence_list = []
        #
        # for doc_id in selected_doc:
        #     cur_r_list, cur_id_list = fever_db.get_all_sent_by_doc_id(db_cursor, doc_id, with_h_links=False)
        all_id_list = []
        all_texts_list = []

        if is_training:
            e_list = fever_utils.check_sentences.check_and_clean_evidence(item)
            all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))

            # Here, we retrieval all the evidence sentence
            gt_evidence_texts = []
            gt_evidence_id = []
            for doc_id, ln in all_evidence_set:
                _, text, _ = fever_db.get_evidence(db_cursor, doc_id, ln)

                gt_evidence_texts.append(text)
                all_texts_list.append(text)

                gt_evidence_id.append(doc_id + '(-.-)' + str(ln))
                all_id_list.append(doc_id + '(-.-)' + str(ln))
        else:
            gt_evidence_texts = []
            gt_evidence_id = []

        for doc_id in selected_doc:
            cur_text_list, cur_id_list = fever_db.get_all_sent_by_doc_id(db_cursor, doc_id, with_h_links=False)
            # Merging to data list and removing duplicate
            for i in range(len(cur_text_list)):
                if cur_id_list[i] in all_id_list:
                    continue
                else:
                    all_texts_list.append(cur_text_list[i])
                    all_id_list.append(cur_id_list[i])

        assert len(all_texts_list) == len(all_id_list)
        fitem_list = []

        for text, sid in zip(all_texts_list, all_id_list):
            fitem = dict()
            fitem['cid'] = str(cur_id)
            fitem['sid'] = str(sid)
            fid = str(uuid.uuid4())
            fitem['fid'] = fid

            fitem['query'] = query

            cur_text = convert_to_formatted_sent(text, sid, contain_head=False)
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
    doc_id, ln = sid.split('(-.-)')[0], int(sid.split('(-.-)')[1])
    doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
    if contain_head and ln != 0 and doc_id_natural_format.lower() not in sent.lower():
        cur_sent = f"{doc_id_natural_format} TTT " + sent
    else:
        cur_sent = sent
    cur_sent = fever_db.convert_brc(cur_sent)

    return cur_sent


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


if __name__ == '__main__':
    d_list = common.load_jsonl(config.FEVER_DEV)
    doc_results = common.load_jsonl(config.RESULT_PATH / "doc_retri_results/fever_results/merged_doc_results/m_doc_dev.jsonl")
    doc_results_dict = list_dict_data_tool.list_to_dict(doc_results, 'id')
    fever_db_cursor = fever_db.get_cursor(config.FEVER_DB)
    forward_items = build_full_wiki_document_forward_item(doc_results_dict, d_list, is_training=False,
                                                          db_cursor=fever_db_cursor)
    # print(forward_items)

    # for item in forward_items:
        # if item['s_labels'] == 'true':
        # print(item['query'], item['context'], item['sid'], item['cid'], item['fid'], item['s_labels'])

    print(len(forward_items))
    # down_sample_neg(forward_items, ratio=0.2)