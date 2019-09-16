import copy
import random

from hotpot_fact_selection_sampler.sampler_utils import select_top_k_and_to_results_dict
from utils import common, list_dict_data_tool
from tqdm import tqdm
import uuid
from wiki_util import wiki_db_tool
import config
from typing import Dict, Tuple


def sanity_check(doc_results, data_list, is_training,
                 db_cursor=None, append_head=True):
    forward_item_list = []
    e_count = 0
    t_count = 0

    print("Build forward items")
    for item in tqdm(data_list):
        qid = item['_id']
        question = item['question']
        selected_doc = doc_results['sp_doc'][qid]

        # sentid2text_dict = dict()
        sentid2text_dict: Dict[Tuple[str, int], str] = dict()
        context_dict = dict()
        for doc, sent_list in item['context']:
            for i, sent in enumerate(sent_list):
                context_dict[(doc, i)] = sent

        if is_training:
            gt_fact = item['supporting_facts']
            gt_doc = list(set([fact[0] for fact in item['supporting_facts']]))
        else:
            gt_fact = []
            gt_doc = []

        selected_fact = []
        for doc in gt_doc:
            text_item = wiki_db_tool.get_item_by_key(db_cursor, key=doc)
            context = wiki_db_tool.get_first_paragraph_from_clean_text_item(text_item, flatten_to_paragraph=False,
                                                                            skip_first=True)
            for i, sentence_token in enumerate(context):
                sentence_text = ' '.join(sentence_token)
                if len(sentence_text) != 0:
                    selected_fact.append([doc, i])
                    sentid2text_dict[(doc, i)] = sentence_text

        for doc in selected_doc:
            if doc in gt_doc:
                continue
            text_item = wiki_db_tool.get_item_by_key(db_cursor, key=doc)
            context = wiki_db_tool.get_first_paragraph_from_clean_text_item(text_item, flatten_to_paragraph=False,
                                                                            skip_first=True)
            for i, sentence_token in enumerate(context):
                sentence_text = ' '.join(sentence_token)
                if len(sentence_text) != 0:
                    selected_fact.append([doc, i])
                    sentid2text_dict[(doc, i)] = sentence_text

        all_fact = []
        for fact in selected_fact + gt_fact:
            if fact not in all_fact:
                all_fact.append(fact)

        for (doc, i) in context_dict:
            if [doc, i] in gt_fact:
                if (doc, i) not in sentid2text_dict:
                    e_count += 1
                else:
                    b_sent = context_dict[(doc, i)]
                    b_sent = b_sent.replace(" ", "")
                    db_sent = sentid2text_dict[(doc, i)]
                    db_sent = db_sent.replace(" ", "")

                    # print(b_sent)
                    # print(db_sent)

                    if b_sent != db_sent:
                        e_count += 1

                t_count += 1

        # for (doc, i), sent in sentid2text_dict.items():
        #     if (doc, i) in context_dict:
        #         b_sent = context_dict[(doc, i)]
        #         b_sent = b_sent.replace(" ", "")
        #         db_sent = sentid2text_dict[(doc, i)]
        #         db_sent = db_sent.replace(" ", "")

        # print(b_sent)
        # print(db_sent)

        # if b_sent != db_sent:
        #     e_count += 1
        # t_count += 1

    print(e_count, t_count, e_count / t_count)

    # print(sentid2text_dict[(doc, i)])

    # fitem_list = []
    #
    # for fact in all_fact:
    #     fitem = dict()
    #     fitem['qid'] = qid
    #     fid = str(uuid.uuid4())
    #     fitem['fid'] = fid
    #
    #     fitem['query'] = question
    #     fitem['element'] = fact
    #     # print(doc)
    #     # print(doc_results['raw_retrieval_set'][qid])
    #     if (fact[0], fact[1]) not in sentid2text_dict:
    #         print()  # 'Immanuel Lutheran School (Perryville, Missouri)'
    #     context = sentid2text_dict[(fact[0], fact[1])]
    #
    #     if append_head and fact[1] != 0:
    #         context = fact[0] + ' . ' + context
    #
    #     fitem['context'] = context
    #
    #     if is_training:
    #         if fact in gt_fact:
    #             fitem['s_labels'] = 'true'
    #         else:
    #             fitem['s_labels'] = 'false'
    #     else:
    #         fitem['s_labels'] = 'hidden'
    #
    #     fitem_list.append(fitem)
    #
    # forward_item_list.extend(fitem_list)

    return forward_item_list


def build_sentence_forward_item(doc_results, data_list, is_training,
                                db_cursor=None, append_head=True):
    forward_item_list = []

    print("Build forward items")
    for item in tqdm(data_list):
        qid = item['_id']
        question = item['question']
        selected_doc = doc_results['sp_doc'][qid]

        # sentid2text_dict = dict()
        sentid2text_dict: Dict[Tuple[str, int], str] = dict()

        if is_training:
            gt_fact = item['supporting_facts']
            gt_doc = list(set([fact[0] for fact in item['supporting_facts']]))
        else:
            gt_fact = []
            gt_doc = []

        selected_fact = []
        for doc in gt_doc:
            text_item = wiki_db_tool.get_item_by_key(db_cursor, key=doc)
            context = wiki_db_tool.get_first_paragraph_from_clean_text_item(text_item, flatten_to_paragraph=False,
                                                                            skip_first=True)
            for i, sentence_token in enumerate(context):
                sentence_text = ' '.join(sentence_token)
                if len(sentence_text) != 0:
                    selected_fact.append([doc, i])
                    sentid2text_dict[(doc, i)] = sentence_text

        for doc in selected_doc:
            if doc in gt_doc:
                continue
            text_item = wiki_db_tool.get_item_by_key(db_cursor, key=doc)
            context = wiki_db_tool.get_first_paragraph_from_clean_text_item(text_item, flatten_to_paragraph=False,
                                                                            skip_first=True)
            for i, sentence_token in enumerate(context):
                sentence_text = ' '.join(sentence_token)
                if len(sentence_text) != 0:
                    selected_fact.append([doc, i])
                    sentid2text_dict[(doc, i)] = sentence_text

        all_fact = []
        for fact in selected_fact + gt_fact:
            if fact not in all_fact:
                all_fact.append(fact)

        fitem_list = []

        for fact in all_fact:
            fitem = dict()
            fitem['qid'] = qid
            fid = str(uuid.uuid4())
            fitem['fid'] = fid

            fitem['query'] = question
            fitem['element'] = fact
            # print(doc)
            # print(doc_results['raw_retrieval_set'][qid])
            if (fact[0], fact[1]) not in sentid2text_dict:
                print(f"Potential Error: {(fact[0], fact[1])} not exists in DB.")  # 'Immanuel Lutheran School (Perryville, Missouri)'
                continue

            context = sentid2text_dict[(fact[0], fact[1])]

            if append_head and fact[1] != 0:
                context = fact[0] + ' . ' + context

            fitem['context'] = context

            if is_training:
                if fact in gt_fact:
                    fitem['s_labels'] = 'true'
                else:
                    fitem['s_labels'] = 'false'
            else:
                fitem['s_labels'] = 'hidden'

            fitem_list.append(fitem)

        forward_item_list.extend(fitem_list)

    return forward_item_list


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


def get_sentence_pair(top_k, d_list, p_level_results_list, is_training, debug_mode=False):
    #
    t_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    #
    # dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    # dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    dev_list = d_list

    # cur_dev_eval_results_list = common.load_jsonl(
    #     config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_document_level/2019_4_17/dev_p_level_bert_v1_results.jsonl")
    cur_dev_eval_results_list = p_level_results_list

    if debug_mode:
        dev_list = dev_list[:100]
        id_set = set([item['_id'] for item in dev_list])
        cur_dev_eval_results_list = [item for item in p_level_results_list if item['qid'] in id_set]

    dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, '_id')

    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    list_dict_data_tool.append_subfield_from_list_to_dict(cur_dev_eval_results_list, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)
    cur_results_dict_top2 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=top_k, filter_value=None)
    # print(cur_results_dict_top2)
    fitems = build_sentence_forward_item(cur_results_dict_top2, dev_list, is_training=is_training,
                                         db_cursor=t_db_cursor)

    return fitems


def get_dev_sentence_pair(top_k, is_training, debug=False, cur_dev_eval_results_list=None):
    dev_list = common.load_json(config.DEV_FULLWIKI_FILE)

    if cur_dev_eval_results_list is None:
        cur_dev_eval_results_list = common.load_jsonl(
            config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/"
                              "i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/dev_p_level_bert_v1_results.jsonl")

    if debug:
        dev_list = dev_list[:100]
        id_set = set([item['_id'] for item in dev_list])
        cur_dev_eval_results_list = [item for item in cur_dev_eval_results_list if item['qid'] in id_set]

    return get_sentence_pair(top_k, dev_list, cur_dev_eval_results_list, is_training)


def get_train_sentence_pair(top_k, is_training, debug=False, cur_train_eval_results_list=None):
    train_list = common.load_json(config.TRAIN_FILE)

    if cur_train_eval_results_list is None:
        cur_train_eval_results_list = common.load_jsonl(
            config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/"
                              "i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/train_p_level_bert_v1_results.jsonl")

    if debug:
        train_list = train_list[:100]
        id_set = set([item['_id'] for item in train_list])
        cur_train_eval_results_list = [item for item in cur_train_eval_results_list if item['qid'] in id_set]

    return get_sentence_pair(top_k, train_list, cur_train_eval_results_list, is_training)


if __name__ == '__main__':
    # # fitems = sanity_check(cur_results_dict_top2, dev_list, is_training=True, db_cursor=t_db_cursor)
    # fitems = get_dev_sentence_pair(3, False)

    # dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    # print(dev_list[100]['_id'])

    fitems = get_dev_sentence_pair(3, True, debug=True)
    # fitems = get_train_sentence_pair(3, True, debug=True)

    down_sample_neg(fitems)
    #
    print(fitems[0])
    # print(len(fitems))

    # t_item = wiki_db_tool.get_item_by_key(t_db_cursor, key='Immanuel Lutheran School (Perryville, Missouri)')
    # t_item = wiki_db_tool.get_item_by_key(t_db_cursor, key='Gajabrishta')
    # t_item = wiki_db_tool.get_item_by_key(t_db_cursor, key='Bill Pollack')
    # context = wiki_db_tool.get_first_paragraph_from_clean_text_item(t_item, flatten_to_paragraph=False, skip_first=True)
    # print(context)
