import collections
import itertools
import random

from tqdm import tqdm
import uuid

import config
from build_rindex import raw_text_db
from utils import common
from utils import list_dict_data_tool
import json
import copy

from wiki_util import wiki_db_tool


def top_k_filter_score_list(d_list, top_k):
    new_d_list = []
    for item in d_list:
        new_item = dict()
        new_item['question'] = item['question']
        new_item['qid'] = item['question']
        new_item['answer'] = item['answer']
        new_item['score_list'] = item['score_list'][:top_k]
        new_d_list.append(new_item)

    return new_d_list


def build_p_level_forward_item(p_level_results_dict, distant_gt_dict, data_list, is_training, db_cursor):
    forward_item_list = []

    print("Build forward items")
    for item in tqdm(data_list):
        cur_id = item['question']
        query = item['question']
        selected_paragraph = p_level_results_dict[cur_id]['score_list']

        # gt_paragraph = gt_dict[cur_id]['score_list']
        distant_gt_paragraph = distant_gt_dict[cur_id]['distant_gt_list']
        distant_gt_paragraph_set = {(title, p_num) for (title, p_num), score in distant_gt_paragraph}

        if is_training:  # If is training, we give distant_gt_list
            total_paragraph = []
            added_set = set()
            for (title, p_num), score in selected_paragraph + distant_gt_paragraph:
                if (title, p_num) not in added_set:
                    added_set.add((title, p_num))
                    total_paragraph.append((title, p_num))

        else:  # Else we only have selected_paragraph
            total_paragraph = []
            added_set = set()
            for (title, p_num), score in selected_paragraph:
                if (title, p_num) not in added_set:
                    added_set.add((title, p_num))
                    total_paragraph.append((title, p_num))

        fitem_list = []

        for title, p_num in total_paragraph:
            fitem = dict()
            fitem['qid'] = str(cur_id)  # query id
            fid = str(uuid.uuid4())
            fitem['fid'] = fid  # forward id
            fitem['query'] = query

            p_list = raw_text_db.query_raw_text(db_cursor, title, p_num=p_num)
            assert len(p_list) == 1
            std_title, p_num, p_sentences = p_list[0]
            paragraph_text = ' '.join(json.loads(p_sentences))

            fitem['context'] = paragraph_text
            fitem['element'] = (title, p_num)

            if is_training:
                if (title, p_num) in distant_gt_paragraph_set:
                    fitem['s_labels'] = 'true'
                else:
                    fitem['s_labels'] = 'false'
            else:
                fitem['s_labels'] = 'hidden'
            fitem_list.append(fitem)

        forward_item_list.extend(fitem_list)

    return forward_item_list


# def get_paragraph_forward_pair(tag, ruleterm_doc_results, is_training, debug=False, ignore_non_verifiable=False):
#     pass
# Re-implement this.


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


def get_distant_top_k_ground_truth(gt_dict, d_list, top_k):
    distant_gt_item_list = []
    for item in d_list:
        qid = item['question']
        score_list = item['score_list']
        gt_list = gt_dict[qid]['gt_p_list']
        distant_gt_list = []
        for para_item, score in score_list:
            if para_item in gt_list:
                distant_gt_list.append([para_item, score])

            if len(distant_gt_list) >= top_k:
                break

        new_item = dict()
        new_item['question'] = item['question']
        new_item['qid'] = new_item['question']
        new_item['answer'] = item['answer']
        new_item['distant_gt_list'] = distant_gt_list
        distant_gt_item_list.append(new_item)

    return distant_gt_item_list


def prepare_forward_data(dataset_name, tag, is_training, upstream_top_k=20, distant_gt_top_k=2, down_sample_ratio=None,
                         debug=False):
    if dataset_name == 'webq' and tag == 'test':
        gt_d_list_path = config.OPEN_WEBQ_TEST_GT
    elif dataset_name == 'webq' and tag == 'train':
        gt_d_list_path = config.OPEN_WEBQ_TRAIN_GT
    elif dataset_name == 'curatedtrec' and tag == 'test':
        gt_d_list_path = config.OPEN_CURATEDTERC_TEST_GT
    elif dataset_name == 'curatedtrec' and tag == 'train':
        gt_d_list_path = config.OPEN_CURATEDTERC_TRAIN_GT
    elif dataset_name == 'squad' and tag == 'dev':
        gt_d_list_path = config.OPEN_SQUAD_DEV_GT
    elif dataset_name == 'squad' and tag == 'train':
        gt_d_list_path = config.OPEN_SQUAD_TRAIN_GT
    elif dataset_name == 'wikimovie' and tag == 'test':
        gt_d_list_path = config.OPEN_WIKIM_TEST_GT
    elif dataset_name == 'wikimovie' and tag == 'train':
        gt_d_list_path = config.OPEN_WIKIM_TRAIN_GT
    else:
        raise NotImplemented()

    t_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_RAW_TEXT)
    # debug = False
    # upstream_top_k = 20
    # distant_gt_top_k = 2
    # down_sample_ratio = None

    if dataset_name != 'wikimovie':
        upstream_d_list_before_filter = common.load_jsonl(
            config.PRO_ROOT / f"data/p_{dataset_name}/tf_idf_p_level/{dataset_name}_{tag}_para_tfidf.jsonl")
    else:
        upstream_d_list_before_filter = common.load_jsonl(
            config.PRO_ROOT / f"data/p_{dataset_name}/kwm_p_level/{dataset_name}_{tag}_kwm_tfidf.jsonl")

    if debug:
        upstream_d_list_before_filter = upstream_d_list_before_filter[:50]
    upstream_d_list = top_k_filter_score_list(upstream_d_list_before_filter, top_k=upstream_top_k)

    upstream_d_dict = list_dict_data_tool.list_to_dict(upstream_d_list, 'question')

    gt_d_list = common.load_jsonl(gt_d_list_path)
    gt_d_dict = list_dict_data_tool.list_to_dict(gt_d_list, 'question')
    distant_gt_item_list = get_distant_top_k_ground_truth(gt_d_dict, upstream_d_list_before_filter,
                                                          top_k=distant_gt_top_k)
    distant_gt_item_dict = list_dict_data_tool.list_to_dict(distant_gt_item_list, 'qid')

    fitems_list = build_p_level_forward_item(upstream_d_dict, distant_gt_item_dict, upstream_d_list, is_training,
                                             t_cursor)
    if is_training:
        return down_sample_neg(fitems_list, down_sample_ratio)
    else:
        return down_sample_neg(fitems_list, None)
    # return fitems_list


if __name__ == '__main__':
    debug = False

    # d_list = prepare_forward_data('webq', 'train', True, upstream_top_k=40, distant_gt_top_k=2, down_sample_ratio=0.25,
    #                               debug=debug)
    # print(len(d_list))
    # # prepare_forward_data('webq', 'test', False, upstream_top_k=40, distant_gt_top_k=2, down_sample_ratio=0.2,
    # #                      debug=debug)
    # d_list = prepare_forward_data('curatedtrec', 'train', True, upstream_top_k=40, distant_gt_top_k=2,
    #                               down_sample_ratio=0.25,
    #                               debug=debug)
    # print(len(d_list))
    # prepare_forward_data('curatedtrec', 'test', False, upstream_top_k=40, distant_gt_top_k=2, down_sample_ratio=0.2,
    #                      debug=debug)

    # d_list = prepare_forward_data('squad', 'train', True, upstream_top_k=40, distant_gt_top_k=2, down_sample_ratio=0.25,
    #                               debug=debug)

    d_list = prepare_forward_data('wikimovie', 'test', True, upstream_top_k=40, distant_gt_top_k=2, down_sample_ratio=0.25,
                                  debug=debug)
    print(len(d_list))

    # print(len(d_list))
    # print(len(gt_d_list))
