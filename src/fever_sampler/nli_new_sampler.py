import uuid
from collections import Counter

from evaluation import fever_scorer
# from fever_models.sentence_selection.bert_v1 import select_top_k_and_to_results_dict, set_gt_nli_label
from fever_sampler.fever_sampler_utils import select_top_k_and_to_results_dict
from fever_utils import check_sentences, fever_db
from utils import common, list_dict_data_tool
import copy
import config
import random
from tqdm import tqdm

TITLE_SEP = '.'


def evidences_to_text(evidences, db_cursor, contain_head=True):
    evidences = sorted(evidences, key=lambda x: (x[0], x[1]))
    cur_head = 'DO NOT INCLUDE THIS FLAG'
    sentences = []
    for doc_id, line_num in evidences:
        _, e_text, _ = fever_db.get_evidence(db_cursor, doc_id, line_num)

        cur_text = ""
        e_text = fever_db.convert_brc(e_text)

        if contain_head and cur_head != doc_id:
            cur_head = doc_id
            doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')

            if line_num != 0:
                cur_text = f"{doc_id_natural_format} {TITLE_SEP} "

        cur_text = cur_text + e_text
        sentences.append(cur_text)

    return sentences


def build_nli_forward_item(data_list, is_training, certain_k=2, db_cursor=None, contain_head=True):
    forward_item_list = []
    flags = []

    print("Build forward items")
    for item in tqdm(data_list):
        cur_id = str(item['id'])
        query = item['claim']
        scored_sent = item['selected_scored_results']

        item_verifiable = item['verifiable'] if 'verifiable' in item else None
        item_label = item['label'] if 'label' in item else None

        # print(cur_id)
        # print(query)
        # print(scored_sent)

        if is_training:  # training mode
            forward_evidences_list = []

            if item['verifiable'] == "VERIFIABLE":
                assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
                e_list = check_sentences.check_and_clean_evidence(item)

                for evidences in e_list:
                    # print(evidences)
                    new_evidences = copy.deepcopy(evidences)
                    n_e = len(evidences)
                    if n_e < 5:
                        current_sample_num = random.randint(0, 5 - n_e)
                        random.shuffle(scored_sent)
                        for score, (doc_id, ln) in scored_sent[:current_sample_num]:
                            # doc_ids = sampled_e.split(fever_scorer.SENT_LINE)[0]
                            # ln = int(sampled_e.split(fever_scorer.SENT_LINE)[1])
                            new_evidences.add_sent(doc_id, ln)

                    if new_evidences != evidences:
                        flag = f"verifiable.non_eq.{len(new_evidences) - len(evidences)}"
                        flags.append(flag)
                    else:
                        flag = "verifiable.eq.0"
                        flags.append(flag)
                    forward_evidences_list.append(new_evidences)
                assert len(forward_evidences_list) == len(e_list)

            elif item['verifiable'] == "NOT VERIFIABLE":
                assert item['label'] == 'NOT ENOUGH INFO'

                e_list = check_sentences.check_and_clean_evidence(item)

                prioritized_additional_evidence_list = sorted(scored_sent, key=lambda x: -x[0])
                # print("Pro:", prioritized_additional_evidence_list)
                top_two_sent = prioritized_additional_evidence_list[:certain_k]

                random.shuffle(scored_sent)
                current_sample_num = random.randint(0, 2)
                raw_evidences_list = []

                for score, (doc_id, ln) in top_two_sent + scored_sent[:current_sample_num]:
                    raw_evidences_list.append((doc_id, ln))
                new_evidences = check_sentences.Evidences(raw_evidences_list)

                if len(new_evidences) == 0:
                    flag = f"verifiable.eq.0"
                    flags.append(flag)
                else:
                    flag = f"not_verifiable.non_eq.{len(new_evidences)}"
                    flags.append(flag)

                assert all(len(e) == 0 for e in e_list)
                forward_evidences_list.append(new_evidences)
                assert len(forward_evidences_list) == 1

            # handle result_sentids_list
            for forward_evidences in forward_evidences_list:
                forward_evidences_text = evidences_to_text(forward_evidences, db_cursor, contain_head=contain_head)
                # print(forward_evidences)
                # print(forward_evidences_text)

                fitem = dict()
                fitem['cid'] = str(cur_id)
                fid = str(uuid.uuid4())
                fitem['fid'] = fid

                fitem['query'] = query
                cur_text = forward_evidences_text
                fitem['context'] = ' '.join(cur_text)

                fitem['label'] = item_label
                fitem['verifiable'] = item_verifiable

                forward_item_list.append(fitem)

        else:  # non-training mode
            pred_evidence_list = []
            for score, (doc_id, ln) in scored_sent:
                pred_evidence_list.append((doc_id, ln))

            forward_evidences = check_sentences.Evidences(pred_evidence_list)

            forward_evidences_text = evidences_to_text(forward_evidences, db_cursor, contain_head=contain_head)
            # print(forward_evidences)
            # print(forward_evidences_text)

            fitem = dict()
            fitem['cid'] = str(cur_id)
            fid = str(uuid.uuid4())
            fitem['fid'] = fid

            fitem['query'] = query
            cur_text = forward_evidences_text
            fitem['context'] = ' '.join(cur_text)

            fitem['label'] = 'hidden'
            fitem['verifiable'] = 'hidden'

            forward_item_list.append(fitem)

    return forward_item_list


def get_nli_pair(tag, is_training, sent_level_results_list,
                 debug=None, sent_top_k=5, sent_filter_value=0.05):
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
        # sent_dict = list_dict_data_tool.list_to_dict(sent_level_results_list):

    d_dict = list_dict_data_tool.list_to_dict(d_list, 'id')

    if debug:
        id_set = set([item['id'] for item in d_list])
        new_sent_list = []
        for item in sent_level_results_list:
            if item["qid"] in id_set:
                new_sent_list.append(item)
        sent_level_results_list = new_sent_list

    list_dict_data_tool.append_subfield_from_list_to_dict(sent_level_results_list, d_dict,
                                                          'qid', 'fid', check=True)

    filltered_sent_dict = select_top_k_and_to_results_dict(d_dict,
                                                           score_field_name='prob',
                                                           top_k=sent_top_k, filter_value=sent_filter_value,
                                                           result_field='predicted_evidence')

    list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(d_list, filltered_sent_dict,
                                                                   'id',
                                                                   ['predicted_evidence', 'selected_scored_results'])

    fever_db_cursor = fever_db.get_cursor(config.FEVER_DB)
    forward_items = build_nli_forward_item(d_list, is_training=is_training, db_cursor=fever_db_cursor)

    return forward_items, d_list


if __name__ == '__main__':
    sent_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_fever/fever_sentence_level/04-24-00-11-19_fever_v0_slevel_retri_(ignore_non_verifiable-True)/fever_s_level_dev_results.jsonl")

    pairs, _ = get_nli_pair('dev', False, sent_list, False)

    lengths = []
    for item in pairs:
        lengths.append(len(item['context'].split(' ')))

    l_counter = Counter(lengths)
    print(l_counter.most_common())
    t_c = 0
    for value, count in l_counter.items():
        if value > 250:
            t_c += count
    print(t_c)
    # print(pairs[0])
    # print(pairs[1])
    # print(pairs[2])