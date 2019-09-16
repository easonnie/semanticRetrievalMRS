from fever_sampler.sentence_selection_sampler import convert_evidence2scoring_format
from utils import common
from evaluation import fever_scorer
from tqdm import tqdm
from fever_utils import check_sentences
from fever_utils import fever_db
import random
import copy


TITLE_SEP = '.'


def select_sent_with_prob_for_eval(input_file, additional_file, prob_dict_file, tokenized=False, pipeline=False):
    """
    This method select sentences with upstream sentence retrieval.

    :param input_file: This should be the file with 5 sentences selected.
    :return:
    """
    cursor = fever_db.get_cursor()

    if prob_dict_file is None:
        prob_dict_file = dict()

    if isinstance(additional_file, list):
        additional_d_list = additional_file
    else:
        additional_d_list = common.load_jsonl(additional_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    d_list = common.load_jsonl(input_file)

    for item in tqdm(d_list):
        e_list = additional_data_dict[item['id']]['predicted_sentids']
        if not pipeline:
            assert additional_data_dict[item['id']]['label'] == item['label']
            assert additional_data_dict[item['id']]['verifiable'] == item['verifiable']
        assert additional_data_dict[item['id']]['id'] == item['id']

        pred_evidence_list = []
        for i, cur_e in enumerate(e_list):
            doc_id = cur_e.split(fever_scorer.SENT_LINE)[0]
            ln = int(cur_e.split(fever_scorer.SENT_LINE)[1])  # Important changes Bugs: July 21
            pred_evidence_list.append((doc_id, ln))

        pred_evidence = check_sentences.Evidences(pred_evidence_list)

        evidence_text_list = evidence_list_to_text_list(cursor, pred_evidence,
                                                        contain_head=True, id_tokenized=tokenized)

        evidences = sorted(pred_evidence, key=lambda x: (x[0], x[1]))
        item_id = int(item['id'])

        evidence_text_list_with_prob = []
        for text, (doc_id, ln) in zip(evidence_text_list, evidences):
            ssid = (item_id, doc_id, int(ln))
            if ssid not in prob_dict_file:
                # print("Some sentence pair don't have 'prob'.")
                prob = 0.5
            else:
                prob = prob_dict_file[ssid]['prob']
                assert item['claim'] == prob_dict_file[ssid]['claim']

            evidence_text_list_with_prob.append((text, prob))

        if tokenized:
            pass
        else:
            raise NotImplemented("Non tokenized is not implemented.")
            # item['claim'] = ' '.join(easy_tokenize(item['claim']))

        item['evid'] = evidence_text_list_with_prob
        item['predicted_evidence'] = convert_evidence2scoring_format(e_list)
        item['predicted_sentids'] = e_list
        # This change need to be saved.
        # item['predicted_label'] = additional_data_dict[item['id']]['label']

    return d_list


def adv_simi_sample_with_prob_v1_1(input_file, additional_file, prob_dict_file, tokenized=False):
    cursor = fever_db.get_cursor()
    d_list = common.load_jsonl(input_file)

    if prob_dict_file is None:
        prob_dict_file = dict()

    if isinstance(additional_file, list):
        additional_d_list = additional_file
    else:
        additional_d_list = common.load_jsonl(additional_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    sampled_data_list = []
    count = 0

    for item in tqdm(d_list):
        # e_list = check_sentences.check_and_clean_evidence(item)
        sampled_e_list, flags = sample_additional_data_for_item_v1_1(item, additional_data_dict)
        # print(flags)
        for i, (sampled_evidence, flag) in enumerate(zip(sampled_e_list, flags)):
            # Do not copy, might change in the future for error analysis
            # new_item = copy.deepcopy(item)
            new_item = dict()
            # print(new_item['claim'])
            # print(e_list)
            # print(sampled_evidence)
            # print(flag)
            evidence_text_list = evidence_list_to_text_list(
                cursor, sampled_evidence,
                contain_head=True, id_tokenized=tokenized)

            evidences = sorted(sampled_evidence, key=lambda x: (x[0], x[1]))
            item_id = int(item['id'])

            evidence_text_list_with_prob = []
            for text, (doc_id, ln) in zip(evidence_text_list, evidences):
                ssid = (int(item_id), doc_id, int(ln))
                if ssid not in prob_dict_file:
                    count += 1
                    # print("Some sentence pair don't have 'prob'.")
                    prob = 0.5
                else:
                    prob = prob_dict_file[ssid]['prob']
                    assert item['claim'] == prob_dict_file[ssid]['claim']

                evidence_text_list_with_prob.append((text, prob))

            new_item['id'] = str(item['id']) + '#' + str(i)

            if tokenized:
                new_item['claim'] = item['claim']
            else:
                raise NotImplemented("Non tokenized is not implemented.")
                # new_item['claim'] = ' '.join(easy_tokenize(item['claim']))

            new_item['evid'] = evidence_text_list_with_prob

            new_item['verifiable'] = item['verifiable']
            new_item['label'] = item['label']

            # print("C:", new_item['claim'])
            # print("E:", new_item['evid'])
            # print("L:", new_item['label'])
            # print()
            sampled_data_list.append(new_item)

    cursor.close()

    print(count)
    return sampled_data_list


def evidence_list_to_text_list(cursor, evidences, contain_head=True, id_tokenized=False):
    # id_tokenized is a deprecated argument.
    # One evidence one text and len(evidences) == len(text_list)
    current_evidence_text_list = []
    evidences = sorted(evidences, key=lambda x: (x[0], x[1]))

    cur_head = 'DO NOT INCLUDE THIS FLAG'

    for doc_id, line_num in evidences:

        _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)

        cur_text = ""

        if contain_head and cur_head != doc_id:
            cur_head = doc_id

            doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')

            # if not id_tokenized:
            #     raise NotImplemented("Non tokenized is not implemented.")
            #     # doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
            #     # t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))
            # else:
            #     t_doc_id_natural_format = common.doc_id_to_tokenized_text(doc_id)

            if line_num != 0:
                cur_text = f"{doc_id_natural_format} {TITLE_SEP} "

        # Important change move one line below: July 16
        # current_evidence_text.append(e_text)
        cur_text = cur_text + e_text

        current_evidence_text_list.append(cur_text)

    assert len(evidences) == len(current_evidence_text_list)
    return current_evidence_text_list


def sample_additional_data_for_item_v1_1(item, additional_data_dictionary):
    res_sentids_list = []
    flags = []

    if item['verifiable'] == "VERIFIABLE":
        assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
        e_list = check_sentences.check_and_clean_evidence(item)
        current_id = item['id']
        assert current_id in additional_data_dictionary
        additional_data = additional_data_dictionary[current_id]['predicted_sentids']
        # additional_data_with_score = additional_data_dictionary[current_id]['scored_sentids']

        # print(len(additional_data))

        for evidences in e_list:
            # print(evidences)
            new_evidences = copy.deepcopy(evidences)
            n_e = len(evidences)
            if n_e < 5:
                current_sample_num = random.randint(0, 5 - n_e)
                random.shuffle(additional_data)
                for sampled_e in additional_data[:current_sample_num]:
                    doc_ids = sampled_e.split(fever_scorer.SENT_LINE)[0]
                    ln = int(sampled_e.split(fever_scorer.SENT_LINE)[1])
                    new_evidences.add_sent(doc_ids, ln)

            if new_evidences != evidences:
                flag = f"verifiable.non_eq.{len(new_evidences) - len(evidences)}"
                flags.append(flag)
                pass
            else:
                flag = "verifiable.eq.0"
                flags.append(flag)
                pass
            res_sentids_list.append(new_evidences)

        assert len(res_sentids_list) == len(e_list)

    elif item['verifiable'] == "NOT VERIFIABLE":
        assert item['label'] == 'NOT ENOUGH INFO'

        e_list = check_sentences.check_and_clean_evidence(item)
        current_id = item['id']

        additional_data = additional_data_dictionary[current_id]['predicted_sentids']
        prioritized_additional_evidence_list = additional_data_dictionary[current_id]['scored_sentids']

        #  cur_predicted_sentids.append((sent_i['sid'], sent_i['score'], sent_i['prob']))
        certain_k = 2
        prioritized_additional_evidence_list = sorted(prioritized_additional_evidence_list, key=lambda x: -x[1])
        top_two_sent = [sid for sid, _, _ in prioritized_additional_evidence_list[:certain_k]]

        random.shuffle(additional_data)
        current_sample_num = random.randint(0, 2)
        raw_evidences_list = []

        # Debug
        # print(prioritized_additional_evidence_list)
        # print(top_two_sent)

        for sampled_e in top_two_sent + additional_data[:current_sample_num]:
            doc_ids = sampled_e.split(fever_scorer.SENT_LINE)[0]
            ln = int(sampled_e.split(fever_scorer.SENT_LINE)[1])
            raw_evidences_list.append((doc_ids, ln))
        new_evidences = check_sentences.Evidences(raw_evidences_list)

        if len(new_evidences) == 0:
            flag = f"verifiable.eq.0"
            flags.append(flag)
            pass
        else:
            flag = f"not_verifiable.non_eq.{len(new_evidences)}"
            flags.append(flag)

        assert all(len(e) == 0 for e in e_list)
        res_sentids_list.append(new_evidences)
        assert len(res_sentids_list) == 1

        # Debug
        # print(res_sentids_list)

    assert len(res_sentids_list) == len(flags)

    return res_sentids_list, flags