from utils import common
from tqdm import tqdm
from evaluation import fever_scorer
from typing import Dict


SMILE_SEP='(-.-)'


def threshold_sampler_insure_unique(org_data_file, full_sent_list, prob_threshold=0.5, logist_threshold=None, top_n=5):
    """
    Providing samples to the Training set by a probability threshold on the upstream selected sentences.
    """
    d_list = common.load_jsonl(org_data_file)
    augmented_dict: Dict[int, Dict[str, Dict]] = dict()
    print("Build selected sentences file:", len(full_sent_list))
    for sent_item in tqdm(full_sent_list):
        selection_id = sent_item['selection_id']  # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        remain_str = selection_id.split('<##>')[1]
        # doc_id = remain_str.split(c_scorer.SENT_LINE)[0]
        # ln = int(remain_str.split(c_scorer.SENT_LINE)[1])
        if org_id in augmented_dict:
            if remain_str not in augmented_dict[org_id]:
                augmented_dict[org_id][remain_str] = sent_item
            else:
                print("Exist")
        else:
            augmented_dict[org_id] = {remain_str: sent_item}

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            # print("Potential error?")
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])].values()
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= prob_threshold:
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score'],
                                                  sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_predicted_sentids[:top_n]  # Important sentences for scaling training. Jul 21.
        item['predicted_sentids'] = [sid for sid, _, _ in item['scored_sentids']][:top_n]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        # item['predicted_label'] = item['label']  # give ground truth label

    return d_list


def threshold_sampler_insure_unique_new_format(org_data_file, full_sent_list, prob_threshold=0.5,
                                               logist_threshold=None, top_n=5):
    """
    Providing samples to the Training set by a probability threshold on the upstream selected sentences.
    """
    d_list = common.load_jsonl(org_data_file)
    augmented_dict: Dict[int, Dict[str, Dict]] = dict()
    print("Build selected sentences file:", len(full_sent_list))
    for sent_item in tqdm(full_sent_list):
        sent_item['element'] = sent_item['element'].replace(SMILE_SEP, fever_scorer.SENT_LINE)

        sid = sent_item['element']  # The id for the current one selection.
        fid = sent_item['fid']
        org_id = int(sent_item['oid'])
        # remain_str = selection_id.split('<##>')[1]
        # doc_id = remain_str.split(c_scorer.SENT_LINE)[0]
        # ln = int(remain_str.split(c_scorer.SENT_LINE)[1])
        if org_id in augmented_dict:
            if sid not in augmented_dict[org_id]:
                augmented_dict[org_id][sid] = sent_item
            else:
                print("Exist")
        else:
            augmented_dict[org_id] = {sid: sent_item}

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            # print("Potential error?")
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])].values()
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= prob_threshold:
                    # cur_predicted_sentids.append((sent_i['sid'], sent_i['score'],
                    #                               sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                    cur_predicted_sentids.append((sent_i['element'], sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_predicted_sentids[:top_n]  # Important sentences for scaling training. Jul 21.
        item['predicted_sentids'] = [sid for sid, _ in item['scored_sentids']][:top_n]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        # item['predicted_label'] = item['label']  # give ground truth label

    return d_list


def convert_evidence2scoring_format(predicted_sentids):
    e_list = predicted_sentids
    pred_evidence_list = []
    for i, cur_e in enumerate(e_list):
        doc_id = cur_e.split(fever_scorer.SENT_LINE)[0]
        ln = cur_e.split(fever_scorer.SENT_LINE)[1]
        pred_evidence_list.append([doc_id, int(ln)])
    return pred_evidence_list


def convert_evidence2scoring_format_smile(predicted_sentids, SMILE_SEP='(-.-)'):
    e_list = predicted_sentids
    pred_evidence_list = []
    for i, cur_e in enumerate(e_list):
        doc_id = cur_e.split(SMILE_SEP)[0]
        ln = cur_e.split(SMILE_SEP)[1]
        pred_evidence_list.append([doc_id, int(ln)])
    return pred_evidence_list