from evaluation import ext_hotpot_eval
from utils import common
import config
from tqdm import tqdm
import numpy as np

ID_SEPARATOR = '<-.->'


def build_sent_match_data_from_distractor_item(item, is_training=False, title_head=True):
    sent_list = []

    supporting_facts = item['supporting_facts']
    supporting_doc = set([fact[0] for fact in item['supporting_facts']])
    # print(supporting_doc)

    all_facts = []
    for title, sentences in item['context']:
        for i in range(len(sentences)):
            all_facts.append([title, i, sentences[i]])

    for fact in all_facts:
        d_item = dict()
        d_item['in_sp_doc'] = False
        title, sent_num, sentence = fact
        pid = str(item['_id']) + ID_SEPARATOR + str(title) + ID_SEPARATOR + str(sent_num)
        if is_training:
            if [title, sent_num] in supporting_facts:
                # Positive example
                d_item['selection_id'] = pid
                d_item['label'] = 'true'
                d_item['question'] = item['question']
                if sent_num != 0:
                    d_item['sentence'] = title + ' [t] ' + sentence
                else:
                    d_item['sentence'] = sentence
                d_item['in_sp_doc'] = True
            else:
                # Negative example
                d_item['selection_id'] = pid
                d_item['label'] = 'false'
                d_item['question'] = item['question']
                if sent_num != 0:
                    d_item['sentence'] = title + ' [t] ' + sentence
                else:
                    d_item['sentence'] = sentence

                if title in supporting_doc:
                    d_item['in_sp_doc'] = True
        else:
            d_item['selection_id'] = pid
            d_item['label'] = 'hidden'
            d_item['question'] = item['question']
            if sent_num != 0:
                d_item['sentence'] = title + ' [t] ' + sentence
            else:
                d_item['sentence'] = sentence

        sent_list.append(d_item)

    return sent_list


def build_sent_match_data_from_distractor_list(d_list, is_training=False, title_head=True):
    whole_data_list = []
    print("Sample sentence match data.")
    for item in tqdm(d_list):
        whole_data_list.extend(build_sent_match_data_from_distractor_item(item, is_training, title_head))
    return whole_data_list


def stimulate_prediction_file(d_list, sent_list):
    pred_dict = {'sp': dict()}

    for sent_item in sent_list:
        sid = sent_item['selection_id']
        oid, title, sent_num = sid.split(ID_SEPARATOR)
        sent_num = int(sent_num)
        # Change this to other later
        if oid not in pred_dict['sp']:
            pred_dict['sp'][oid] = []
        if sent_item['label'] == 'true':
            pred_dict['sp'][oid].append([title, sent_num])

    return pred_dict


def downsample_negative_examples(sent_p_list, selection_prob, same_doc_prob):
    r_list = []
    for p_item in sent_p_list:
        # Down sample neg examples
        if p_item['label'] == 'false' and not p_item['in_sp_doc']:
            p_v = np.random.rand()
            if p_v < selection_prob:
                r_list.append(p_item)
        elif p_item['label'] == 'false' and p_item['in_sp_doc']:
            p_v = np.random.rand()
            if p_v < same_doc_prob:
                r_list.append(p_item)
        else:
            r_list.append(p_item)

    return r_list


if __name__ == '__main__':
    train_list = common.load_json(config.TRAIN_FILE)
    # train_list = common.load_json(config.DEV_FULLWIKI_FILE)
    # train_list = common.load_json(config.DEV_DISTRACTOR_FILE)
    # print(len(train_list))
    train_sent_data_list = build_sent_match_data_from_distractor_list(train_list, is_training=True)
    print(len(train_sent_data_list))
    train_sent_data_list = downsample_negative_examples(train_sent_data_list, 0.1, 1)
    print(len(train_sent_data_list))
    neg = 0
    pos = 0
    in_sp_doc = 0
    for p_item in train_sent_data_list:
        if p_item['label'] == 'true':
            pos += 1
        elif p_item['label'] == 'false':
            neg += 1

        if p_item['in_sp_doc']:
            in_sp_doc += 1

    print(in_sp_doc)
    print(pos, neg, pos + neg)
    # pred_d = stimulate_prediction_file(train_list, train_sent_data_list)
    #
    # ext_hotpot_eval.eval(pred_d, train_list)
