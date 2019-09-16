from evaluation import ext_hotpot_eval
from utils import common
import config
from tqdm import tqdm
import numpy as np

ID_SEPARATOR = '<-.->'


def build_paragraph_match_data_from_list(d_list, is_training=False, title_head=True):
    # This method should be combined with the method below (build_paragraph_match_data_from_item) to provide paragraph_match data to be forwarded through neural nets.
    whole_data_list = []
    print("Sample sentence match data.")
    for item in tqdm(d_list):
        whole_data_list.extend(build_paragraph_match_data_from_item(item, is_training, title_head))
    return whole_data_list


def build_paragraph_match_data_from_item(item, is_training=False, title_head=True):
    # This method should be combined with the method above (build_paragraph_match_data_from_list)
    forward_list = []

    qid = item['_id']
    contexts = item['context']
    question = item['question']

    if is_training:
        supporting_facts = item['supporting_facts']
        supporting_doc = set([fact[0] for fact in supporting_facts])
    else:
        supporting_facts = []
        supporting_doc = set()

    retrieved_context_dict = dict()
    # This is the retrieved upstream context.   The key is document title and the nested dict is (line_num, sentence).

    for doc_title, context_sents in contexts:
        if doc_title not in retrieved_context_dict:
            retrieved_context_dict[doc_title] = dict()

        for i, sent in enumerate(context_sents):
            retrieved_context_dict[doc_title][i] = sent

    for doc_title, context_sents_dict in retrieved_context_dict.items():
        fitem = dict()
        paragraph = ''.join([s for k, s in context_sents_dict.items()])
        if title_head:
            paragraph = doc_title + ' [T] ' + paragraph
        fitem['context'] = paragraph
        fitem['query'] = question
        fitem['qid'] = qid
        fid = str(qid) + ID_SEPARATOR + str(doc_title)  # This id number is deprecated, but we keep it here.
        fitem['doc_title'] = str(doc_title)
        fitem['fid'] = fid

        if is_training:
            if doc_title in supporting_doc:
                # Positive example
                fitem['selection_label'] = 'true'
            else:
                # Negative example
                fitem['selection_label'] = 'false'
        else:
            fitem['selection_label'] = 'hidden'

        forward_list.append(fitem)

    return forward_list


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
                    d_item['sentence'] = title + ' [T] ' + sentence
                else:
                    d_item['sentence'] = sentence
                d_item['in_sp_doc'] = True
            else:
                # Negative example
                d_item['selection_id'] = pid
                d_item['label'] = 'false'
                d_item['question'] = item['question']
                if sent_num != 0:
                    d_item['sentence'] = title + ' [T] ' + sentence
                else:
                    d_item['sentence'] = sentence

                if title in supporting_doc:
                    d_item['in_sp_doc'] = True
        else:
            d_item['selection_id'] = pid
            d_item['label'] = 'hidden'
            d_item['question'] = item['question']
            if sent_num != 0:
                d_item['sentence'] = title + ' [T] ' + sentence
            else:
                d_item['sentence'] = sentence

        sent_list.append(d_item)

    return sent_list


if __name__ == '__main__':
    dev_fullwiki = common.load_json(config.DEV_FULLWIKI_FILE)
    fdata_list = build_paragraph_match_data_from_list(dev_fullwiki, is_training=False)
    print(fdata_list[200])

