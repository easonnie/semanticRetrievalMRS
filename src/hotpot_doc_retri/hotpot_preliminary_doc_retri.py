import unicodedata

import regex
from flashtext import KeywordProcessor

from build_rindex.build_rvindex import load_from_file
from build_rindex.rvindex_scoring import get_query_ngrams, get_query_doc_score
from wiki_util.title_entities_set import get_title_entity_set
import wiki_util.title_entities_set
from wiki_util import wiki_db_tool
from tqdm import tqdm
import config
from utils import common
from evaluation import ext_hotpot_eval
import re
from hotpot_doc_retri import retrieval_utils
from typing import Dict, List, Tuple
import collections
import json
from hotpot_doc_retri.retrieval_utils import RetrievedItem, RetrievedSet


def filter_disamb_doc(input_string):
    if ' (disambiguation)' in input_string:
        return True
    else:
        return False


def check_arabic(input_string):
    res = re.findall(
        r'[\U00010E60-\U00010E7F]|[\U0001EE00-\U0001EEFF]|[\u0750-\u077F]|[\u08A0-\u08FF]|[\uFB50-\uFDFF]|[\uFE70-\uFEFF]|[\u0600-\u06FF]',
        input_string)

    if len(res) != 0:
        return True
    else:
        return False


def filter_document_id(input_string, remove_disambiguation_doc=True):
    pid_words = input_string.strip().replace('_', ' ')
    match = re.search('[a-zA-Z]', pid_words)
    if match is None:  # filter id that contains no alphabets characters
        return True
    elif check_arabic(pid_words):  # remove id that contain arabic characters.
        return True
    else:
        if remove_disambiguation_doc:
            if filter_disamb_doc(input_string):
                return True
        return False


STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def filter_word(text):
    """Take out english stopwords, punctuation, and compound endings."""
    text = normalize(text)
    if regex.match(r'^\p{P}+$', text):
        return True
    if text.lower() in STOPWORDS:
        return True
    return False


def toy_init_results():
    ner_set = get_title_entity_set()

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print(len(dev_fullwiki_list))

    keyword_processor = KeywordProcessor(case_sensitive=True)

    print("Build Processor")
    for kw in tqdm(ner_set):
        if kw.lower() in STOPWORDS or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            keyword_processor.add_keyword(kw, {kw})

    doc_pred_dict = {'sp_doc': dict()}

    for item in tqdm(dev_fullwiki_list):
        question = item['question']
        qid = item['_id']
        finded_keys = keyword_processor.extract_keywords(question)
        finded_keys_set = set()
        if isinstance(finded_keys, list) and len(finded_keys) != 0:
            finded_keys_set = set.union(*finded_keys)

        # Addons cut retrieved document to contain only two
        finded_keys_set = sorted(list(finded_keys_set), key=lambda x: len(x), reverse=True)
        top_n = 2
        finded_keys_set = finded_keys_set[:top_n]

        doc_pred_dict['sp_doc'][qid] = list(finded_keys_set)

    common.save_json(doc_pred_dict, "toy_doc_rm_stopword_top2_pred_file.json")


def toy_init_results_v1():
    ner_set = get_title_entity_set()

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print(len(dev_fullwiki_list))

    keyword_processor = KeywordProcessor(case_sensitive=True)

    print("Build Processor")
    for kw in tqdm(ner_set):
        if kw.lower() in STOPWORDS or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            keyword_processor.add_keyword(kw, {kw})

    doc_pred_dict = {'sp_doc': dict(), 'raw_retrieval_set': dict()}

    for item in tqdm(dev_fullwiki_list):
        question = item['question']
        qid = item['_id']
        finded_keys = keyword_processor.extract_keywords(question)
        finded_keys_set = set()
        retrieved_set = retrieval_utils.RetrievedSet()
        # .1 We first find the raw matching.
        if isinstance(finded_keys, list) and len(finded_keys) != 0:
            finded_keys_set = set.union(*finded_keys)

        for page_name in finded_keys_set:
            retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm'))

        # .2 Then we add more disambiguation titles.    # Comment out this to remove this function.
        for keyword in finded_keys_set:
            if keyword in wiki_util.title_entities_set.disambiguation_group:
                for page_name in wiki_util.title_entities_set.disambiguation_group[keyword]:
                    retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb'))
                # finded_keys_set = set.union(finded_keys_set, wiki_util.title_entities_set.disambiguation_group[keyword])

        finded_keys_set = set(
            retrieved_set.to_id_list())  # for finding hyperlinked pages we do for both keyword matching and disambiguration group.
        # .3 We then add some hyperlinked title
        db_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_DB)
        hyperlinked_title = []
        for keyword in finded_keys_set:
            flatten_hyperlinks = []
            hyperlinks = wiki_db_tool.get_first_paragraph_hyperlinks(db_cursor, keyword)
            for hls in hyperlinks:
                flatten_hyperlinks.extend(hls)

            for hl in flatten_hyperlinks:
                potential_title = hl.href
                if potential_title in ner_set and potential_title.lower() not in STOPWORDS or not filter_document_id(
                        potential_title):
                    hyperlinked_title.append(potential_title)
                    # finded_keys_set.add(potential_title)

        # finded_keys_set = set.union(set(hyperlinked_title), finded_keys_set)
        for page_name in hyperlinked_title:
            retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb_hlinked'))
        # Addons cut retrieved document to contain only two
        # finded_keys_set = sorted(list(finded_keys_set), key=lambda x: len(x), reverse=True)
        # top_n = 2
        # finded_keys_set = finded_keys_set[:top_n]
        retrieved_list = retrieved_set.to_id_list()

        doc_pred_dict['sp_doc'][qid] = list(retrieved_list)
        doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    common.save_json(doc_pred_dict, "doc_raw_matching_with_disamb_with_hyperlinked_file.json")


def toy_init_results_v2():
    ner_set = get_title_entity_set()

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print(len(dev_fullwiki_list))

    keyword_processor = KeywordProcessor(case_sensitive=True)
    keyword_processor_disamb = KeywordProcessor(case_sensitive=True)

    print("Build Processor")
    for kw in tqdm(ner_set):
        if kw.lower() in STOPWORDS or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            keyword_processor.add_keyword(kw, {kw})

    for kw in wiki_util.title_entities_set.disambiguation_group:
        if kw.lower() in STOPWORDS or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            keyword_processor_disamb.add_keyword(kw, wiki_util.title_entities_set.disambiguation_group[kw])

    doc_pred_dict = {'sp_doc': dict(), 'raw_retrieval_set': dict()}

    for item in tqdm(dev_fullwiki_list):
        question = item['question']
        qid = item['_id']
        finded_keys = keyword_processor.extract_keywords(question)
        finded_keys_set = set()
        retrieved_set = retrieval_utils.RetrievedSet()
        # .1 We first find the raw matching.
        if isinstance(finded_keys, list) and len(finded_keys) != 0:
            finded_keys_set = set.union(*finded_keys)

        for page_name in finded_keys_set:
            retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm'))

        # del keyword_processor

        # .2 Then we add more disambiguation titles.    # Comment out this to remove this function.
        finded_keys_disamb = keyword_processor_disamb.extract_keywords(question)
        finded_keys_disamb_set = set()
        if isinstance(finded_keys_disamb, list) and len(finded_keys_disamb) != 0:
            finded_keys_disamb_set = set.union(*finded_keys_disamb)

        for page_name in finded_keys_disamb_set:
            # There will be duplicate pages, then we just ignore.
            retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb'))

        # del keyword_processor_disamb

        finded_keys_set = set(
            retrieved_set.to_id_list())  # for finding hyperlinked pages we do for both keyword matching and disambiguration group.
        # .3 We then add some hyperlinked title
        db_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_DB)
        hyperlinked_title = []
        for keyword in finded_keys_set:
            flatten_hyperlinks = []
            hyperlinks = wiki_db_tool.get_first_paragraph_hyperlinks(db_cursor, keyword)
            for hls in hyperlinks:
                flatten_hyperlinks.extend(hls)

            for hl in flatten_hyperlinks:
                potential_title = hl.href
                if potential_title in ner_set and potential_title.lower() not in STOPWORDS or not filter_document_id(
                        potential_title):
                    hyperlinked_title.append(potential_title)
                    # finded_keys_set.add(potential_title)

        # finded_keys_set = set.union(set(hyperlinked_title), finded_keys_set)
        for page_name in hyperlinked_title:
            retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb_hlinked'))
        # Addons cut retrieved document to contain only two
        # finded_keys_set = sorted(list(finded_keys_set), key=lambda x: len(x), reverse=True)
        # top_n = 2
        # finded_keys_set = finded_keys_set[:top_n]
        retrieved_list = retrieved_set.to_id_list()

        doc_pred_dict['sp_doc'][qid] = list(retrieved_list)
        doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    common.save_json(doc_pred_dict, "doc_raw_matching_with_disamb_with_hyperlinked_v2_file.json")


def toy_init_results_v3():
    # 2019 - 03 - 27
    # We want to merge raw key word matching and disambiguration group.
    ner_set = get_title_entity_set()

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print(len(dev_fullwiki_list))

    # keyword_processor = KeywordProcessor(case_sensitive=True)
    keyword_processor = KeywordProcessor(case_sensitive=False)
    # The structure for keyword_processor is {keyword: str: dict{kw: str: method: str} }

    print("Build Processor")
    for kw in tqdm(ner_set):
        if kw.lower() in STOPWORDS or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            keyword_processor.add_keyword(kw, {kw: 'kwm'})

    for kw in wiki_util.title_entities_set.disambiguation_group:
        if kw.lower() in STOPWORDS or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            if kw in keyword_processor:
                # if the kw existed in the kw_processor, we update its dict to add more disamb items
                existing_dict: Dict = keyword_processor.get_keyword(kw)
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    if disamb_kw not in existing_dict:
                        existing_dict[disamb_kw] = 'kwm_disamb'
            else:
                new_dict = dict()
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    new_dict[disamb_kw] = 'kwm_disamb'
                keyword_processor.add_keyword(kw, new_dict)

    doc_pred_dict = {'sp_doc': dict(), 'raw_retrieval_set': dict()}

    for item in tqdm(dev_fullwiki_list):
        question = item['question']
        qid = item['_id']

        finded_keys: List[Dict[str: str]] = keyword_processor.extract_keywords(question)
        finded_keys_list: List[Tuple[str, str]] = []
        retrieved_set = retrieval_utils.RetrievedSet()

        for finded_key in finded_keys:
            for title, method in finded_key.items():
                finded_keys_list.append((title, method))

        # .1 We first find the raw matching.
        for title, method in finded_keys_list:
            if method == 'kwm':
                retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm'))

        # .2 Then, we find the raw matching.
        for title, method in finded_keys_list:
            if method == 'kwm_disamb':
                retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm_disamb'))

        # .2 Then we add more disambiguation titles.
        # Since we merge the two dictionary, we don't need to processing again.
        # Comment out this to remove this function.
        # finded_keys_disamb = keyword_processor_disamb.extract_keywords(question)
        # finded_keys_disamb_set = set()
        # if isinstance(finded_keys_disamb, list) and len(finded_keys_disamb) != 0:
        #     finded_keys_disamb_set = set.union(*finded_keys_disamb)
        #
        # for page_name in finded_keys_disamb_set:
        #     # There will be duplicate pages, then we just ignore.
        #     retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb'))

        # del keyword_processor_disamb

        finded_keys_set = set(
            retrieved_set.to_id_list())  # for finding hyperlinked pages we do for both keyword matching and disambiguration group.
        # .3 We then add some hyperlinked title
        db_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_DB)
        hyperlinked_title = []
        for keyword in finded_keys_set:
            flatten_hyperlinks = []
            hyperlinks = wiki_db_tool.get_first_paragraph_hyperlinks(db_cursor, keyword)
            for hls in hyperlinks:
                flatten_hyperlinks.extend(hls)

            for hl in flatten_hyperlinks:
                potential_title = hl.href
                if potential_title in ner_set and potential_title.lower() not in STOPWORDS or not filter_document_id(
                        potential_title):
                    hyperlinked_title.append(potential_title)
                    # finded_keys_set.add(potential_title)

        # finded_keys_set = set.union(set(hyperlinked_title), finded_keys_set)
        for page_name in hyperlinked_title:
            retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb_hlinked'))
        # Addons cut retrieved document to contain only two
        # finded_keys_set = sorted(list(finded_keys_set), key=lambda x: len(x), reverse=True)
        # top_n = 2
        # finded_keys_set = finded_keys_set[:top_n]
        retrieved_list = retrieved_set.to_id_list()

        doc_pred_dict['sp_doc'][qid] = list(retrieved_list)
        doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    common.save_json(doc_pred_dict, "doc_raw_matching_with_disamb_with_hyperlinked_uncased_v3_file.json")


def toy_init_results_v4():
    # 2019-03-28
    # We first do raw key word matching and then disambiguation and
    # remove the overlapping span of kw and disambiguating.
    ner_set = get_title_entity_set()

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print(len(dev_fullwiki_list))

    keyword_processor = KeywordProcessor(case_sensitive=True)
    keyword_processor_disamb = KeywordProcessor(case_sensitive=True)

    print("Build Processor")
    for kw in tqdm(ner_set):
        if kw.lower() in STOPWORDS or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            keyword_processor.add_keyword(kw, {kw: 'kwm'})

    for kw in wiki_util.title_entities_set.disambiguation_group:
        if kw.lower() in STOPWORDS or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            if kw in keyword_processor:
                # if the kw existed in the kw_processor, we update its dict to add more disamb items
                existing_dict: Dict = keyword_processor.get_keyword(kw)
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    if disamb_kw not in existing_dict:
                        existing_dict[disamb_kw] = 'kwm_disamb'
            else:   # If not we add it to the keyword_processor_disamb, which is set to be lower priority
                new_dict = dict()
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    new_dict[disamb_kw] = 'kwm_disamb'
                keyword_processor_disamb.add_keyword(kw, new_dict)

    doc_pred_dict = {'sp_doc': dict(), 'raw_retrieval_set': dict()}

    for item in tqdm(dev_fullwiki_list):
        question = item['question']
        qid = item['_id']

        # 1. First retrieve raw key word matching.
        finded_keys_kwm: List[Dict[str: str], int, int] = keyword_processor.extract_keywords(question, span_info=True)
        finded_keys_kwm_disamb: List[Dict[str: str], int, int] = keyword_processor_disamb.extract_keywords(question,
                                                                                                           span_info=True)
        finded_keys_list: List[Tuple[str, str, int, int]] = []
        retrieved_set = retrieval_utils.RetrievedSet()

        whole_span = [False for _ in question]

        for finded_key, start, end in finded_keys_kwm:
            for i in range(start, end):
                whole_span[i] = True    # We mark the span as extracted by key word matching.
            for title, method in finded_key.items():
                finded_keys_list.append((title, method, start, end))

        for finded_key, start, end in finded_keys_kwm_disamb:
            valid = True
            for i in range(start, end):
                if whole_span[i]:
                    valid = False   # If we want a span overlapping, we just ignore this item.
                    break
            if valid:
                for title, method in finded_key.items():
                    finded_keys_list.append((title, method, start, end))

        # .1 We first find the raw matching.
        for title, method, start, end in finded_keys_list:
            if method == 'kwm':
                retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm'))

        # .2 Then, we find the raw matching.
        for title, method, start, end in finded_keys_list:
            if method == 'kwm_disamb':
                retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm_disamb'))

        finded_keys_set = set(
            retrieved_set.to_id_list())  # for finding hyperlinked pages we do for both keyword matching and disambiguration group.
        # .3 We then add some hyperlinked title
        db_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_DB)
        hyperlinked_title = []
        for keyword in finded_keys_set:
            flatten_hyperlinks = []
            hyperlinks = wiki_db_tool.get_first_paragraph_hyperlinks(db_cursor, keyword)
            for hls in hyperlinks:
                flatten_hyperlinks.extend(hls)

            for hl in flatten_hyperlinks:
                potential_title = hl.href
                if potential_title in ner_set and potential_title.lower() not in STOPWORDS or not filter_document_id(
                        potential_title):
                    hyperlinked_title.append(potential_title)
                    # finded_keys_set.add(potential_title)

        # finded_keys_set = set.union(set(hyperlinked_title), finded_keys_set)
        for page_name in hyperlinked_title:
            retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb_hlinked'))
        # Addons cut retrieved document to contain only two
        # finded_keys_set = sorted(list(finded_keys_set), key=lambda x: len(x), reverse=True)
        # top_n = 2
        # finded_keys_set = finded_keys_set[:top_n]
        retrieved_list = retrieved_set.to_id_list()

        doc_pred_dict['sp_doc'][qid] = list(retrieved_list)
        doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    common.save_json(doc_pred_dict, "doc_raw_matching_with_disamb_with_hyperlinked_v4_file.json")


def toy_init_results_v5():
    # 2019-03-28
    # We first do raw key word matching and then disambiguation and
    # remove the overlapping span of kw and disambiguating.
    ner_set = get_title_entity_set()

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print(len(dev_fullwiki_list))

    keyword_processor = KeywordProcessor(case_sensitive=True)
    keyword_processor_disamb = KeywordProcessor(case_sensitive=True)

    print("Build Processor")
    for kw in tqdm(ner_set):
        # if kw.lower() in STOPWORDS or filter_document_id(kw):
        if filter_word(kw) or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            keyword_processor.add_keyword(kw, {kw: 'kwm'})

    for kw in wiki_util.title_entities_set.disambiguation_group:
        # if kw.lower() in STOPWORDS or filter_document_id(kw):
        if filter_word(kw) or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            if kw in keyword_processor:
                # if the kw existed in the kw_processor, we update its dict to add more disamb items
                existing_dict: Dict = keyword_processor.get_keyword(kw)
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    if disamb_kw not in existing_dict:
                        existing_dict[disamb_kw] = 'kwm_disamb'
            else:   # If not we add it to the keyword_processor_disamb, which is set to be lower priority
                new_dict = dict()
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    new_dict[disamb_kw] = 'kwm_disamb'
                keyword_processor_disamb.add_keyword(kw, new_dict)

    doc_pred_dict = {'sp_doc': dict(), 'raw_retrieval_set': dict()}

    for item in tqdm(dev_fullwiki_list):
        question = item['question']
        qid = item['_id']

        # 1. First retrieve raw key word matching.
        finded_keys_kwm: List[Dict[str: str], int, int] = keyword_processor.extract_keywords(question, span_info=True)
        finded_keys_kwm_disamb: List[Dict[str: str], int, int] = keyword_processor_disamb.extract_keywords(question,
                                                                                                           span_info=True)
        finded_keys_list: List[Tuple[str, str, int, int]] = []
        retrieved_set = retrieval_utils.RetrievedSet()

        all_finded_span = []

        for finded_key, start, end in finded_keys_kwm:
            for i in range(start, end):
                all_finded_span.append((start, end))

            for title, method in finded_key.items():
                finded_keys_list.append((title, method, start, end))

        for finded_key, start, end in finded_keys_kwm_disamb:
            not_valid = False
            for e_start, e_end in all_finded_span:
                if e_start <= start and e_end >= end:
                    not_valid = True
                    break

            if not not_valid:
                for title, method in finded_key.items():
                    finded_keys_list.append((title, method, start, end))

        # .1 We first find the raw matching.
        for title, method, start, end in finded_keys_list:
            if method == 'kwm':
                retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm'))

        # .2 Then, we find the raw matching.
        for title, method, start, end in finded_keys_list:
            if method == 'kwm_disamb':
                retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm_disamb'))

        finded_keys_set = set(
            retrieved_set.to_id_list())  # for finding hyperlinked pages we do for both keyword matching and disambiguration group.
        # .3 We then add some hyperlinked title
        db_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_DB)
        hyperlinked_title = []
        for keyword in finded_keys_set:
            flatten_hyperlinks = []
            hyperlinks = wiki_db_tool.get_first_paragraph_hyperlinks(db_cursor, keyword)
            for hls in hyperlinks:
                flatten_hyperlinks.extend(hls)

            for hl in flatten_hyperlinks:
                potential_title = hl.href
                if potential_title in ner_set and potential_title.lower() not in STOPWORDS or not filter_document_id(
                        potential_title):
                    hyperlinked_title.append(potential_title)

        for page_name in hyperlinked_title:
            retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb_hlinked'))

        # Addons cut retrieved document to contain only two
        # finded_keys_set = sorted(list(finded_keys_set), key=lambda x: len(x), reverse=True)
        # top_n = 2
        # finded_keys_set = finded_keys_set[:top_n]
        retrieved_list = retrieved_set.to_id_list()

        doc_pred_dict['sp_doc'][qid] = list(retrieved_list)
        doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    common.save_json(doc_pred_dict, "doc_raw_matching_with_disamb_withiout_hyperlinked_v5_file.json")


def toy_init_results_v6():
    # 2019-03-28
    # We first do raw key word matching and then disambiguation and
    # remove the overlapping span of kw and disambiguating.
    match_filtering_k = 3
    ner_set = get_title_entity_set()

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print(len(dev_fullwiki_list))

    # Load tf-idf_score function:
    g_score_dict = dict()
    load_from_file(g_score_dict,
                   config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")

    keyword_processor = KeywordProcessor(case_sensitive=True)
    keyword_processor_disamb = KeywordProcessor(case_sensitive=True)

    _MatchedObject = collections.namedtuple(  # pylint: disable=invalid-name
        "MatchedObject", ["matched_key_word", "matched_keywords_info"])
    # Extracted key word is the key word in the database, matched word is the word in the input question.

    print("Build Processor")
    for kw in tqdm(ner_set):
        # if kw.lower() in STOPWORDS or filter_document_id(kw):
        if filter_word(kw) or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            matched_obj = _MatchedObject(matched_key_word=kw, matched_keywords_info={kw: 'kwm'})
            keyword_processor.add_keyword(kw, matched_obj)

    for kw in wiki_util.title_entities_set.disambiguation_group:
        # if kw.lower() in STOPWORDS or filter_document_id(kw):
        if filter_word(kw) or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            if kw in keyword_processor:
                # if the kw existed in the kw_processor, we update its dict to add more disamb items
                existing_matched_obj: _MatchedObject = keyword_processor.get_keyword(kw)
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    if filter_document_id(disamb_kw):
                        continue
                    if disamb_kw not in existing_matched_obj.matched_keywords_info:
                        existing_matched_obj.matched_keywords_info[disamb_kw] = 'kwm_disamb'
            else:   # If not we add it to the keyword_processor_disamb, which is set to be lower priority
                # new_dict = dict()
                matched_obj = _MatchedObject(matched_key_word=kw, matched_keywords_info=dict())
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    if filter_document_id(disamb_kw):
                        continue
                    matched_obj.matched_keywords_info[disamb_kw] = 'kwm_disamb'
                    # new_dict[disamb_kw] = 'kwm_disamb'
                keyword_processor_disamb.add_keyword(kw, matched_obj)

    doc_pred_dict = {'sp_doc': dict(), 'raw_retrieval_set': dict()}

    for item in tqdm(dev_fullwiki_list):
        question = item['question']
        qid = item['_id']

        query_terms = get_query_ngrams(question)
        valid_query_terms = [term for term in query_terms if term in g_score_dict]

        # 1. First retrieve raw key word matching.
        finded_keys_kwm: List[_MatchedObject, int, int] = keyword_processor.extract_keywords(question, span_info=True)
        finded_keys_kwm_disamb: List[_MatchedObject, int, int] = keyword_processor_disamb.extract_keywords(question,
                                                                                                           span_info=True)
        finded_keys_list: List[Tuple[str, str, str, int, int]] = []
        retrieved_set = retrieval_utils.RetrievedSet()

        all_finded_span = []
        all_finded_span_2 = []

        for finded_matched_obj, start, end in finded_keys_kwm:
            for i in range(start, end):
                all_finded_span.append((start, end))
                all_finded_span_2.append((start, end))

            # for matched_obj in finded_matched_obj.:
            matched_words = finded_matched_obj.matched_key_word
            for extracted_keyword, method in finded_matched_obj.matched_keywords_info.items():
                finded_keys_list.append((matched_words, extracted_keyword, method, start, end))

        for finded_matched_obj, start, end in finded_keys_kwm_disamb:
            not_valid = False
            for e_start, e_end in all_finded_span:
                if e_start <= start and e_end >= end:
                    not_valid = True
                    break

            if not not_valid:
                matched_words = finded_matched_obj.matched_key_word
                for extracted_keyword, method in finded_matched_obj.matched_keywords_info.items():
                    finded_keys_list.append((matched_words, extracted_keyword, method, start, end))
                    all_finded_span_2.append((start, end))

        all_raw_matched_word = set()
        # .1 We first find the raw matching.

        for matched_word, title, method, start, end in finded_keys_list:
            # add after debug_2
            not_valid = False
            for e_start, e_end in all_finded_span_2:
                if (e_start < start and e_end >= end) or (e_start <= start and e_end > end):
                    not_valid = True    # Skip this match bc this match is already contained in some other match.
                    break

            if not_valid:
                continue
            # add finished

            if method == 'kwm':
                retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm'))
                score = get_query_doc_score(valid_query_terms, title,
                                            g_score_dict)   # A function to compute between title and query
                retrieved_set.score_item(title, score, namespace=matched_word)
                all_raw_matched_word.add(matched_word)

        # .2 Then, we find the raw matching.
        for matched_word, title, method, start, end in finded_keys_list:
            # add after debug_2
            not_valid = False
            for e_start, e_end in all_finded_span_2:
                if (e_start < start and e_end >= end) or (e_start <= start and e_end > end):
                    not_valid = True    # Skip this match bc this match is already contained in some other match.
                    break

            if not_valid:
                continue
            # add finished

            if method == 'kwm_disamb':
                retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm_disamb'))
                score = get_query_doc_score(valid_query_terms, title,
                                            g_score_dict)  # A function to compute between title and query
                retrieved_set.score_item(title, score, namespace=matched_word)
                all_raw_matched_word.add(matched_word)

        for matched_word in all_raw_matched_word:
            retrieved_set.sort_and_filter(matched_word, top_k=match_filtering_k)

        # We don't worry about the hyperlink so far.

        # finded_keys_set = set(
        #     retrieved_set.to_id_list())  # for finding hyperlinked pages we do for both keyword matching and disambiguration group.
        # # .3 We then add some hyperlinked title
        # db_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_DB)
        # hyperlinked_title = []
        # for keyword in finded_keys_set:
        #     flatten_hyperlinks = []
        #     hyperlinks = wiki_db_tool.get_first_paragraph_hyperlinks(db_cursor, keyword)
        #     for hls in hyperlinks:
        #         flatten_hyperlinks.extend(hls)
        #
        #     for hl in flatten_hyperlinks:
        #         potential_title = hl.href
        #         if potential_title in ner_set and potential_title.lower() not in STOPWORDS or not filter_document_id(
        #                 potential_title):
        #             hyperlinked_title.append(potential_title)
        #
        # for page_name in hyperlinked_title:
        #     retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb_hlinked'))

        # Addons cut retrieved document to contain only two
        # finded_keys_set = sorted(list(finded_keys_set), key=lambda x: len(x), reverse=True)
        # top_n = 2
        # finded_keys_set = finded_keys_set[:top_n]
        retrieved_list = retrieved_set.to_id_list()

        doc_pred_dict['sp_doc'][qid] = list(retrieved_list)
        doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    common.save_json(doc_pred_dict, "doc_raw_matching_with_disamb_withiout_hyperlinked_v6_file_debug_4_redo_0.json")
    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    ext_hotpot_eval.eval(doc_pred_dict, dev_fullwiki_list)


def toy_init_results_v7_pre():
    # 2019-03-28
    # We first do raw key word matching and then disambiguation and
    # remove the overlapping span of kw and disambiguating.

    # match_filtering_k = 3
    ner_set = get_title_entity_set()
    term_retrieval_top_k = 5
    multihop_retrieval_top_k = None
    #
    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print(len(dev_fullwiki_list))

    terms_based_results = common.load_jsonl(config.RESULT_PATH / "doc_retri_results/term_based_methods_results/hotpot_tf_idf_dev.jsonl")

    terms_based_results_dict = dict()
    for item in terms_based_results:
        terms_based_results_dict[item['qid']] = item
        # print(item)

    # Load tf-idf_score function:
    g_score_dict = dict()
    load_from_file(g_score_dict,
                   config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")

    # keyword_processor = KeywordProcessor(case_sensitive=True)
    # keyword_processor_disamb = KeywordProcessor(case_sensitive=True)

    # _MatchedObject = collections.namedtuple(  # pylint: disable=invalid-name
    #     "MatchedObject", ["matched_key_word", "matched_keywords_info"])
    # Extracted key word is the key word in the database, matched word is the word in the input question.

    # print("Build Processor")
    # for kw in tqdm(ner_set):
    #     if kw.lower() in STOPWORDS or filter_document_id(kw):
        # if filter_word(kw) or filter_document_id(kw):
        #     continue  # if the keyword is filtered by above function or is stopwords
        # else:
        #     matched_obj = _MatchedObject(matched_key_word=kw, matched_keywords_info={kw: 'kwm'})
        #     keyword_processor.add_keyword(kw, matched_obj)

    # for kw in wiki_util.title_entities_set.disambiguation_group:
    #     # if kw.lower() in STOPWORDS or filter_document_id(kw):
    #     if filter_word(kw) or filter_document_id(kw):
    #         continue  # if the keyword is filtered by above function or is stopwords
    #     else:
    #         if kw in keyword_processor:
    #             # if the kw existed in the kw_processor, we update its dict to add more disamb items
    #             existing_matched_obj: _MatchedObject = keyword_processor.get_keyword(kw)
    #             for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
    #                 if filter_document_id(disamb_kw):
    #                     continue
    #                 if disamb_kw not in existing_matched_obj.matched_keywords_info:
    #                     existing_matched_obj.matched_keywords_info[disamb_kw] = 'kwm_disamb'
    #         else:   # If not we add it to the keyword_processor_disamb, which is set to be lower priority
    #             # new_dict = dict()
    #             matched_obj = _MatchedObject(matched_key_word=kw, matched_keywords_info=dict())
    #             for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
    #                 if filter_document_id(disamb_kw):
    #                     continue
    #                 matched_obj.matched_keywords_info[disamb_kw] = 'kwm_disamb'
    #                 # new_dict[disamb_kw] = 'kwm_disamb'
    #             keyword_processor_disamb.add_keyword(kw, matched_obj)
    #
    # doc_pred_dict = {'sp_doc': dict(), 'raw_retrieval_set': dict()}

    # Load some preobtained results.
    doc_pred_dict = common.load_json(config.RESULT_PATH / "doc_retri_results/doc_retrieval_debug_v6/doc_raw_matching_with_disamb_withiout_hyperlinked_v6_file_debug_4.json")

    for item in tqdm(dev_fullwiki_list):
        question = item['question']
        qid = item['_id']
        retrieved_set = doc_pred_dict['raw_retrieval_set'][qid]
        # print(type(retrieved_set))

        query_terms = get_query_ngrams(question)
        valid_query_terms = [term for term in query_terms if term in g_score_dict]

        new_sent_from_tf_idf = []
        for score, title in sorted(
                terms_based_results_dict[qid]['doc_list'], key=lambda x: x[0], reverse=True)[:term_retrieval_top_k]:
            # doc_pred_dict['sp_doc'][qid].append(title)
            retrieved_set.add_item(RetrievedItem(title, 'tf-idf'))

        finded_keys_set = set(
                retrieved_set.to_id_list())  # for finding hyperlinked pages we do for both keyword matching and disambiguration group.
            # .3 We then add some hyperlinked title
        db_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_DB)

        # hyperlinked_title = []
        # keyword_group = []

        for keyword_group in finded_keys_set:
            flatten_hyperlinks = []
            hyperlinks = wiki_db_tool.get_first_paragraph_hyperlinks(db_cursor, keyword_group)
            for hls in hyperlinks:
                flatten_hyperlinks.extend(hls)

            for hl in flatten_hyperlinks:
                potential_title = hl.href
                # if potential_title in ner_set and potential_title.lower() not in STOPWORDS or not filter_document_id(
                if potential_title in ner_set and not filter_word(potential_title) or not filter_document_id(
                        potential_title):
                    # hyperlinked_title.append(potential_title)

                    # if not filter_document_id(potential_title):
                    score = get_query_doc_score(valid_query_terms, potential_title, g_score_dict)
                    retrieved_set.add_item(retrieval_utils.RetrievedItem(potential_title, 'kwm_disamb_hlinked'))
                    retrieved_set.score_item(potential_title, score, namespace=keyword_group + '-2-hop')
    #                                         g_score_dict)   # A function to compute between title and query

        for keyword_group in finded_keys_set:   # Group ordering and filtering
            retrieved_set.sort_and_filter(keyword_group + '-2-hop', top_k=multihop_retrieval_top_k)

        # for page_name in hyperlinked_title:
        #     retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb_hlinked'))

        doc_pred_dict['sp_doc'][qid] = retrieved_set.to_id_list()
    #
    # for item in tqdm(dev_fullwiki_list):
    #     question = item['question']
    #     qid = item['_id']
    #
    #     query_terms = get_query_ngrams(question)
    #     valid_query_terms = [term for term in query_terms if term in g_score_dict]
    #
    #     # 1. First retrieve raw key word matching.
    #     finded_keys_kwm: List[_MatchedObject, int, int] = keyword_processor.extract_keywords(question, span_info=True)
    #     finded_keys_kwm_disamb: List[_MatchedObject, int, int] = keyword_processor_disamb.extract_keywords(question,
    #                                                                                                        span_info=True)
    #     finded_keys_list: List[Tuple[str, str, str, int, int]] = []
    #     retrieved_set = retrieval_utils.RetrievedSet()
    #
    #     all_finded_span = []
    #     all_finded_span_2 = []
    #
    #     for finded_matched_obj, start, end in finded_keys_kwm:
    #         for i in range(start, end):
    #             all_finded_span.append((start, end))
    #             all_finded_span_2.append((start, end))
    #
    #         # for matched_obj in finded_matched_obj.:
    #         matched_words = finded_matched_obj.matched_key_word
    #         for extracted_keyword, method in finded_matched_obj.matched_keywords_info.items():
    #             finded_keys_list.append((matched_words, extracted_keyword, method, start, end))
    #
    #     for finded_matched_obj, start, end in finded_keys_kwm_disamb:
    #         not_valid = False
    #         for e_start, e_end in all_finded_span:
    #             if e_start <= start and e_end >= end:
    #                 not_valid = True
    #                 break
    #
    #         if not not_valid:
    #             matched_words = finded_matched_obj.matched_key_word
    #             for extracted_keyword, method in finded_matched_obj.matched_keywords_info.items():
    #                 finded_keys_list.append((matched_words, extracted_keyword, method, start, end))
    #                 all_finded_span_2.append((start, end))
    #
    #     all_raw_matched_word = set()
    #     # .1 We first find the raw matching.
    #
    #     for matched_word, title, method, start, end in finded_keys_list:
    #         # add after debug_2
    #         not_valid = False
    #         for e_start, e_end in all_finded_span_2:
    #             if (e_start < start and e_end >= end) or (e_start <= start and e_end > end):
    #                 not_valid = True    # Skip this match bc this match is already contained in some other match.
    #                 break
    #
    #         if not_valid:
    #             continue
    #         # add finished
    #
    #         if method == 'kwm':
    #             retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm'))
    #             score = get_query_doc_score(valid_query_terms, title,
    #                                         g_score_dict)   # A function to compute between title and query
    #             retrieved_set.score_item(title, score, namespace=matched_word)
    #             all_raw_matched_word.add(matched_word)
    #
    #     # .2 Then, we find the raw matching.
    #     for matched_word, title, method, start, end in finded_keys_list:
    #         # add after debug_2
    #         not_valid = False
    #         for e_start, e_end in all_finded_span_2:
    #             if (e_start < start and e_end >= end) or (e_start <= start and e_end > end):
    #                 not_valid = True    # Skip this match bc this match is already contained in some other match.
    #                 break
    #
    #         if not_valid:
    #             continue
    #         # add finished
    #
    #         if method == 'kwm_disamb':
    #             retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm_disamb'))
    #             score = get_query_doc_score(valid_query_terms, title,
    #                                         g_score_dict)  # A function to compute between title and query
    #             retrieved_set.score_item(title, score, namespace=matched_word)
    #             all_raw_matched_word.add(matched_word)
    #
    #     for matched_word in all_raw_matched_word:
    #         retrieved_set.sort_and_filter(matched_word, top_k=match_filtering_k)

        # We don't worry about the hyperlink so far.

        # finded_keys_set = set(
        #     retrieved_set.to_id_list())  # for finding hyperlinked pages we do for both keyword matching and disambiguration group.
        # # .3 We then add some hyperlinked title
        # db_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_DB)
        # hyperlinked_title = []
        # for keyword in finded_keys_set:
        #     flatten_hyperlinks = []
        #     hyperlinks = wiki_db_tool.get_first_paragraph_hyperlinks(db_cursor, keyword)
        #     for hls in hyperlinks:
        #         flatten_hyperlinks.extend(hls)
        #
        #     for hl in flatten_hyperlinks:
        #         potential_title = hl.href
        #         if potential_title in ner_set and potential_title.lower() not in STOPWORDS or not filter_document_id(
        #                 potential_title):
        #             hyperlinked_title.append(potential_title)
        #
        # for page_name in hyperlinked_title:
        #     retrieved_set.add_item(retrieval_utils.RetrievedItem(page_name, 'kwm_disamb_hlinked'))

        # Addons cut retrieved document to contain only two
        # finded_keys_set = sorted(list(finded_keys_set), key=lambda x: len(x), reverse=True)
        # top_n = 2
        # finded_keys_set = finded_keys_set[:top_n]
        # retrieved_list = retrieved_set.to_id_list()
        #
        # doc_pred_dict['sp_doc'][qid] = list(retrieved_list)
        # doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    common.save_json(doc_pred_dict, "doc_raw_matching_with_disamb_with_hyperlinked_v7_file_debug_top_none.json")
    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    ext_hotpot_eval.eval(doc_pred_dict, dev_fullwiki_list)


def toy_init_results_v7():
    # 2019-04-05
    # The complete v7 version of retrieval
    # We first do raw key word matching and then disambiguation and
    # remove the overlapping span of kw and disambiguating.

    ner_set = get_title_entity_set()
    match_filtering_k = 3
    term_retrieval_top_k = 5
    multihop_retrieval_top_k = None

    #
    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print(len(dev_fullwiki_list))

    # We load term-based results
    terms_based_results = common.load_jsonl(config.RESULT_PATH / "doc_retri_results/term_based_methods_results/hotpot_tf_idf_dev.jsonl")

    terms_based_results_dict = dict()
    for item in terms_based_results:
        terms_based_results_dict[item['qid']] = item

    # Load tf-idf_score function:
    g_score_dict = dict()
    load_from_file(g_score_dict,
                   config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")

    keyword_processor = KeywordProcessor(case_sensitive=True)
    keyword_processor_disamb = KeywordProcessor(case_sensitive=True)

    _MatchedObject = collections.namedtuple(  # pylint: disable=invalid-name
        "MatchedObject", ["matched_key_word", "matched_keywords_info"])
    # Extracted key word is the key word in the database, matched word is the word in the input question.

    print("Build Processor")
    for kw in tqdm(ner_set):
        if filter_word(kw) or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            # matched_key_word is the original matched span. we need to save it for group ordering.
            matched_obj = _MatchedObject(matched_key_word=kw, matched_keywords_info={kw: 'kwm'})
            keyword_processor.add_keyword(kw, matched_obj)
    #
    for kw in wiki_util.title_entities_set.disambiguation_group:
        if filter_word(kw) or filter_document_id(kw):
            continue  # if the keyword is filtered by above function or is stopwords
        else:
            if kw in keyword_processor:
                # if the kw existed in the kw_processor, we update its dict to add more disamb items
                existing_matched_obj: _MatchedObject = keyword_processor.get_keyword(kw)
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    if filter_document_id(disamb_kw):
                        continue
                    if disamb_kw not in existing_matched_obj.matched_keywords_info:
                        existing_matched_obj.matched_keywords_info[disamb_kw] = 'kwm_disamb'
            else:   # If not we add it to the keyword_processor_disamb, which is set to be lower priority
                # new_dict = dict()
                matched_obj = _MatchedObject(matched_key_word=kw, matched_keywords_info=dict())
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    if filter_document_id(disamb_kw):
                        continue
                    matched_obj.matched_keywords_info[disamb_kw] = 'kwm_disamb'
                    # new_dict[disamb_kw] = 'kwm_disamb'
                keyword_processor_disamb.add_keyword(kw, matched_obj)

    doc_pred_dict = {'sp_doc': dict(), 'raw_retrieval_set': dict()}
    doc_pred_dict_p1 = {'sp_doc': dict(), 'raw_retrieval_set': dict()}

    for item in tqdm(dev_fullwiki_list):
        question = item['question']
        qid = item['_id']

        query_terms = get_query_ngrams(question)
        valid_query_terms = [term for term in query_terms if term in g_score_dict]

        # 1. First retrieve raw key word matching.
        finded_keys_kwm: List[_MatchedObject, int, int] = keyword_processor.extract_keywords(question, span_info=True)
        finded_keys_kwm_disamb: List[_MatchedObject, int, int] = keyword_processor_disamb.extract_keywords(question,
                                                                                                           span_info=True)
        finded_keys_list: List[Tuple[str, str, str, int, int]] = []
        retrieved_set = retrieval_utils.RetrievedSet()

        all_finded_span = []
        all_finded_span_2 = []

        for finded_matched_obj, start, end in finded_keys_kwm:
            for i in range(start, end):
                all_finded_span.append((start, end))
                all_finded_span_2.append((start, end))

            # for matched_obj in finded_matched_obj.:
            matched_words = finded_matched_obj.matched_key_word
            for extracted_keyword, method in finded_matched_obj.matched_keywords_info.items():
                finded_keys_list.append((matched_words, extracted_keyword, method, start, end))

        for finded_matched_obj, start, end in finded_keys_kwm_disamb:
            not_valid = False
            for e_start, e_end in all_finded_span:
                if e_start <= start and e_end >= end:
                    not_valid = True
                    break

            if not not_valid:
                matched_words = finded_matched_obj.matched_key_word
                for extracted_keyword, method in finded_matched_obj.matched_keywords_info.items():
                    finded_keys_list.append((matched_words, extracted_keyword, method, start, end))
                    all_finded_span_2.append((start, end))

        all_raw_matched_word = set()
        # .1 We first find the raw matching.

        for matched_word, title, method, start, end in finded_keys_list:
            # add after debug_2
            not_valid = False
            for e_start, e_end in all_finded_span_2:
                if (e_start < start and e_end >= end) or (e_start <= start and e_end > end):
                    not_valid = True    # Skip this match bc this match is already contained in some other match.
                    break

            if not_valid:
                continue
            # add finished

            if method == 'kwm':
                retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm'))
                score = get_query_doc_score(valid_query_terms, title,
                                            g_score_dict)   # A function to compute between title and query
                retrieved_set.score_item(title, score, namespace=matched_word)
                all_raw_matched_word.add(matched_word)

        # .2 Then, we find the raw matching.
        for matched_word, title, method, start, end in finded_keys_list:
            # add after debug_2
            not_valid = False
            for e_start, e_end in all_finded_span_2:
                if (e_start < start and e_end >= end) or (e_start <= start and e_end > end):
                    not_valid = True    # Skip this match bc this match is already contained in some other match.
                    break

            if not_valid:
                continue
            # add finished

            if method == 'kwm_disamb':
                retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm_disamb'))
                score = get_query_doc_score(valid_query_terms, title,
                                            g_score_dict)  # A function to compute between title and query
                retrieved_set.score_item(title, score, namespace=matched_word)
                all_raw_matched_word.add(matched_word)

        for matched_word in all_raw_matched_word:
            retrieved_set.sort_and_filter(matched_word, top_k=match_filtering_k)

        doc_pred_dict_p1['sp_doc'][qid] = retrieved_set.to_id_list()
        doc_pred_dict_p1['raw_retrieval_set'][qid] = retrieved_set

        # Then we add term-based matching results
        added_count = 0
        for score, title in sorted(
                terms_based_results_dict[qid]['doc_list'], key=lambda x: x[0], reverse=True)[:term_retrieval_top_k + 3]:
            if not filter_word(title) and not filter_document_id(title):
                retrieved_set.add_item(RetrievedItem(title, 'tf-idf'))
                added_count += 1
                if term_retrieval_top_k is not None and added_count >= term_retrieval_top_k:
                    break

        # Add hyperlinked pages:
        finded_keys_set = set(
            retrieved_set.to_id_list())  # for finding hyperlinked pages we do for both keyword matching and disambiguration group.
        # .3 We then add some hyperlinked title
        db_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_DB)

        for keyword_group in finded_keys_set:
            flatten_hyperlinks = []
            hyperlinks = wiki_db_tool.get_first_paragraph_hyperlinks(db_cursor, keyword_group)
            for hls in hyperlinks:
                flatten_hyperlinks.extend(hls)

            for hl in flatten_hyperlinks:
                potential_title = hl.href
                if potential_title in ner_set and not filter_word(potential_title) or not filter_document_id(
                        potential_title):
                    # hyperlinked_title.append(potential_title)

                    # if not filter_document_id(potential_title):
                    score = get_query_doc_score(valid_query_terms, potential_title, g_score_dict)
                    retrieved_set.add_item(retrieval_utils.RetrievedItem(potential_title, 'kwm_disamb_hlinked'))
                    retrieved_set.score_item(potential_title, score, namespace=keyword_group + '-2-hop')

        for keyword_group in finded_keys_set:
            retrieved_set.sort_and_filter(keyword_group + '-2-hop', top_k=multihop_retrieval_top_k)

        doc_pred_dict['sp_doc'][qid] = retrieved_set.to_id_list()
        doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    common.save_json(doc_pred_dict, "doc_raw_matching_with_disamb_with_hyperlinked_v7_file_pipeline_top_none_redo_0.json")
    common.save_json(doc_pred_dict_p1, "doc_raw_matching_with_disamb_with_hyperlinked_v7_file_pipeline_top_none_debug_p1.json")

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)

    ext_hotpot_eval.eval(doc_pred_dict, dev_fullwiki_list)
    ext_hotpot_eval.eval(doc_pred_dict_p1, dev_fullwiki_list)


def toy_init_pos_results_v7():
    hyperlinked_top_k = None

    pred_dict = common.load_json(config.PRO_ROOT / "src/doc_retri/doc_raw_matching_with_disamb_with_hyperlinked_v7_file_debug_top_none.json")

    new_doc_pred_dict = {'sp_doc': dict(), 'raw_retrieval_set': dict()}

    for key in pred_dict['raw_retrieval_set'].keys():
        qid = key
        retrieved_set: RetrievedSet = pred_dict['raw_retrieval_set'][key]

        hyperlinked_keyword_group = set()
        for item in retrieved_set.retrieved_dict.values():
            for keyword_gourp_name in item.scores_dict.keys():
                if keyword_gourp_name.endswith('-2-hop'):
                    hyperlinked_keyword_group.add(keyword_gourp_name)
                    # If the current scored one is 2-hop retrieval

        for keyword_group in hyperlinked_keyword_group: # The group already has '-2-hop' in the end
            # retrieved_set.sort_and_filter(keyword_group + '-2-hop', top_k=hyperlinked_top_k)
            retrieved_set.sort_and_filter(keyword_group, top_k=hyperlinked_top_k)

        new_doc_pred_dict['sp_doc'][qid] = retrieved_set.to_id_list()
        new_doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    # common.save_json(new_doc_pred_dict, "doc_raw_matching_with_disamb_with_hyperlinked_v7_file_pipeline_top_none.json")
    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    ext_hotpot_eval.eval(new_doc_pred_dict, dev_fullwiki_list)


if __name__ == '__main__':
    # toy_init_results_v1()
    # toy_init_results_v2()
    # toy_init_results_v3()
    # toy_init_results_v4()
    # toy_init_results_v5()
    # toy_init_results_v6()
    # toy_init_results_v7_pre()
    toy_init_results_v7()
    # toy_init_pos_results_v7()
    # get_title_entity_set()
    # print(wiki_util.title_entities_set.disambiguation_group)
    # pred_dev = common.load_json(config.RESULT_PATH / "doc_retri_results/toy_doc_rm_stopword_pred_file.json")
    # pred_dev = common.load_json(config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_file.json")
    # pred_dev = common.load_json(
    #     config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_v2_file.json")
    # pred_dev = common.load_json(
    #     config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_v3_file.json")
    # pred_dev = common.load_json(
    #     config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_v4_file.json")
    # pred_dev = common.load_json(
    #     config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_v5_file.json")
    # pred_dev = common.load_json(
    #     "/Users/yixin/projects/extinguishHotpot/src/doc_retri/doc_raw_matching_with_disamb_with_hyperlinked_v5_file.json")
    # pred_dev = common.load_json(
    #     "/Users/yixin/projects/extinguishHotpot/src/doc_retri/doc_raw_matching_with_disamb_withiout_hyperlinked_v6_file_debug_4.json")

    # pred_dev = common.load_json(config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_file.json")
    # pred_dev = common.load_json(config.RESULT_PATH / "doc_retri_results/doc_raw_matching_file.json")
    # pred_dev = common.load_json(config.RESULT_PATH / "doc_retri_results/toy_doc_rm_stopword_top2_pred_file.json")
    # pred_dev = common.load_json(config.RESULT_PATH / "doc_retri_results/toy_doc_rm_stopword_top2_pred_file.json")
    # print(len(pred_dev))
    # dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    # ext_hotpot_eval.eval(pred_dev, dev_fullwiki_list)

# 'doc_em': 0.04577987846049966, 'doc_f1': 0.35197722376656215, 'doc_prec': 0.33270779074627865, 'doc_recall': 0.412018906144497
# 'doc_em': 0.1489534098582039,  'doc_f1': 0.4042764580306798,  'doc_prec': 0.43380716590034823, 'doc_recall': 0.412018906144497

# 'doc_em': 0.15192437542201215, 'doc_f1': 0.4141590789733938,  'doc_prec': 0.4542443436974177,  'doc_recall': 0.41188386225523294
# 'doc_em': 0.18230925050641458, 'doc_f1': 0.4293720459149168,  'doc_prec': 0.4835246455097907,  'doc_recall': 0.4022957461174882

# upperbound 2-hop 'doc_recall': 0.8951384199864956
# 0.8951384199864956

# Pipeline expected results.
# 'doc_f1': 0.07500399833405762, 'doc_prec': 0.039512565895725084, 'doc_recall': 0.8951384199864956

# V7 cut
# {'em': 0.0, 'f1': 0.0, 'prec': 0.0, 'recall': 0.0, 'doc_em': 0.12032410533423363, 'doc_f1': 0.40915052664529045, 'doc_prec': 0.42250535090360103, 'doc_recall': 0.4812288993923025
#                                                    'doc_em': 0.12032410533423363, 'doc_f1': 0.40921579785843476, 'doc_prec': 0.42256161919079444, 'doc_recall': 0.4812964213369345

# 'doc_f1': 0.07481397826629528, 'doc_prec': 0.039406719378171556, 'doc_recall': 0.8950033760972316

# V6 reran
# 'doc_em': 0.12032410533423363, 'doc_f1': 0.40908020833729497, 'doc_prec': 0.4224800569687865, 'doc_recall': 0.48095881161377446
# 'doc_em': 0.12032410533423363, 'doc_f1': 0.4092293022473612, 'doc_prec': 0.4225616191907944, 'doc_recall': 0.4813639432815665