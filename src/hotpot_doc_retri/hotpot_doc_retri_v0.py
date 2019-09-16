import unicodedata

import regex
from flashtext import KeywordProcessor

from build_rindex.build_rvindex import load_from_file
from build_rindex.rvindex_scoring import get_query_ngrams, get_query_doc_score
# from hotpot_fact_selection_sampler.sampler_full_wiki import append_baseline_context
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
import numpy as np


def append_baseline_context(doc_results, baseline_data_list):
    data_list = baseline_data_list
    for item in data_list:
        key = item['_id']
        contexts = item['context']
        provided_title = []
        for title, paragraph in contexts:
            provided_title.append(title)

        doc_results['sp_doc'][key] = list(set.union(set(doc_results['sp_doc'][key]), set(provided_title)))


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


_MatchedObject = collections.namedtuple(  # pylint: disable=invalid-name
        "MatchedObject", ["matched_key_word", "matched_keywords_info"])
    # Extracted key word is the key word in the database, matched word is the word in the input question.


def get_kw_matching_results(question, valid_query_terms, retrieved_set, match_filtering_k,
                            g_score_dict,  keyword_processor, keyword_processor_disamb):
    # 1. First retrieve raw key word matching.

    finded_keys_kwm: List[_MatchedObject, int, int] = keyword_processor.extract_keywords(question, span_info=True)
    finded_keys_kwm_disamb: List[_MatchedObject, int, int] = keyword_processor_disamb.extract_keywords(question,
                                                                                                       span_info=True)
    finded_keys_list: List[Tuple[str, str, str, int, int]] = []
    # retrieved_set = retrieval_utils.RetrievedSet()

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
                not_valid = True  # Skip this match bc this match is already contained in some other match.
                break

        if not_valid:
            continue
        # add finished

        if method == 'kwm':
            retrieved_set.add_item(retrieval_utils.RetrievedItem(title, 'kwm'))
            score = get_query_doc_score(valid_query_terms, title,
                                        g_score_dict)  # A function to compute between title and query
            retrieved_set.score_item(title, score, namespace=matched_word)
            all_raw_matched_word.add(matched_word)

    # .2 Then, we find the raw matching.
    for matched_word, title, method, start, end in finded_keys_list:
        # add after debug_2
        not_valid = False
        for e_start, e_end in all_finded_span_2:
            if (e_start < start and e_end >= end) or (e_start <= start and e_end > end):
                not_valid = True  # Skip this match bc this match is already contained in some other match.
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

    return retrieved_set


def init_results_v8(data_list, gt_data_list,
                    terms_based_resutls, g_score_dict,
                    match_filtering_k=3,
                    term_retrieval_top_k=5,
                    multihop_retrieval_top_k=None):
    # 2019-04-06
    # The complete v7 version of retrieval

    ner_set = get_title_entity_set()

    # dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print("Total data length:")
    print(len(data_list))

    # We load term-based results
    print("Load term-based results.")
    terms_based_results_dict = dict()
    for item in terms_based_resutls:
        terms_based_results_dict[item['qid']] = item

    # Load tf-idf_score function:
    # g_score_dict = dict()
    # load_from_file(g_score_dict,
    #                config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")

    keyword_processor = KeywordProcessor(case_sensitive=True)
    keyword_processor_disamb = KeywordProcessor(case_sensitive=True)

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
    # doc_pred_dict_p1 = {'sp_doc': dict(), 'raw_retrieval_set': dict()}

    for item in tqdm(data_list):
        question = item['question']
        qid = item['_id']

        query_terms = get_query_ngrams(question)
        valid_query_terms = [term for term in query_terms if term in g_score_dict]

        retrieved_set = RetrievedSet()

        # This method will add the keyword match results in-place to retrieved_set.
        get_kw_matching_results(question, valid_query_terms, retrieved_set, match_filtering_k,
                                g_score_dict, keyword_processor, keyword_processor_disamb)

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
                if potential_title in ner_set and not filter_word(potential_title) and not filter_document_id(
                        potential_title):   # important bug fixing 'or' to 'and'
                    # hyperlinked_title.append(potential_title)

                    # if not filter_document_id(potential_title):
                    score = get_query_doc_score(valid_query_terms, potential_title, g_score_dict)
                    retrieved_set.add_item(retrieval_utils.RetrievedItem(potential_title, 'kwm_disamb_hlinked'))
                    retrieved_set.score_item(potential_title, score, namespace=keyword_group + '-2-hop')

        for keyword_group in finded_keys_set:
            retrieved_set.sort_and_filter(keyword_group + '-2-hop', top_k=multihop_retrieval_top_k)

        doc_pred_dict['sp_doc'][qid] = retrieved_set.to_id_list()
        doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    if gt_data_list is not None:
        ext_hotpot_eval.eval(doc_pred_dict, gt_data_list)
    return doc_pred_dict


def results_multihop_filtering(pred_dict, multihop_retrieval_top_k=3, strict_mode=False):
    ner_set = get_title_entity_set()

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

        for keyword_group in hyperlinked_keyword_group:  # The group already has '-2-hop' in the end
            # if keyword_group not in ner_set:
            #     continue    # Important update 2019-04-07, we skip the one not in wiki title set.
            # retrieved_set.sort_and_filter(keyword_group + '-2-hop', top_k=hyperlinked_top_k)
            retrieved_set.sort_and_filter(keyword_group, top_k=multihop_retrieval_top_k, strict_mode=strict_mode)

        new_doc_pred_dict['sp_doc'][qid] = retrieved_set.to_id_list()
        new_doc_pred_dict['raw_retrieval_set'][qid] = retrieved_set

    return new_doc_pred_dict


def experiment_dev_full_wiki():
    multihop_retrieval_top_k = 3
    match_filtering_k = 3
    term_retrieval_top_k = 5

    data_list = common.load_json(config.DEV_FULLWIKI_FILE)
    terms_based_results_list = common.load_jsonl(config.RESULT_PATH / "doc_retri_results/term_based_methods_results/hotpot_tf_idf_dev.jsonl")
    g_score_dict = dict()
    load_from_file(g_score_dict,
                   config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")
    doc_retri_pred_dict = init_results_v8(data_list, data_list, terms_based_results_list, g_score_dict,
                                          match_filtering_k=match_filtering_k,
                                          term_retrieval_top_k=term_retrieval_top_k)

    len_list = []
    for rset in doc_retri_pred_dict['sp_doc'].values():
        len_list.append(len(rset))

    print("Results without filtering:")
    print(collections.Counter(len_list).most_common(10000))
    print(len(len_list))
    print("Mean:\t", np.mean(len_list))
    print("Std:\t", np.std(len_list))
    print("Max:\t", np.max(len_list))
    print("Min:\t", np.min(len_list))

    common.save_json(doc_retri_pred_dict, "hotpot_dev_doc_retrieval_v8_before_multihop_filtering.json")

    # Filtering
    new_doc_retri_pred_dict = results_multihop_filtering(doc_retri_pred_dict,
                                                         multihop_retrieval_top_k=multihop_retrieval_top_k)
    print("Results with filtering:")

    len_list = []
    for rset in new_doc_retri_pred_dict['sp_doc'].values():
        len_list.append(len(rset))

    print("Results with filtering:")
    print(collections.Counter(len_list).most_common(10000))
    print(len(len_list))
    print("Mean:\t", np.mean(len_list))
    print("Std:\t", np.std(len_list))
    print("Max:\t", np.max(len_list))
    print("Min:\t", np.min(len_list))

    ext_hotpot_eval.eval(new_doc_retri_pred_dict, data_list)
    common.save_json(new_doc_retri_pred_dict, "hotpot_dev_doc_retrieval_v8.json")


def experiment_test_full_wiki():
    multihop_retrieval_top_k = 3
    match_filtering_k = 3
    term_retrieval_top_k = 5

    data_list = common.load_json(config.TEST_FULLWIKI_FILE)
    terms_based_results_list = common.load_jsonl(config.RESULT_PATH / "doc_retri_results/term_based_methods_results/hotpot_tf_idf_test.jsonl")
    g_score_dict = dict()
    load_from_file(g_score_dict,
                   config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")
    # WE need to give gt data None.
    doc_retri_pred_dict = init_results_v8(data_list, None, terms_based_results_list, g_score_dict,
                                          match_filtering_k=match_filtering_k,
                                          term_retrieval_top_k=term_retrieval_top_k)

    len_list = []
    for rset in doc_retri_pred_dict['sp_doc'].values():
        len_list.append(len(rset))

    print("Results without filtering:")
    print(collections.Counter(len_list).most_common(10000))
    print(len(len_list))
    print("Mean:\t", np.mean(len_list))
    print("Std:\t", np.std(len_list))
    print("Max:\t", np.max(len_list))
    print("Min:\t", np.min(len_list))

    common.save_json(doc_retri_pred_dict, "hotpot_test_doc_retrieval_v8_before_multihop_filtering.json")

    # Filtering
    new_doc_retri_pred_dict = results_multihop_filtering(doc_retri_pred_dict,
                                                         multihop_retrieval_top_k=multihop_retrieval_top_k)
    print("Results with filtering:")

    len_list = []
    for rset in new_doc_retri_pred_dict['sp_doc'].values():
        len_list.append(len(rset))

    print("Results with filtering:")
    print(collections.Counter(len_list).most_common(10000))
    print(len(len_list))
    print("Mean:\t", np.mean(len_list))
    print("Std:\t", np.std(len_list))
    print("Max:\t", np.max(len_list))
    print("Min:\t", np.min(len_list))

    # ext_hotpot_eval.eval(new_doc_retri_pred_dict, data_list)
    common.save_json(new_doc_retri_pred_dict, "hotpot_test_doc_retrieval_v8.json")


def experiment_train_full_wiki():
    multihop_retrieval_top_k = 3
    match_filtering_k = 3
    term_retrieval_top_k = 5
    multihop_strict_mode = True
    debug_mode = None

    # data_list = common.load_json(config.DEV_FULLWIKI_FILE)
    data_list = common.load_json(config.TRAIN_FILE)

    if debug_mode is not None:
        data_list = data_list[:debug_mode]

    terms_based_results_list = common.load_jsonl(config.RESULT_PATH / "doc_retri_results/term_based_methods_results/hotpot_tf_idf_train.jsonl")

    g_score_dict = dict()
    load_from_file(g_score_dict,
                   config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")
    doc_retri_pred_dict = init_results_v8(data_list, data_list, terms_based_results_list, g_score_dict,
                                          match_filtering_k=match_filtering_k, term_retrieval_top_k=term_retrieval_top_k)

    len_list = []
    for rset in doc_retri_pred_dict['sp_doc'].values():
        len_list.append(len(rset))

    print("Results without filtering:")
    print(collections.Counter(len_list).most_common(10000))
    print(len(len_list))
    print("Mean:\t", np.mean(len_list))
    print("Std:\t", np.std(len_list))
    print("Max:\t", np.max(len_list))
    print("Min:\t", np.min(len_list))

    # common.save_json(doc_retri_pred_dict, f"hotpot_doc_retrieval_v8_before_multihop_filtering_{debug_mode}.json")
    common.save_json(doc_retri_pred_dict, f"hotpot_train_doc_retrieval_v8_before_multihop_filtering.json")

    # Filtering
    new_doc_retri_pred_dict = results_multihop_filtering(doc_retri_pred_dict,
                                                         multihop_retrieval_top_k=multihop_retrieval_top_k,
                                                         strict_mode=multihop_strict_mode)
    print("Results with filtering:")

    len_list = []
    for rset in new_doc_retri_pred_dict['sp_doc'].values():
        len_list.append(len(rset))

    print("Results with filtering:")
    print(collections.Counter(len_list).most_common(10000))
    print(len(len_list))
    print("Mean:\t", np.mean(len_list))
    print("Std:\t", np.std(len_list))
    print("Max:\t", np.max(len_list))
    print("Min:\t", np.min(len_list))

    ext_hotpot_eval.eval(new_doc_retri_pred_dict, data_list)
    # common.save_json(new_doc_retri_pred_dict, f"hotpot_doc_retrieval_v8_{debug_mode}.json")
    common.save_json(new_doc_retri_pred_dict, f"hotpot_train_doc_retrieval_v8.json")


if __name__ == '__main__':
    # experiment_dev_full_wiki()
    # experiment_test_full_wiki()
    # experiment_train_full_wiki()
    multihop_retrieval_top_k = 3
    match_filtering_k = 3
    term_retrieval_top_k = 5

    data_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print("Results with filtering:")
    new_doc_retri_pred_dict = common.load_json("/Users/yixin/projects/extinguishHotpot/results/doc_retri_results/doc_retrieval_final_v8/hotpot_dev_doc_retrieval_v8_before_multihop_filtering.json")
    len_list = []
    for rset in new_doc_retri_pred_dict['sp_doc'].values():
        len_list.append(len(rset))

    print("Results with filtering:")
    print(collections.Counter(len_list).most_common(10000))
    print(len(len_list))
    print("Mean:\t", np.mean(len_list))
    print("Std:\t", np.std(len_list))
    print("Max:\t", np.max(len_list))
    print("Min:\t", np.min(len_list))

    new_doc_retri_pred_dict = results_multihop_filtering(new_doc_retri_pred_dict,
                                                         multihop_retrieval_top_k=multihop_retrieval_top_k, strict_mode=False)

    # data_list = common.load_json(config.DEV_FULLWIKI_FILE)
    # append_baseline_context(new_doc_retri_pred_dict, data_list)

    print("Results with filtering:")

    len_list = []
    for rset in new_doc_retri_pred_dict['sp_doc'].values():
        len_list.append(len(rset))

    print("Results with filtering:")
    print(collections.Counter(len_list).most_common(10000))
    print(len(len_list))
    print("Mean:\t", np.mean(len_list))
    print("Std:\t", np.std(len_list))
    print("Max:\t", np.max(len_list))
    print("Min:\t", np.min(len_list))

    ext_hotpot_eval.eval(new_doc_retri_pred_dict, data_list)
    # analysis old:


    # doc_results = common.load_json(config.PRO_ROOT / "results/doc_retri_results/doc_retrieval_final_v8/hotpot_train_doc_retrieval_v8_before_multihop_filtering.json")
    # doc_results = results_multihop_filtering(doc_results, multihop_retrieval_top_k=3, strict_mode=True)
    #
    # # doc_results = common.load_json(config.RESULT_PATH / "doc_retri_results/doc_retrieval_debug_v7/doc_raw_matching_with_disamb_with_hyperlinked_v7_file_pipeline_top_none_redo_0.json")
    # # doc_results = results_multihop_filtering(doc_results, multihop_retrieval_top_k=3, strict_mode=True)
    #
    # len_list = []
    # for rset in doc_results['sp_doc'].values():
    #     len_list.append(len(rset))
    #
    # print("Results with filtering:")
    #
    # print(collections.Counter(len_list).most_common(10000))
    # print(len(len_list))
    # print("Mean:\t", np.mean(len_list))
    # print("Std:\t", np.std(len_list))
    # print("Max:\t", np.max(len_list))
    # print("Min:\t", np.min(len_list))
    # # data_list = common.load_json(config.DEV_FULLWIKI_FILE)
    # data_list = common.load_json(config.TRAIN_FILE)
    # ext_hotpot_eval.eval(doc_results, data_list)


# 'doc_f1': 0.07480048235601756, 'doc_prec': 0.03940029436426468, 'doc_recall': 0.8945982444294396

