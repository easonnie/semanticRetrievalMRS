import collections

from flashtext import KeywordProcessor

import wiki_util
from build_rindex.build_rvindex import load_from_file
from build_rindex.rvindex_scoring import get_query_ngrams
from fever_utils.fever_db import reverse_convert_brc
from hotpot_doc_retri.hotpot_doc_retri_v0 import filter_word, filter_document_id, get_kw_matching_results

from hotpot_doc_retri.retrieval_utils import RetrievedSet, RetrievedItem
from utils import common
import config
from utils import list_dict_data_tool
from tqdm import tqdm
from evaluation import fever_scorer
from wiki_util.title_entities_set import get_title_entity_set
import numpy as np

_MatchedObject = collections.namedtuple(  # pylint: disable=invalid-name
    "MatchedObject", ["matched_key_word", "matched_keywords_info"])


# Extracted key word is the key word in the database, matched word is the word in the input question.

def item_resorting(d_list, top_k=None):
    for item in d_list:
        t_claim = ' '.join(item['claim_tokens'])
        item['predicted_docids'] = []
        # for it in item['prioritized_docids']:
        #     if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
        #         item['predicted_docids'].append(it[0])

        # Reset Exact match
        # t_claim = ' '.join(item['claim_tokens'])
        # item['predicted_docids'] = []
        for k, it in enumerate(item['prioritized_docids']):
            if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
                item['prioritized_docids'][k] = [it[0], 5.0]
                item['predicted_docids'].append(it[0])

        for it in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0])):
            if it[0] not in item['predicted_docids']:
                item['predicted_docids'].append(it[0])

        if top_k is not None and len(item['predicted_docids']) > top_k:
            item['predicted_docids'] = item['predicted_docids'][:top_k]


def fever_retrieval_v0(term_retrieval_top_k=3, match_filtering_k=2, tag='dev'):
    # term_retrieval_top_k = 20
    # term_retrieval_top_k = 20

    # term_retrieval_top_k = 3
    # match_filtering_k = 2

    if tag == 'dev':
        d_list = common.load_jsonl(config.FEVER_DEV)
    elif tag == 'train':
        d_list = common.load_jsonl(config.FEVER_TRAIN)
    elif tag == 'test':
        d_list = common.load_jsonl(config.FEVER_TEST)
    else:
        raise ValueError(f"Tag:{tag} not supported.")

    d_tf_idf = common.load_jsonl(config.RESULT_PATH /
                                 f"doc_retri_results/term_based_methods_results/fever_tf_idf_{tag}.jsonl")

    tf_idf_dict = list_dict_data_tool.list_to_dict(d_tf_idf, 'id')

    r_list = []

    ner_set = get_title_entity_set()

    g_score_dict = dict()
    load_from_file(g_score_dict,
                   config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")

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
            else:  # If not we add it to the keyword_processor_disamb, which is set to be lower priority
                # new_dict = dict()
                matched_obj = _MatchedObject(matched_key_word=kw, matched_keywords_info=dict())
                for disamb_kw in wiki_util.title_entities_set.disambiguation_group[kw]:
                    if filter_document_id(disamb_kw):
                        continue
                    matched_obj.matched_keywords_info[disamb_kw] = 'kwm_disamb'
                    # new_dict[disamb_kw] = 'kwm_disamb'
                keyword_processor_disamb.add_keyword(kw, matched_obj)

    for item in tqdm(d_list):
        cur_id = str(item['id'])
        query = item['claim']

        query_terms = get_query_ngrams(query)
        valid_query_terms = [term for term in query_terms if term in g_score_dict]

        retrieved_set = RetrievedSet()
        # print(tf_idf_doc_list)
        get_kw_matching_results(query, valid_query_terms, retrieved_set, match_filtering_k,
                                g_score_dict, keyword_processor, keyword_processor_disamb)

        tf_idf_doc_list = tf_idf_dict[cur_id]['retrieved_list']
        added_count = 0
        for score, title in sorted(
                tf_idf_doc_list, key=lambda x: x[0], reverse=True)[:term_retrieval_top_k + 3]:
            if not filter_word(title) and not filter_document_id(title) and not title.startswith('List of '):
                retrieved_set.add_item(RetrievedItem(title, 'tf-idf'))
                added_count += 1
                if term_retrieval_top_k is not None and added_count >= term_retrieval_top_k:
                    break

        predicted_docids = retrieved_set.to_id_list()
        # print(retrieved_set)
        # print(item['claim'], predicted_docids)

        r_item = dict()
        r_item['id'] = int(cur_id)
        r_item['claim'] = item['claim']
        r_item['predicted_docids'] = predicted_docids
        if tag != 'test':
            r_item['label'] = item['label']
        r_list.append(r_item)

    # r_list = common.load_jsonl('dev-debug.jsonl')

    # We need to modify the existing retrieved document for naming consistency
    for i, item in enumerate(r_list):
        predicted_docids = item['predicted_docids']
        modified_docids = []
        for docid in predicted_docids:
            docid = docid.replace(' ', '_')
            docid = reverse_convert_brc(docid)
            modified_docids.append(docid)
        item['predicted_docids'] = modified_docids
    # Modify finished

    # print(r_list[0:10])
    len_list = []
    for rset in r_list:
        len_list.append(len(rset['predicted_docids']))

    print(collections.Counter(len_list).most_common(10000))

    print(np.mean(len_list))
    print(np.std(len_list))
    print(np.max(len_list))
    print(np.min(len_list))

    common.save_jsonl(r_list, f'fever_term_based_retri_results_'
    f'{tag}_term_topk:{term_retrieval_top_k}_match_filtering_k:{match_filtering_k}.jsonl')

    mode = {'standard': False, 'check_doc_id_correct': True}
    # fever_scorer.fever_score_analysis(r_list, d_list, mode=mode, max_evidence=None)
    fever_scorer.fever_score(r_list, d_list, mode=mode, max_evidence=None)


def merge_results_with_haonao_module(term_retrieval_top_k=3, match_filtering_k=2, haonan_topk=10, tag='dev',
                                     save=False):
    if tag == 'dev':
        d_list = common.load_jsonl(config.FEVER_DEV)
        task_name = 'shared_task_dev'
    elif tag == 'train':
        d_list = common.load_jsonl(config.FEVER_TRAIN)
        task_name = 'train'
    elif tag == 'test':
        d_list = common.load_jsonl(config.FEVER_TEST)
        task_name = 'shared_task_test'
    else:
        raise ValueError(f"Tag:{tag} not supported.")

    # r_list = common.load_jsonl(config.RESULT_PATH / f'doc_retri_results/fever_results/standard_term_based_results/'
    # f'fever_term_based_retri_results_{tag}_term_topk:{term_retrieval_top_k}_match_filtering_k:{match_filtering_k}.jsonl')

    r_list = common.load_jsonl(config.RESULT_PATH / f'doc_retri_results/fever_results/standard_term_based_results/'
    f'fever_term_based_retri_results_{tag}_term_topk:{term_retrieval_top_k}_match_filtering_k:{match_filtering_k}.jsonl')

    old_result_list = common.load_jsonl(config.RESULT_PATH /
                                        f"doc_retri_results/fever_results/haonans_results/dr_{tag}.jsonl")
    item_resorting(old_result_list, top_k=haonan_topk)

    old_result_dict = list_dict_data_tool.list_to_dict(old_result_list, 'id')

    for i, item in enumerate(r_list):
        predicted_docids = item['predicted_docids']
        modified_docids = []
        for docid in predicted_docids:
            docid = docid.replace(' ', '_')
            docid = reverse_convert_brc(docid)
            modified_docids.append(docid)
        item['predicted_docids'] = modified_docids
        # item['predicted_docids'] = []

    merged_result_list = []
    for item in tqdm(r_list):
        cur_id = int(item['id'])
        old_retrieval_doc = old_result_dict[cur_id]['predicted_docids']
        new_retrieval_doc = item['predicted_docids']
        m_predicted_docids = set.union(set(old_retrieval_doc), set(new_retrieval_doc))
        # print(m_predicted_docids)
        m_predicted_docids = [docid for docid in m_predicted_docids if not docid.startswith('List_of_')]
        item['predicted_docids'] = list(m_predicted_docids)
        # print(item['predicted_docids'])

    mode = {'standard': False, 'check_doc_id_correct': True}
    if tag != 'test':
        fever_scorer.fever_score_analysis(r_list, d_list, mode=mode, max_evidence=None)

    if save:
        print("Saved to:")
        common.save_jsonl(r_list, config.RESULT_PATH /
                          f"doc_retri_results/fever_results/merged_doc_results/m_doc_{tag}.jsonl")

    # States information.
    len_list = []
    for rset in r_list:
        len_list.append(len(rset['predicted_docids']))

    print(collections.Counter(len_list).most_common(10000))

    print(np.mean(len_list))
    print(np.std(len_list))
    print(np.max(len_list))
    print(np.min(len_list))


if __name__ == '__main__':
    # fever_retrieval_v0(tag='test', term_retrieval_top_k=3, match_filtering_k=1)
    #
    merge_results_with_haonao_module(tag='train', term_retrieval_top_k=3, match_filtering_k=2, haonan_topk=10)

    # merge_results_with_haonao_module(term_retrieval_top_k=3, match_filtering_k=2, haonan_topk=10)

    # d_list = common.load_jsonl(config.FEVER_DEV)
    # r_list = common.load_jsonl(config.RESULT_PATH /
    #                                     f"doc_retri_results/fever_results/haonans_results/doc_retr_1_shared_task_dev.jsonl")
    # item_resorting(r_list, top_k=10)

    # print(old_result_list)

    # merge_results_with_haonao_module(tag='dev', save=True)
    # merge_results_with_haonao_module(tag='train', save=True)

    # old_result_list = common.load_jsonl(config.RESULT_PATH /
    #                                     f"doc_retri_results/fever_results/haonans_results/dr_test.jsonl")
    # item_resorting(old_result_list, top_k=5)
    # common.save_jsonl(old_result_list, "n_test_prediction.jsonl")

    # print(fever_scorer.fever_doc_only(old_result_list, d_list, max_evidence=5))

    # print()

    # r_list = common.load_jsonl('dev-debug-3-2.jsonl')
    #
    # old_result_list = common.load_jsonl(
    #     "/Users/yixin/projects/extinguishHotpot/results/doc_retri_results/old_fever_retrieval/doc_retr_1_shared_task_dev.jsonl")
    # # for item in old_results_list:
    # #     print(item['predicted_docids'])
    # old_result_dict = list_dict_data_tool.list_to_dict(old_result_list, 'id')
    #
    # for i, item in enumerate(r_list):
    #     predicted_docids = item['predicted_docids']
    #     modified_docids = []
    #     for docid in predicted_docids:
    #         docid = docid.replace(' ', '_')
    #         docid = reverse_convert_brc(docid)
    #         modified_docids.append(docid)
    #     item['predicted_docids'] = modified_docids
    #     # item['predicted_docids'] = []
    #
    # merged_result_list = []
    # for item in tqdm(r_list):
    #     cur_id = int(item['id'])
    #     old_retrieval_doc = old_result_dict[cur_id]['predicted_docids']
    #     new_retrieval_doc = item['predicted_docids']
    #     m_predicted_docids = set.union(set(old_retrieval_doc), set(new_retrieval_doc))
    #     # print(m_predicted_docids)
    #     m_predicted_docids = [docid for docid in m_predicted_docids if not docid.startswith('List_of_')]
    #     item['predicted_docids'] = list(m_predicted_docids)
    #     # print(item['predicted_docids'])
    #

    # mode = {'standard': False, 'check_doc_id_correct': True}
    # fever_scorer.fever_score(r_list, d_list, mode=mode, max_evidence=None)
    # fever_scorer.fever_doc_only(r_list, d_list, max_evidence=None)
    #
    # # States information.
    # len_list = []
    # for rset in r_list:
    #     len_list.append(len(rset['predicted_docids']))
    #
    # print(collections.Counter(len_list).most_common(10000))
    #
    # print(np.mean(len_list))
    # print(np.std(len_list))
    # print(np.max(len_list))
    # print(np.min(len_list))
