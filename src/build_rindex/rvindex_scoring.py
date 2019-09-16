import heapq
from functools import partial
from pathlib import Path
from tqdm import tqdm

import spacy

import config
from build_rindex.build_rvindex import IndexDB, save_to_file, load_from_file
from build_rindex.build_wiki_rindex import filter_ngram, get_ngrams

nlp = spacy.load('en')

nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')
nlp.remove_pipe('ner')


def pre_compute_abs_if_idf_scores():
    abs_rindexdb = IndexDB()
    abs_rindexdb.load_from_file(config.PDATA_ROOT / "reverse_indexing/abs_rindexdb")
    print("Number of terms:", len(abs_rindexdb.inverted_index.index))
    abs_rindexdb.inverted_index.build_Nt_table()
    # exit(0)

    abs_rindexdb.pre_compute_scores()
    save_to_file(abs_rindexdb.score_db['default-tf-idf'],
                 config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")


def compute_abs_score(db_path, score_path="scored_db/default-tf-idf.score.txt", with_int_type=False,
                      memory_efficient=False, iteratively=False):
    abs_rindexdb = IndexDB()
    abs_rindexdb.load_from_file(db_path, with_int_type, memory_saving=memory_efficient)
    print("Number of terms:", len(abs_rindexdb.inverted_index.index))
    abs_rindexdb.inverted_index.build_Nt_table()

    if not iteratively:
        abs_rindexdb.pre_compute_scores()
        if not (Path(db_path) / score_path).parent.is_dir():
            (Path(db_path) / score_path).parent.mkdir()

        save_to_file(abs_rindexdb.score_db['default-tf-idf'],
                     Path(db_path) / score_path, memory_efficient=memory_efficient)
    else:
        if not (Path(db_path) / score_path).parent.is_dir():
            (Path(db_path) / score_path).parent.mkdir()
        abs_rindexdb.pre_compute_scores_iteratively(Path(db_path) / score_path)


def get_query_ngrams(query):
    tokens = [t.text for t in nlp(query)]
    query_ngrams = get_ngrams(tokens, None, 3,
                              filter_fn=partial(filter_ngram, mode='any'),
                              included_tags=None)
    return query_ngrams


def get_query_doc_score(query_ngrams, doc, score_dict):
    score = 0
    for term in query_ngrams:
        if term in score_dict:
            if doc in score_dict[term]:
                score += score_dict[term][doc]
    return score


def get_candidate_page_list(terms, score_dict):
    candidate_doc_set = set()
    v_terms = []
    v_docids = []
    for term in terms:
        if term in score_dict:
            v_terms.append(term)
            cur_v_docids = list(score_dict[term].keys())
            v_docids.append(cur_v_docids)
            for v_docid in cur_v_docids:
                candidate_doc_set.add(v_docid)

    return v_terms, v_docids, list(candidate_doc_set)


def get_ranked_score(v_terms, v_docids, candidate_doc_list, top_k, score_dict):
    cached_scored_results = dict()

    # We first access the global dict to cached a local score dict.
    for term, mset in zip(v_terms, v_docids):
        cached_scored_results[term] = dict()
        for docid in mset:
            cached_scored_results[term][docid] = score_dict[term][docid]

    scored_doc = []

    for cur_doc in candidate_doc_list:
        cur_doc_score = 0
        for cur_term in v_terms:
            if cur_doc not in cached_scored_results[cur_term]:
                cur_doc_score += 0
            else:
                cur_doc_score += cached_scored_results[cur_term][cur_doc]

        if top_k is None:
            scored_doc.append((cur_doc_score, cur_doc))
        else:
            if top_k is not None and 0 <= top_k == len(scored_doc):
                heapq.heappushpop(scored_doc, (cur_doc_score, cur_doc))
            else:
                heapq.heappush(scored_doc, (cur_doc_score, cur_doc))

    return scored_doc


def get_top_ranked_tf_idf_doc(query, rindexdb, top_k):
    tokens = [t.text for t in nlp(query)]
    query_ngrams = get_ngrams(tokens, None, 3,
                              filter_fn=partial(filter_ngram, mode='any'),
                              included_tags=None)

    candidate_pages_set = set()
    valid_terms = []
    for q_ngram in query_ngrams:
        candidate_pages = rindexdb.inverted_index.get_containing_document(q_ngram)
        if candidate_pages is not None:
            valid_terms.append(q_ngram)
            candidate_pages_set |= candidate_pages

    doc_list = rindexdb.get_relevant_document(candidate_pages_set, valid_terms, top_k=top_k)

    return doc_list


def sanity_check():
    # pre_compute_abs_if_idf_scores()
    #
    abs_rindexdb = IndexDB()
    abs_rindexdb.load_from_file(config.PDATA_ROOT / "reverse_indexing/abs_rindexdb")
    print("Number of terms:", len(abs_rindexdb.inverted_index.index))
    abs_rindexdb.inverted_index.build_Nt_table()
    abs_rindexdb.score_db['default-tf-idf'] = dict()
    load_from_file(abs_rindexdb.score_db['default-tf-idf'],
                   config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")

    # # exit(0)
    #
    # abs_rindexdb.pre_compute_scores()
    # save_to_file(abs_rindexdb.score_db['default-tf-idf'], config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")

    # exit(0)

    query = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
    tokens = [t.text for t in nlp(query)]
    # poss = [t.text for t in nlp(query)]
    query_ngrams = get_ngrams(tokens, None, 3,
                              filter_fn=partial(filter_ngram, mode='any'),
                              included_tags=None)

    # print(query_ngram)
    candidate_pages_set = set()
    valid_terms = []
    for q_ngram in query_ngrams:
        candidate_pages = abs_rindexdb.inverted_index.get_containing_document(q_ngram)
        if candidate_pages is not None:
            valid_terms.append(q_ngram)
            candidate_pages_set |= candidate_pages

    print('Animorphs' in candidate_pages_set)
    print(abs_rindexdb.get_relevant_document(['Animorphs'], valid_terms))
    doc_list = abs_rindexdb.get_relevant_document(candidate_pages_set, valid_terms, top_k=100)

    # print(candidate_pages_set)
    print(query_ngrams)
    print(len(candidate_pages_set))
    print(doc_list)

    # 902_396


if __name__ == '__main__':
    # compute_abs_score(db_path=config.PDATA_ROOT / "reverse_indexing/wiki_p_level_unigram_rindexdb")
    # compute_abs_score(db_path=config.PDATA_ROOT / "reverse_indexing/wiki_p_level_unigram_rindexdb_hash_size_limited",
    #                   with_int_type=True, memory_efficient=True)

    compute_abs_score(db_path=config.PDATA_ROOT / "reverse_indexing/wiki_p_level_unigram_rindexdb",
                      with_int_type=True, memory_efficient=True, iteratively=True)

    # 937642696

    # g_score_dict = dict()
    # load_from_file(g_score_dict, config.PDATA_ROOT /
    #                "reverse_indexing/wiki_p_level_unigram_rindexdb/scored_db/default-tf-idf.score.txt",
    #                memory_efficient=True, value_type='float')
    # print(g_score_dict.keys())
    # print(g_score_dict[4175403251])
    # abs_rindexdb = IndexDB()
    # abs_rindexdb.load_from_file(config.PDATA_ROOT / "reverse_indexing/abs_rindexdb")
    # print("Number of terms:", len(abs_rindexdb.inverted_index.index))
    # abs_rindexdb.inverted_index.build_Nt_table()
    # abs_rindexdb.score_db['default-tf-idf'] = dict()
    # load_from_file(abs_rindexdb.score_db['default-tf-idf'],
    #                config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")
    #
    # query = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
    # doc_list = get_top_ranked_tf_idf_doc(query, abs_rindexdb, top_k=10)
    # print(doc_list)

