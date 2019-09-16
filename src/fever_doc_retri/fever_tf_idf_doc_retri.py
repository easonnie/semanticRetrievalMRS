from functools import partial
from multiprocessing import Pool
import multiprocessing
from pathlib import Path

from build_rindex.build_rvindex import load_from_file
from build_rindex.redis_index import RedisScoreIndex
from build_rindex.rvindex_scoring import get_query_ngrams, get_candidate_page_list, get_ranked_score
from utils import common
import config
from tqdm import tqdm
import redis
import json
import math


tf_idf_score_redis = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
redis_score_index = RedisScoreIndex(tf_idf_score_redis)
# top_k = 20


def process_fever_item_multiprocessing(item, top_k, query_field='claim', id_field='id'):
    # For multiprocess we set some global variable, but be careful and don't give memory intensive global variable.
    # The variable will be copied to other processes
    global redis_score_index

    results_item = dict()
    query = item[query_field]
    qid = str(item[id_field])

    query_ngrams = get_query_ngrams(query)

    candidate_doc_list, valid_set_list, valid_terms = redis_score_index.get_candidate_set_from_batched_terms(query_ngrams)
    # print(candidate_doc_list)
    scored_dict = redis_score_index.get_scores_from_batched_term_doc_pairs(valid_terms, valid_set_list)
    # print(scored_dict)
    doc_list = redis_score_index.scored_dict_ranking(candidate_doc_list, scored_dict, top_k=top_k)

    results_item[query_field] = query
    results_item[id_field] = qid
    results_item['retrieved_list'] = doc_list

    return results_item


def process_fever_item_with_score_dict(item, top_k, global_score_dict, query_field='claim', id_field='id'):
    results_item = dict()
    # question = item['question']
    query = item[query_field]
    qid = str(item[id_field])

    query_ngrams = get_query_ngrams(query)

    v_terms, v_docids, candidate_doc_list = get_candidate_page_list(query_ngrams, global_score_dict)
    # print(candidate_doc_list)
    doc_list = get_ranked_score(v_terms, v_docids, candidate_doc_list, top_k, global_score_dict)
    # print(scored_dict)
    # doc_list = redis_score_index.scored_dict_ranking(candidate_doc_list, scored_dict, top_k=top_k)

    results_item[query_field] = query
    results_item[id_field] = qid
    results_item['retrieved_list'] = doc_list

    return results_item


def single_process_fever_with_dict(start=0, end=None, tag='dev'):
    task_name = 'fever'
    debug = False
    top_k = 20

    query_fieldname = 'claim'
    id_fieldname = 'id'
    debug_name = 'debug' if debug else ""

    g_score_dict = dict()
    g_score_dict = load_from_file(g_score_dict,
                                  config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")

    if tag == 'dev':
        d_list = common.load_jsonl(config.FEVER_DEV)
    elif tag == 'train':
        d_list = common.load_jsonl(config.FEVER_TRAIN)
    elif tag == 'test':
        d_list = common.load_jsonl(config.FEVER_TEST)
    else:
        raise ValueError(f"Tag:{tag} not supported.")

    # Important Set this number !!!
    print("Total length:", len(d_list))
    # start, end = 0, len(d_list)
    # Important End !!!

    print(f"Task:{task_name}, Tag:{tag}, TopK:{top_k}, Start/End:{start}/{end}")
    d_list = d_list[start:end]

    print("Data length:", len(d_list))
    if debug:
        d_list = d_list[:10]
        start, end = 0, 10
    print("Data length (Pos-filtering):", len(d_list))

    r_item_list = []

    incr_file = config.RESULT_PATH / f"doc_retri_results/term_based_methods_results/{task_name}_tf_idf_{tag}_incr_({start},{end})_{debug_name}.jsonl"
    if incr_file.is_file():
        print("Warning save file exists.")

    save_path: Path = config.RESULT_PATH / f"doc_retri_results/term_based_methods_results/{task_name}_tf_idf_{tag}_({start},{end})_{debug_name}.jsonl"
    if save_path.is_file():
        print("Warning save file exists.")

    with open(incr_file, mode='w', encoding='utf-8') as out_f:
        process_func = partial(process_fever_item_with_score_dict,
                               top_k=top_k, query_field=query_fieldname, id_field=id_fieldname,
                               global_score_dict=g_score_dict)

        for item in tqdm(d_list, total=len(d_list)):
            r_item = process_func(item)
            r_item_list.append(r_item)
            out_f.write(json.dumps(item) + '\n')
            out_f.flush()

    print(len(r_item_list))
    common.save_jsonl(r_item_list, save_path)


def multi_process(start=0, end=None, tag='dev'):
    task_name = 'fever'
    debug = False
    top_k = 20
    num_process = 3
    query_fieldname = 'claim'
    id_fieldname = 'id'
    debug_name = 'debug' if debug else ""

    # print(multiprocessing.cpu_count())
    print("CPU Count:", multiprocessing.cpu_count())

    if tag == 'dev':
        d_list = common.load_jsonl(config.FEVER_DEV)
    elif tag == 'train':
        d_list = common.load_jsonl(config.FEVER_TRAIN)
    elif tag == 'test':
        d_list = common.load_jsonl(config.FEVER_TEST)
    else:
        raise ValueError(f"Tag:{tag} not supported.")

    print("Total length:", len(d_list))
    # Important Set this number !!!
    # start, end = 0, None
    # Important End !!!

    print(f"Task:{task_name}, Tag:{tag}, TopK:{top_k}, Start/End:{start}/{end}")
    d_list = d_list[start:end]

    print("Data length:", len(d_list))
    if debug:
        d_list = d_list[:10]
        start, end = 0, 10
    print("Data length (Pos-filtering):", len(d_list))

    r_list = []

    incr_file = config.RESULT_PATH / f"doc_retri_results/term_based_methods_results/{task_name}_tf_idf_{tag}_incr_({start},{end})_{debug_name}.jsonl"
    if incr_file.is_file():
        print("Warning save file exists.")

    save_path: Path = config.RESULT_PATH / f"doc_retri_results/term_based_methods_results/{task_name}_tf_idf_{tag}_({start},{end})_{debug_name}.jsonl"
    if save_path.is_file():
        print("Warning save file exists.")

    with open(incr_file, mode='w', encoding='utf-8') as out_f:
        with Pool(processes=num_process, maxtasksperchild=1000) as pool:

            process_func = partial(process_fever_item_multiprocessing,
                                   top_k=top_k, query_field=query_fieldname, id_field=id_fieldname)

            p_item_list = pool.imap_unordered(process_func, d_list)
            for item in tqdm(p_item_list, total=len(d_list)):
                r_list.append(item)
                out_f.write(json.dumps(item) + '\n')
                out_f.flush()

    print(len(r_list))
    common.save_jsonl(r_list, save_path)


if __name__ == '__main__':
    # single_process_fever_with_dict('dev')
    start = 0
    # end = 70_000
    # start = 70_000
    end = None
    single_process_fever_with_dict(start=start, end=end, tag='test')

    # multi_process(start=5, end=15, tag='train')

    # fever_dev_list = common.load_jsonl(config.FEVER_DEV)
    # print(len(fever_dev_list))
    # for item in fever_dev_list:
    #     print(item)
        # pass
