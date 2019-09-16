import heapq
import json
import redis
from tqdm import tqdm
from typing import List

import config
from build_rindex.build_rvindex import IndexDB, load_from_file


class RedisScoreIndexOld:
    """
    The inverted index is basically a dictionary, with key: term, value {docid: num_of_occurs the terms in the doc}
    """

    def __init__(self, redis_db: redis.Redis):
        self.redis_db: redis.Redis = redis_db

    def get_containing_document(self, term):
        item = self.redis_db.get(term)
        if item is None:
            return None
        else:
            return json.loads(item).keys()

    def get_score(self, term, docid):
        item = self.redis_db.get(term)
        if item is None:
            return 0
        else:
            item = json.loads(item)
            if docid not in item:
                return 0
            else:
                return item[docid]

    def get_score_item(self, term):
        item = self.redis_db.get(term)
        if item is None:
            return None
        else:
            return json.loads(item)

    def save_scored_index(self, scored_index):
        print("Save scored term-doc index to Redis.")
        for key in tqdm(scored_index.keys()):
            item = scored_index[key]
            self.redis_db.set(key, json.dumps(item))

        self.redis_db.save()


class RedisScoreIndex(object):
    """
    The inverted index is basically a dictionary, with key: term, value {docid: num_of_occurs the terms in the doc}
    """
    TERM_PREFIX = 't'
    SCORE_PREFIX = 's'
    SEP_SYB = ':'

    @staticmethod
    def scored_dict_ranking(candidate_doc_list, scored_dict, top_k):
        scored_doc = []
        v_terms = scored_dict.keys()

        for cur_doc in candidate_doc_list:
            cur_doc_score = 0
            for cur_term in v_terms:
                if cur_doc not in scored_dict[cur_term]:
                    cur_doc_score += 0
                else:
                    cur_doc_score += scored_dict[cur_term][cur_doc]

            if top_k is not None and 0 <= top_k == len(scored_doc):
                heapq.heappushpop(scored_doc, (cur_doc_score, cur_doc))
            else:
                heapq.heappush(scored_doc, (cur_doc_score, cur_doc))

        return scored_doc

    def __init__(self, redis_db: redis.Redis):
        self.redis_db: redis.Redis = redis_db

    def get_containing_document(self, term):
        key = self.TERM_PREFIX + self.SEP_SYB + term
        item = self.redis_db.smembers(key)
        if item is None:
            return None
        else:
            return item

    def get_score(self, term, docid):
        key = self.SEP_SYB.join([self.SCORE_PREFIX, term, docid])
        item = self.redis_db.get(key)
        if item is None:
            return 0
        else:
            return float(item)

    def get_candidate_set_from_batched_terms(self, terms):
        pipe = self.redis_db.pipeline()

        valid_terms = []
        valid_set_list = []

        for term in terms:
            key = self.TERM_PREFIX + self.SEP_SYB + term
            pipe.smembers(key)

        result_set_list = pipe.execute()

        for term, mset in zip(terms, result_set_list):
            if len(mset) > 0:
                valid_terms.append(term)
                valid_set_list.append(mset)

        return list(set.union(*valid_set_list)), valid_set_list, valid_terms

    def get_scores_from_batched_term_doc_pairs(self, terms: List, valid_set_list: List):
        scored_results = dict()

        # Remember order matters:
        for term, mset in zip(terms, valid_set_list):
            pipe = self.redis_db.pipeline()
            for docid in mset:
                key = self.SEP_SYB.join([self.SCORE_PREFIX, term, docid])
                pipe.get(key)
            ritems = pipe.execute()

            scored_results[term] = dict()
            cur_ptr = 0
            for docid in mset:
                scored_results[term][docid] = float(ritems[cur_ptr])
                cur_ptr += 1

        return scored_results

    def save_scored_index(self, scored_index):
        print("Save scored term-doc index to Redis.")
        for term in tqdm(scored_index.keys()):
            pipe = self.redis_db.pipeline()
            item = scored_index[term]
            doc_set = scored_index[term].keys()
            term_key = self.TERM_PREFIX + self.SEP_SYB + term
            for docid, score in item.items():
                score_key = self.SEP_SYB.join([self.SCORE_PREFIX, term, docid])
                pipe.set(score_key, score)
            pipe.sadd(term_key, *doc_set)
            pipe.execute()

        # self.redis_db.save()


def load_tf_idf_score_to_redis_cache():
    tf_idf_score_redis = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_score_index = RedisScoreIndex(tf_idf_score_redis)
    # abs_rindexdb = IndexDB()
    # abs_rindexdb.load_from_file(config.PDATA_ROOT / "reverse_indexing/abs_rindexdb")
    # print("Number of terms:", len(abs_rindexdb.inverted_index.index))
    # abs_rindexdb.inverted_index.build_Nt_table()
    score_db = dict()
    load_from_file(score_db,
                   config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")

    redis_score_index.save_scored_index(score_db)


if __name__ == '__main__':
    # load_tf_idf_score_to_redis_cache()
    tf_idf_score_redis = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_score_index = RedisScoreIndex(tf_idf_score_redis)
    #
    # pipe = tf_idf_score_redis.pipeline()
    # pipe.smembers('t:bansal')
    # pipe.smembers('t:mohit')
    # pipe.smembers('t:&(&(')
    # r = pipe.execute()
    # print(type(r))
    # print(set.union(*r))

    candidate_list, results_set_list, valid_terms = redis_score_index.get_candidate_set_from_batched_terms(['bansal', 'mohit', '&(&('])
    scores_dict = redis_score_index.get_scores_from_batched_term_doc_pairs(valid_terms, results_set_list)
    print(scores_dict)
    print(redis_score_index.scored_dict_ranking(candidate_list, scores_dict, top_k=5))

    print(tf_idf_score_redis.get('s:mohit:Mohit Banerji'))

    # saved_item = {
    #     'a': {'x': 1.0, 'y': 2.0},
    #     'b': {'x': 0.5, 'z': 3.0}
    # }
    #
    # redis_score_index.save_scored_index(saved_item)
    # print(redis_score_index.get_containing_document('a'))
    # print(redis_score_index.get_containing_document('b'))
    # print(redis_score_inde)

    # for i in tqdm(range(100000)):
    # redis_score_index.get_score('a', 'x')
    # redis_score_index.get_containing_document('a')
    # for i in tqdm(range(1000000)):
    # print(redis_score_index.get_containing_document('china'))
    #     a = redis_score_index.get_containing_document('')
    #     print(len(a))
    # for i in tqdm(range(100000)):
    # print(redis_score_index.get_score('china', 'Beijing babbler'))
    # redis_score_index.redis_db.get('china')
    # redis_score_index.get_score_item('china')

    # redis_score_index.redis_db.delete('foo-12345')
    # redis_score_index.redis_db.sadd('foo-1234', 'bar-1', 'bar-12', 'bar-123', 'bar-1234', 'bar-12345',
    #                                 'foo-1', 'foo-12', 'foo-123', 'foo-1234', 'foo-12345', 1)
    # redis_score_index.redis_db.set('foo-12345', 'bar-123456789 bar-123456789 bar-123456789 bar-123456789 bar-123456789')
    # for i in tqdm(range(1000000)):
    # a = redis_score_index.redis_db.get('foo-12345').decode('utf-8')
    # a = redis_score_index.redis_db.get('foo-12345')
    # for _ in range(10000):
    #     a.decode('utf-8')
    # print(a)
    # a = redis_score_index.redis_db.smembers('foo-1234')
    # print(a)
    # for _ in range(1000):
    #     for e in a:
    #         e.decode('utf-8')
    # print(a)
