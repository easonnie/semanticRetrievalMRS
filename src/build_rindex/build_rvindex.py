# invdx.py
# An inverted index
import heapq
from pathlib import Path

__author__ = 'Yixin Nie'
import json
import config
from tqdm import tqdm
import math


# 2019 - 02 - 12

# naive tf-idf scoring.
def default_scoring_fn(**kwargs):
    tf = kwargs['tf']
    Nt = kwargs['Nt']
    N = kwargs['N']
    score = math.log(tf + 1) * math.log((N - Nt + 0.5) / (Nt + 0.5))
    return score


def save_to_file(dict_db, file, memory_efficient=False):
    print(f"Save scored file to {file}")
    with open(file, mode='w', encoding='utf-8') as out_f:
        if not memory_efficient:
            for term in tqdm(dict_db.keys()):
                item = {'t': term, 'v': dict_db[term]}
                out_f.write(json.dumps(item) + "\n")
        else:
            for term in tqdm(dict_db):
                term_row = dict_db[term]
                for key, value in term_row.items():
                    out_f.write('/'.join([str(term), str(key), str(value)]) + "\n")


def load_from_file(dict_db, file, with_int_type=False, memory_efficient=False, value_type='int'):
    print(f"Load scored file from {file}")
    with open(file, mode='r', encoding='utf-8') as in_f:
        if not memory_efficient:
            if not with_int_type:
                for line in tqdm(in_f):
                    item = json.loads(line)
                    dict_db[item['t']] = item['v']
            else:
                for line in tqdm(in_f):
                    item = json.loads(line)
                    dict_db[int(item['t'])] = dict()
                    for key, value in item['v'].items():
                        dict_db[int(item['t'])][int(key)] = value
        else:
            for line in tqdm(in_f):
                line = line.strip()
                sp_line = line.split('/')
                term = int(sp_line[0])
                doc = int(sp_line[1])
                if value_type == 'int':
                    value = int(sp_line[2])
                elif value_type == 'float':
                    value = float(sp_line[2])
                if term not in dict_db:
                    dict_db[term] = {doc: value}
                else:
                    dict_db[term][doc] = value
    return dict_db


class IndexDB(object):
    def __init__(self):
        super().__init__()
        self.inverted_index = InvertedIndex()
        self.document_length_table = DocumentLengthTable()
        self.score_db = dict()  # This is a important score_db

    def save_to_file(self, filename: Path, memory_saving=False):
        if not filename.exists():
            filename.mkdir(parents=True, exist_ok=False)
        self.inverted_index.save_to_file(filename / "inverted_index.txt", memory_efficient=memory_saving)
        self.document_length_table.save_to_file(filename / "doc_length_table.txt")
        # score_file_name = filename / "scored_db"
        # if not score_file_name.exists():
        #     score_file_name.mkdir()
        # for key in self.score_db.keys():
        #     save_to_file(self.score_db[key], score_file_name / f"{key}.score.txt")

    def load_from_file(self, filename: Path, with_int_type=False, memory_saving=False):
        if not filename.is_dir():
            raise FileNotFoundError(f"{filename} is not a valid indexDB directory.")
        self.inverted_index.load_from_file(filename / "inverted_index.txt", with_int_type=with_int_type,
                                           memory_efficient=memory_saving)
        self.document_length_table.load_from_file(filename / "doc_length_table.txt", with_int_type=with_int_type)

        # We let the user to retrieve the score file because it is time-consuming.

        # score_file_name = filename / "scored_db"
        # if score_file_name.exists():
        #     for

    def pre_compute_scores(self, scoring_fn=default_scoring_fn, score_name='default-tf-idf'):
        self.score_db[score_name] = dict()

        the_cur_score_db = self.score_db[score_name]

        print(f"Pre computing scores for {score_name}.")
        for cur_term in tqdm(self.inverted_index.index.keys()):
            for cur_doc in self.inverted_index.index[cur_term].keys():
                tf = self.inverted_index.get_tf(cur_term, cur_doc)
                Nt = self.inverted_index.get_Nt(cur_term)
                N = len(self.document_length_table)
                score = scoring_fn(tf=tf, Nt=Nt, N=N)
                if cur_term not in the_cur_score_db:
                    the_cur_score_db[cur_term] = dict()

                the_cur_score_db[cur_term][cur_doc] = score
            # break

    def pre_compute_scores_iteratively(self, save_path, scoring_fn=default_scoring_fn, score_name='default-tf-idf'):
        print(f"Pre computing scores for {score_name}.")
        with open(save_path, mode='w', encoding='utf-8') as out_f:
            for cur_term in tqdm(self.inverted_index.index.keys()):
                for cur_doc in self.inverted_index.index[cur_term].keys():
                    tf = self.inverted_index.get_tf(cur_term, cur_doc)
                    Nt = self.inverted_index.get_Nt(cur_term)
                    N = len(self.document_length_table)
                    score = scoring_fn(tf=tf, Nt=Nt, N=N)
                    out_f.write('/'.join([str(cur_term), str(cur_doc), str(score)]) + "\n")

    def get_relevant_document(self, candidate_doc, valid_terms, score_name='default-tf-idf',
                              scoring_fn=default_scoring_fn, top_k=None):
        scored_doc = []

        for cur_doc in candidate_doc:
            cur_doc_score = 0
            for cur_term in valid_terms:

                if score_name in self.score_db:
                    # print("Found precomputed_value.")
                    if cur_term not in self.score_db[score_name]:
                        score = 0
                    elif cur_doc not in self.score_db[score_name][cur_term]:
                        score = 0
                    else:
                        score = self.score_db[score_name][cur_term][cur_doc]
                else:
                    tf = self.inverted_index.get_tf(cur_term, cur_doc)
                    Nt = self.inverted_index.get_Nt(cur_term)
                    N = len(self.document_length_table)
                    score = scoring_fn(tf=tf, Nt=Nt, N=N)

                cur_doc_score += score

            if top_k is not None and 0 <= top_k == len(scored_doc):
                heapq.heappushpop(scored_doc, (cur_doc_score, cur_doc))
            else:
                heapq.heappush(scored_doc, (cur_doc_score, cur_doc))

        return scored_doc


class InvertedIndex:
    """
    The inverted index is basically a dictionary, with key: term, value {docid: num_of_occurs the terms in the doc}
    """

    def __init__(self):
        self.index = dict()
        self.Nt_table = dict()

    def __contains__(self, item):
        return item in self.index

    def __getitem__(self, item):
        return self.index[item]

    def build_Nt_table(self):
        print("Build Nt table.")
        for term in tqdm(self.index.keys()):
            self.Nt_table[term] = len(self.index[term])

    def add(self, term, docid):
        if term in self.index:
            if docid in self.index[term]:
                self.index[term][docid] += 1
            else:
                self.index[term][docid] = 1
        else:
            d = dict()
            d[docid] = 1
            self.index[term] = d

    def save_to_file(self, file, memory_efficient=False):
        print(f"Save indexing file to {file}")
        with open(file, mode='w', encoding='utf-8') as out_f:
            if not memory_efficient:
                for term in tqdm(self.index):
                    item = {'t': term, 'i': self.index[term]}
                    out_f.write(json.dumps(item) + "\n")
            else:
                for term in tqdm(self.index):
                    term_row = self.index[term]
                    for key, value in term_row.items():
                        out_f.write('/'.join([str(term), str(key), str(value)]) + "\n")

    def load_from_file(self, file, with_int_type=False, memory_efficient=False):
        print(f"Load indexing file from {file}")
        with open(file, mode='r', encoding='utf-8') as in_f:
            if not memory_efficient:
                if not with_int_type:
                    for line in tqdm(in_f):
                        item = json.loads(line)
                        self.index[item['t']] = item['i']
                        del item
                else:
                    for line in tqdm(in_f):
                        item = json.loads(line)
                        self.index[int(item['t'])] = dict()
                        for key, value in item['i'].items():
                            self.index[int(item['t'])][int(key)] = value
                        del item
            else:
                for line in tqdm(in_f):
                    line = line.strip()
                    sp_line = line.split('/')
                    term = int(sp_line[0])
                    doc = int(sp_line[1])
                    value = int(sp_line[2])
                    if term not in self.index:
                        self.index[term] = {doc: value}
                    else:
                        self.index[term][doc] = value


    def get_containing_document(self, term):
        if term not in self.index:
            return None
        else:
            return set(self.index[term].keys())

    def get_tf(self, term, docid):
        if term not in self.index:
            return 0
        elif docid not in self.index[term]:
            return 0
        else:
            return self.index[term][docid]

    def get_Nt(self, term):
        if self.Nt_table is None or len(self.Nt_table) == 0:
            raise Exception("Nt table not built.")
        if term not in self.Nt_table:
            return 0
        else:
            return self.Nt_table[term]

    # frequency of word in document
    # def get_document_frequency(self, word, docid):
    #     if word in self.index:
    #         if docid in self.index[word]:
    #             return self.index[word][docid]
    #         else:
    #             raise LookupError('%s not in document %s' % (str(word), str(docid)))
    #     else:
    #         raise LookupError('%s not in index' % str(word))
    #
    # # frequency of word in index, i.e. number of documents that contain word
    # def get_index_frequency(self, word):
    #     if word in self.index:
    #         return len(self.index[word])
    #     else:
    #         raise LookupError('%s not in index' % word)


class DocumentLengthTable:
    def __init__(self):
        self.table = dict()
        self.total_doc_num = -1

    def __len__(self):
        if self.total_doc_num != -1:
            return self.total_doc_num
        else:
            self.total_doc_num = len(self.table)
            return self.total_doc_num

    def add(self, docid, length):
        self.table[docid] = length

    def get_length(self, docid):
        if docid in self.table:
            return self.table[docid]
        else:
            raise LookupError('%s not found in table' % str(docid))

    def get_average_length(self):
        sum = 0
        for length in self.table.values():
            sum += length
        return float(sum) / float(len(self.table))

    def save_to_file(self, file):
        print(f"Save Document lengths file to {file}")
        with open(file, mode='w', encoding='utf-8') as out_f:
            for title in tqdm(self.table):
                out_f.write(str(title) + '<#.#>' + str(self.table[title]) + '\n')

    def load_from_file(self, file, with_int_type=False):
        print(f"Load Document lengths file from {file}")
        with open(file, mode='r', encoding='utf-8') as in_f:
            if not with_int_type:
                for line in in_f:
                    line = line.strip()
                    title, lens_str = line.split('<#.#>')
                    self.table[title] = int(lens_str)
            else:
                for line in in_f:
                    line = line.strip()
                    title, lens_str = line.split('<#.#>')
                    self.table[int(title)] = int(lens_str)

        self.total_doc_num = len(self.table)


# def build_data_structures(corpus):
#     idx = InvertedIndex()
#     dlt = DocumentLengthTable()
#     for docid in corpus:
#         # build inverted index
#         for word in corpus[docid]:
#             idx.add(str(word), str(docid))
#
#         # build document length table
#         length = len(corpus[str(docid)])
#         dlt.add(docid, length)
#     return idx, dlt

if __name__ == '__main__':
    index_db = IndexDB()
    # index_db.inverted_index.add('abb', 'aff')
    # index_db.inverted_index.add('abv', 'aff')
    # index_db.inverted_index.add('abas', 'afb')
    # index_db.document_length_table.add('aff', 2)
    # index_db.document_length_table.add('afb', 1)
    # index_db.save_to_file(config.PDATA_ROOT / "reverse_indexing/utest_save_index_db")
    index_db.load_from_file(config.PDATA_ROOT / "reverse_indexing/utest_save_index_db")
    print(index_db.inverted_index.index)
    print(index_db.document_length_table.table)
