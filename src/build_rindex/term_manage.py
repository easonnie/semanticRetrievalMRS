import config
from build_rindex.build_rvindex import load_from_file
from tqdm import tqdm


def save_wiki_abstract_terms(save_file):
    g_score_dict = dict()
    g_score_dict = load_from_file(g_score_dict,
                                  config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")
    with open(save_file, encoding='utf-8', mode='w') as out_f:
        for term in tqdm(g_score_dict.keys()):
            out_f.write(term + '\n')


def load_wiki_abstract_terms(save_file):
    abs_terms = []
    with open(save_file, encoding='utf-8', mode='r') as in_f:
        for line in tqdm(in_f):
            abs_terms.append(line.strip())

    return abs_terms


if __name__ == '__main__':
    # save_wiki_abstract_terms(config.PRO_ROOT / "data/processed/wiki_abs_3gram_terms.txt")
    abs_terms = load_wiki_abstract_terms(config.PRO_ROOT / "data/processed/wiki_abs_3gram_terms.txt")
    print(len(abs_terms))

    # g_score_dict = dict()
    # g_score_dict = load_from_file(g_score_dict,
    #                               config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")