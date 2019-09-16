from flashtext import KeywordProcessor

from build_rindex.rindex_analysis import query_get_terms, load_terms
from utils import common
import config
from tqdm import tqdm


def get_kwterm_matching(kw_terms, d_list, chuck_size=10_000_000):
    kw_terms = list(kw_terms)
    kw_terms_total_size = len(kw_terms)
    for start in range(0, kw_terms_total_size, chuck_size):
        print(start, start + chuck_size)
        current_kw_terms = kw_terms[start:start + chuck_size]
        keyword_processor = KeywordProcessor(case_sensitive=True)
        for word in tqdm(current_kw_terms):
            keyword_processor.add_keyword(word)

        for item in tqdm(d_list):
            query = item['question']
            terms = query_get_terms(query, keyword_processor)
            if 'kw_matches' not in item:
                item['kw_matches'] = []
            item['kw_matches'].extend(terms)

        del keyword_processor

    return d_list


if __name__ == '__main__':
    kw_terms = load_terms(config.PDATA_ROOT / "reverse_indexing/terms.txt")
    # d_list = common.load_json(config.DEV_FULLWIKI_FILE)
    d_list = common.load_json(config.TRAIN_FILE)
    d_list = get_kwterm_matching(kw_terms, d_list)
    # common.save_jsonl(d_list, config.RESULT_PATH / "kw_term_match_result/dev_term_match_result.jsonl")
    common.save_jsonl(d_list, config.RESULT_PATH / "kw_term_match_result/train_term_match_result.jsonl")
