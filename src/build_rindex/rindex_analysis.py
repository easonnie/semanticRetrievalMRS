import config
from build_rindex.bm25 import get_query_result
from build_rindex.build_rvindex import InvertedIndex, DocumentLengthTable
from flashtext import KeywordProcessor
from tqdm import tqdm
import spacy
# Total term: 57667685

nlp = spacy.load('en')

nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')
nlp.remove_pipe('ner')


def build_processor(idx):
    keyword_processor = KeywordProcessor(case_sensitive=True)
    for word in tqdm(idx.index.keys()):
        keyword_processor.add_keyword(word)
    return keyword_processor


def query_get_terms(query, kw_processor):
    tokenized_query = ' '.join([t.text for t in nlp(query)])
    terms = kw_processor.extract_keywords(tokenized_query)
    return terms


def save_terms(idx: InvertedIndex, file):
    with open(file, mode='w', encoding='utf-8') as out_f:
        for w in idx.index.keys():
            out_f.write(w + '\n')


def load_terms(file):
    terms = set()
    with open(file, mode='r', encoding='utf-8') as in_f:
        for line in in_f:
            terms.add(line.strip())

    return terms


if __name__ == '__main__':
    # idx = InvertedIndex()
    # dlt = DocumentLengthTable()
    #
    # idx.load_from_file(config.PDATA_ROOT / "reverse_indexing/abs_idx.txt")
    # dlt.load_from_file(config.PDATA_ROOT / "reverse_indexing/abs_dlt.txt")
    #
    # # print(len(idx.index))
    # save_terms(idx, config.PDATA_ROOT / "reverse_indexing/terms.txt")
    kw_terms = load_terms(config.PDATA_ROOT / "reverse_indexing/terms.txt")

    keyword_processor = KeywordProcessor(case_sensitive=True)
    for word in tqdm(kw_terms):
        keyword_processor.add_keyword(word)

    kw_processor = keyword_processor
    # exit(0)
    # kw_processor = build_processor(idx)
    query = "What year did the British politician born in 1967 began to represent the Daventry in the UK House of Commons?"
    terms = query_get_terms(query, kw_processor)
    # for term in terms:
    print(terms)
    # dlt_total_length = len(dlt)
    # dlt_avdl = dlt.get_average_length()
    # query_results = get_query_result(terms, idx, dlt, dlt_total_length, dlt_avdl)
    # print(query_results)
    # print(len(query_results))
