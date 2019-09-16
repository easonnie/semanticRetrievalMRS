from build_rindex.build_rvindex import InvertedIndex, DocumentLengthTable
from hotpot_doc_retri.hotpot_preliminary_doc_retri import STOPWORDS
from inspect_wikidump.inspect_whole_file import get_first_paragraph_index
from utils import common
import config
from sqlitedict import SqliteDict
import json
from nltk import ngrams
import spacy
from tqdm import tqdm
from wiki_util import wiki_db_tool

nlp = spacy.load('en')

nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')
nlp.remove_pipe('ner')


def get_ngrams(terms, poss=None, n=1, included_tags=None, as_strings=True):
    """Returns a list of all ngrams from length 1 to n.
    """
    ngrams = [(s, e + 1)
              for s in range(len(terms))
              for e in range(s, min(s + n, len(terms)))]

    if poss is not None and included_tags is not None:  # We do filtering according to pos.
        # ngrampos = [(s, e + 1)
        #             for s in range(len(poss))
        #             for e in range(s, min(s + n, len(poss)))]

        filtered_ngram = []
        for (s, e) in ngrams:
            if any([poss[i] in included_tags for i in range(s, e)]):
                filtered_ngram.append((s, e))

        ngrams = filtered_ngram

    # Concatenate into strings
    if as_strings:
        ngrams = ['{}'.format(' '.join(terms[s:e])) for (s, e) in ngrams]

    return ngrams


# Open class words	Closed class words	Other
# ADJ	            ADP	                PUNCT
# ADV	            AUX	                SYM
# INTJ	            CCONJ	            X
# NOUN	            DET
# PROPN	            NUM
# VERB	            PART
#                   PRON
#                   SCONJ

POS_INCLUDED = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB']


# POS_MULTIGRAM = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']


# def get_term(title, item, gram_count=(1, 2, 3)):
#     terms = []
#     tokenized_title = ' '.join([t.text for t in nlp(title)])
#     terms.append(tokenized_title)
#     t_list = []
#     for token, pos in zip(item['tokens'], item['poss']):
#         t_list.append((token, pos))
#
#     for gram in gram_count:
#         for ngram_item in ngrams(t_list, n=gram):
#             if gram == 1:
#                 token, pos = ngram_item[0]
#                 if pos in POS_INCLUDED and (token not in STOPWORDS):
#                     terms.append(token)
#             else:
#                 tokens = []
#                 hit_pos = 0
#                 skip = False
#                 for token, pos in ngram_item:
#                     if pos == 'PUNCT':
#                         skip = True
#                         break
#
#                     if pos in POS_INCLUDED and token not in STOPWORDS:
#                         hit_pos += 1
#                     tokens.append(token)
#                 if hit_pos >= 2 and not skip:
#                     terms.append(' '.join(tokens))
#
#     return terms


# def analysis_doc_terms(debug=None):
#     count = 0
#
#     idx = InvertedIndex()
#     dlt = DocumentLengthTable()
#     dlt_term = DocumentLengthTable()
#
#     with SqliteDict(str(config.ABS_PROCESS_FOR_RINDEX_DB), flag='r', encode=json.dumps, decode=json.loads) as abs_db:
#         for title, item in tqdm(abs_db.iteritems()):
#             count += 1
#
#             # print(k)
#             # print(item.keys())
#             # # print(item)
#             #
#             # t_list = []
#             # for token, pos in zip(item['tokens'], item['poss']):
#             #     t_list.append((token, pos))
#             #
#             # # print(t_list)
#             #
#             # for ngram_item in ngrams(t_list, n=1):
#             #     print(ngram_item)
#             terms = get_term(title, item)
#
#             for term in terms:
#                 idx.add(term, title)
#
#             # build document length table
#             length = len(item['tokens'])
#             dlt.add(title, length)
#             dlt_term.add(title, len(terms))
#
#             if debug is not None and count == debug:
#                 break
#
#         print(len(idx.index))
#         print(len(dlt.table))
#
#         idx.save_to_file(config.PDATA_ROOT / "reverse_indexing/abs_idx.txt")
#         dlt.save_to_file(config.PDATA_ROOT / "reverse_indexing/abs_dlt.txt")
#         dlt_term.save_to_file(config.PDATA_ROOT / "reverse_indexing/abs_term_dlt.txt")
#
#         # idx = InvertedIndex()
#         # dlt = DocumentLengthTable()
#         #
#         # idx.load_from_file(config.PDATA_ROOT / "reverse_indexing/utest_idx.txt")
#         # dlt.load_from_file(config.PDATA_ROOT / "reverse_indexing/utest_dlt.txt")
#         #
#         # print(len(idx.index))
#         # print(len(dlt.table))
#
#         abs_db.close()


# def item_to_terms(title_terms, content_terms, title_ngram=3, content_ngram=2,
#                   title_included_tags=None, content_):
#     """
#     Importance: When building title terms matching, we don't delete any words and we use all the abstract to do word count.
#     """
#     title_ngram_terms = get_ngrams(title_terms, n=title_ngram)
#     abs_ngram_terms = get_ngrams(content_terms, n=content_ngram)
#     return title_ngram_terms, abs_ngram_terms
# def raw_terms_filtering(terms, ngram, included_tags):
#     get_ngrams(terms, ngram, included_tags)


def whole_wiki_pages_analysis():
    whole_tokenized_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    whole_tokenized_db_cursor.execute("SELECT * from unnamed")

    with SqliteDict(str(config.WHOLE_WIKI_DB), flag='r', encode=json.dumps, decode=json.loads) as whole_wiki_db:
        for key, value in whole_tokenized_db_cursor:
            valid_page = True
            item = json.loads(value)
            # print(item)
            article_title = item['title']
            article_clean_text = item['clean_text']
            article_poss = item['poss']

            abs_index = get_first_paragraph_index(whole_wiki_db[article_title])

            if abs_index == -1:
                valid_page = False
                # print(whole_wiki_db[article_title])
                # This pages is not valid.

            article_term_list = []
            article_poss_list = []

            title_term_list = []
            title_poss_list = []

            abstract_term_list = []
            abstract_poss_list = []

            assert len(article_clean_text) == len(article_poss)

            for p_i, (paragraph_text, paragraph_poss) in enumerate(zip(article_clean_text, article_poss)):
                for sent_text, sent_poss in zip(paragraph_text, paragraph_poss):
                    if p_i == 0:  # In title.
                        title_term_list.extend(sent_text)
                        title_poss_list.extend(sent_poss)
                        continue  # If the terms are in title, we don't those terms in abstract and article term.
                    else:
                        if p_i == abs_index:  # If the terms are in abstract
                            abstract_term_list.extend(sent_text)
                            abstract_poss_list.extend(sent_poss)

                        article_term_list.extend(sent_text)
                        article_poss_list.extend(sent_poss)

            print("Title:", title_term_list, title_poss_list)

            print("Title:(ngram):", get_ngrams(title_term_list, title_poss_list, 3, included_tags=POS_INCLUDED))

            # print(abstract_term_list, abstract_poss_list)
            # print(article_term_list, article_poss_list)


if __name__ == '__main__':
    whole_wiki_pages_analysis()
    # pass
    # print(get_ngrams(['abs', 'fwef', 'fwe'], n=3))
    # analysis_doc_terms(debug=None)
    # get_term('HIS', )
