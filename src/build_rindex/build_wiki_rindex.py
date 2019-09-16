import unicodedata
from functools import partial

import regex
from sqlitedict import SqliteDict
import json
import config
from build_rindex.build_rvindex import IndexDB, load_from_file
# from doc_retri.hotpot_preliminary_doc_retri import filter_word
from build_rindex.persistent_index_db import IndexingDB
from build_rindex.term_manage import load_wiki_abstract_terms
from inspect_wikidump.init_inspect import TOTAL_NUM_DOC
from inspect_wikidump.inspect_whole_file import get_first_paragraph_index
from wiki_util import wiki_db_tool
from tqdm import tqdm
from typing import Dict, Tuple, List
from sklearn.utils import murmurhash3_32


POS_INCLUDED = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB']

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


def filter_ngram(gram, mode='any'):
    """Decide whether to keep or discard an n-gram.
    Args:
        gram: list of tokens (length N)
        mode: Option to throw out ngram if
          'any': any single token passes filter_word
          'all': all tokens pass filter_word
          'ends': book-ended by filterable tokens
    """
    filtered = [filter_word(w) for w in gram]
    if mode == 'any':
        return any(filtered)
    elif mode == 'all':
        return all(filtered)
    elif mode == 'ends':
        return filtered[0] or filtered[-1]
    else:
        raise ValueError('Invalid mode: %s' % mode)


def get_ngrams(terms, poss=None, n=1, filter_fn=None, included_tags=None, as_strings=True, lower=True):
    """Returns a list of all ngrams from length 1 to n.
    """

    def _skip(gram):
        if not filter_fn:
            return False
        return filter_fn(gram)

    ngrams = [(s, e + 1)
              for s in range(len(terms))
              for e in range(s, min(s + n, len(terms)))
              if not _skip(terms[s:e + 1])]

    if poss is not None and included_tags is not None:  # We do filtering according to pos.
        filtered_ngram = []
        for (s, e) in ngrams:
            if any([poss[i] in included_tags for i in range(s, e)]):
                filtered_ngram.append((s, e))

        ngrams = filtered_ngram

    # Concatenate into strings
    if as_strings:
        r_list = []
        for (s, e) in ngrams:
            if lower:
                r_list.append(' '.join(terms[s:e]).lower())
            else:
                r_list.append(' '.join(terms[s:e]))

        return r_list

    else:
        return ngrams


def whole_wiki_pages_title_raw_indexing():
    whole_tokenized_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    whole_tokenized_db_cursor.execute("SELECT * from unnamed")

    title_abs_raw_indexdb = IndexDB()
    abs_file_name = config.PDATA_ROOT / "reverse_indexing/abs_rindexdb"

    content_indexdb = IndexDB()
    content_index_file_name = ''

    with SqliteDict(str(config.WHOLE_WIKI_DB), flag='r', encode=json.dumps, decode=json.loads) as whole_wiki_db:
        for key, value in tqdm(whole_tokenized_db_cursor, total=TOTAL_NUM_DOC):
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

            # print("Title:", title_term_list, title_poss_list)

            title_ngram = get_ngrams(title_term_list, title_poss_list, 3,
                                     filter_fn=partial(filter_ngram, mode='any'),
                                     included_tags=POS_INCLUDED)

            abs_ngram = get_ngrams(abstract_term_list, abstract_poss_list, 3,
                                   filter_fn=partial(filter_ngram, mode='any'),
                                   included_tags=POS_INCLUDED)

            # print(article_title)
            # print(title_ngram)
            # print(abs_ngram)

            added_terms_num = 0
            for added_term in title_ngram + abs_ngram:
                title_abs_raw_indexdb.inverted_index.add(added_term, article_title)
                added_terms_num += 1

            title_abs_raw_indexdb.document_length_table.add(article_title, added_terms_num)
            # break

        #     content_t_ngram = get_ngrams(title_term_list, title_poss_list, 3,
        #                                  filter_fn=partial(filter_ngram, mode='any'),
        #                                  included_tags=POS_INCLUDED)
        #
        #     content_c_ngram = get_ngrams(abstract_term_list, abstract_poss_list, 3,
        #                                  filter_fn=partial(filter_ngram, mode='any'),
        #                                  included_tags=POS_INCLUDED)
        #
        #     added_terms_num = 0
        #     for added_term in content_t_ngram + content_c_ngram:
        #         content_indexdb.inverted_index.add(added_term, article_title)
        #         added_terms_num += 1
        #
        #     content_indexdb.document_length_table.add(article_title, added_terms_num)
        #
        title_abs_raw_indexdb.save_to_file(abs_file_name)
        # print(title_term_list)
        # print(title_ngram)
        # print(abs_ngram)

        # print("Title:(ngram):", get_ngrams(title_term_list, title_poss_list, 3, included_tags=POS_INCLUDED))

        # print(abstract_term_list, abstract_poss_list)
        # print(article_term_list, article_poss_list)


def whole_wiki_pages_title_raw_indexing_paragraph_level(limited_terms=True):
    key_separator = '/'
    whole_tokenized_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    whole_tokenized_db_cursor.execute("SELECT * from unnamed")

    wiki_p_level_indexdb = IndexDB()
    file_name = config.PDATA_ROOT / "reverse_indexing/wiki_p_level_limited_gram_rindexdb"

    count = 0

    if limited_terms:
        limited_terms_set = load_wiki_abstract_terms(config.PRO_ROOT / "data/processed/wiki_abs_3gram_terms.txt")
    else:
        limited_terms_set = []

    limited_terms_set = set(limited_terms_set)

    for key, value in tqdm(whole_tokenized_db_cursor, total=TOTAL_NUM_DOC):
        item = json.loads(value)
        article_title = item['title']
        article_clean_text = item['clean_text']
        article_poss = item['poss']

        title_term_list = []
        title_poss_list = []

        title_ngram = None

        assert len(article_clean_text) == len(article_poss)

        for p_i, (paragraph_text, paragraph_poss) in enumerate(zip(article_clean_text, article_poss)):
            paragraph_term_list = []
            paragraph_poss_list = []
            for sent_text, sent_poss in zip(paragraph_text, paragraph_poss):
                if p_i == 0:  # In title.
                    title_term_list.extend(sent_text)
                    title_poss_list.extend(sent_poss)
                    continue  # If the terms are in title, we don't those terms in abstract and article term.
                else:  # p_i != 0
                    paragraph_term_list.extend(sent_text)
                    paragraph_poss_list.extend(sent_poss)

            if p_i == 0 and title_ngram is None:
                title_ngram = get_ngrams(title_term_list, title_poss_list, 2,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)

            if p_i >= 100:
                break

            paragraph_ngram = get_ngrams(paragraph_term_list, paragraph_poss_list, 2,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)

            if len(paragraph_ngram) == 0:
                continue

            added_terms_num = 0

            paragraph_key = key_separator.join((article_title, str(p_i)))

            for added_term in title_ngram + paragraph_ngram:
                if added_term in limited_terms_set:
                    wiki_p_level_indexdb.inverted_index.add(added_term, paragraph_key)
                    added_terms_num += 1
                elif ' ' not in added_term:
                    wiki_p_level_indexdb.inverted_index.add(added_term, paragraph_key)
                    added_terms_num += 1
                else:
                    pass

            wiki_p_level_indexdb.document_length_table.add(paragraph_key, added_terms_num)

            count += 1

        # if count >= 1000:
        #     break

    wiki_p_level_indexdb.save_to_file(file_name)


def whole_wiki_pages_title_raw_indexing_paragraph_level_unigram():
    key_separator = '/'
    whole_tokenized_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    whole_tokenized_db_cursor.execute("SELECT * from unnamed")

    wiki_p_level_indexdb = IndexDB()
    file_name = config.PDATA_ROOT / "reverse_indexing/wiki_p_level_unigram_rindexdb"

    count = 0

    # if limited_terms:
    #     limited_terms_set = load_wiki_abstract_terms(config.PRO_ROOT / "data/processed/wiki_abs_3gram_terms.txt")
    # else:
    #     limited_terms_set = []
    #
    # limited_terms_set = set(limited_terms_set)

    for key, value in tqdm(whole_tokenized_db_cursor, total=TOTAL_NUM_DOC):
        item = json.loads(value)
        article_title = item['title']
        article_clean_text = item['clean_text']
        article_poss = item['poss']

        title_term_list = []
        title_poss_list = []

        title_ngram = None

        assert len(article_clean_text) == len(article_poss)

        for p_i, (paragraph_text, paragraph_poss) in enumerate(zip(article_clean_text, article_poss)):
            paragraph_term_list = []
            paragraph_poss_list = []
            for sent_text, sent_poss in zip(paragraph_text, paragraph_poss):
                if p_i == 0:  # In title.
                    title_term_list.extend(sent_text)
                    title_poss_list.extend(sent_poss)
                    continue  # If the terms are in title, we don't those terms in abstract and article term.
                else:  # p_i != 0
                    paragraph_term_list.extend(sent_text)
                    paragraph_poss_list.extend(sent_poss)

            if p_i == 0 and title_ngram is None:
                title_ngram = get_ngrams(title_term_list, title_poss_list, 1,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)

            if p_i >= 100:
                break

            paragraph_ngram = get_ngrams(paragraph_term_list, paragraph_poss_list, 1,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)

            if len(paragraph_ngram) == 0:
                continue

            added_terms_num = 0

            paragraph_key = key_separator.join((article_title, str(p_i)))

            for added_term in title_ngram + paragraph_ngram:
                # if added_term in limited_terms_set:
                #     wiki_p_level_indexdb.inverted_index.add(added_term, paragraph_key)
                #     added_terms_num += 1
                # elif ' ' not in added_term:
                wiki_p_level_indexdb.inverted_index.add(added_term, paragraph_key)
                added_terms_num += 1
                # else:
                #     pass

            wiki_p_level_indexdb.document_length_table.add(paragraph_key, added_terms_num)

            count += 1

        if count >= 1000:
            break

    wiki_p_level_indexdb.save_to_file(file_name)


def whole_wiki_pages_title_raw_indexing_paragraph_level_unigram_size_limited(hash_size=2**24):
    key_separator = '/'
    whole_tokenized_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    whole_tokenized_db_cursor.execute("SELECT * from unnamed")

    wiki_p_level_indexdb = IndexDB()
    file_name = config.PDATA_ROOT / "reverse_indexing/wiki_p_level_unigram_rindexdb_hash_size_limited"

    count = 0
    # if limited_terms:
    #     limited_terms_set = load_wiki_abstract_terms(config.PRO_ROOT / "data/processed/wiki_abs_3gram_terms.txt")
    # else:
    #     limited_terms_set = []
    #
    # limited_terms_set = set(limited_terms_set)

    for key, value in tqdm(whole_tokenized_db_cursor, total=TOTAL_NUM_DOC):
        item = json.loads(value)
        article_title = item['title']
        article_clean_text = item['clean_text']
        article_poss = item['poss']

        title_term_list = []
        title_poss_list = []

        title_ngram = None

        assert len(article_clean_text) == len(article_poss)

        for p_i, (paragraph_text, paragraph_poss) in enumerate(zip(article_clean_text, article_poss)):
            paragraph_term_list = []
            paragraph_poss_list = []
            for sent_text, sent_poss in zip(paragraph_text, paragraph_poss):
                if p_i == 0:  # In title.
                    title_term_list.extend(sent_text)
                    title_poss_list.extend(sent_poss)
                    continue  # If the terms are in title, we don't those terms in abstract and article term.
                else:  # p_i != 0
                    paragraph_term_list.extend(sent_text)
                    paragraph_poss_list.extend(sent_poss)

            if p_i == 0 and title_ngram is None:
                title_ngram = get_ngrams(title_term_list, title_poss_list, 2,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)

            if p_i >= 100:
                break

            paragraph_ngram = get_ngrams(paragraph_term_list, paragraph_poss_list, 2,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)

            if len(paragraph_ngram) == 0:
                continue

            added_terms_num = 0

            paragraph_key = key_separator.join((article_title, str(p_i)))

            for added_term in title_ngram + paragraph_ngram:
                # if added_term in limited_terms_set:
                #     wiki_p_level_indexdb.inverted_index.add(added_term, paragraph_key)
                #     added_terms_num += 1
                # elif ' ' not in added_term:
                hash_value_added_term = hash(added_term, hash_size)
                hash_value_paragraph_key = hash(paragraph_key)
                wiki_p_level_indexdb.inverted_index.add(hash_value_added_term, hash_value_paragraph_key)
                added_terms_num += 1
                # else:
                #     pass

            hash_value_paragraph_key = hash(paragraph_key)
            wiki_p_level_indexdb.document_length_table.add(hash_value_paragraph_key, added_terms_num)

            count += 1

        if count >= 1000:
            break

    wiki_p_level_indexdb.save_to_file(file_name)




def whole_wiki_pages_title_raw_indexing_paragraph_level_unigram_size_limited_memory_saving():
    key_separator = '/'
    whole_tokenized_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    whole_tokenized_db_cursor.execute("SELECT * from unnamed")

    wiki_p_level_indexdb = IndexDB()
    file_name = config.PDATA_ROOT / "reverse_indexing/wiki_p_level_unigram_rindexdb"

    count = 0
    # if limited_terms:
    #     limited_terms_set = load_wiki_abstract_terms(config.PRO_ROOT / "data/processed/wiki_abs_3gram_terms.txt")
    # else:
    #     limited_terms_set = []
    #
    # limited_terms_set = set(limited_terms_set)

    for key, value in tqdm(whole_tokenized_db_cursor, total=TOTAL_NUM_DOC):
        item = json.loads(value)
        article_title = item['title']
        article_clean_text = item['clean_text']
        article_poss = item['poss']

        title_term_list = []
        title_poss_list = []

        title_ngram = None

        assert len(article_clean_text) == len(article_poss)

        for p_i, (paragraph_text, paragraph_poss) in enumerate(zip(article_clean_text, article_poss)):
            paragraph_term_list = []
            paragraph_poss_list = []
            for sent_text, sent_poss in zip(paragraph_text, paragraph_poss):
                if p_i == 0:  # In title.
                    title_term_list.extend(sent_text)
                    title_poss_list.extend(sent_poss)
                    continue  # If the terms are in title, we don't those terms in abstract and article term.
                else:  # p_i != 0
                    paragraph_term_list.extend(sent_text)
                    paragraph_poss_list.extend(sent_poss)

            if p_i == 0 and title_ngram is None:
                title_ngram = get_ngrams(title_term_list, title_poss_list, 1,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)

            if p_i >= 100:
                break

            paragraph_ngram = get_ngrams(paragraph_term_list, paragraph_poss_list, 1,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)

            if len(paragraph_ngram) == 0:
                continue

            added_terms_num = 0

            paragraph_key = key_separator.join((article_title, str(p_i)))

            for added_term in title_ngram + paragraph_ngram:
                # if added_term in limited_terms_set:
                #     wiki_p_level_indexdb.inverted_index.add(added_term, paragraph_key)
                #     added_terms_num += 1
                # elif ' ' not in added_term:
                hash_value_added_term = hash(added_term)
                hash_value_paragraph_key = hash(paragraph_key)
                wiki_p_level_indexdb.inverted_index.add(hash_value_added_term, hash_value_paragraph_key)
                added_terms_num += 1
                # else:
                #     pass

            hash_value_paragraph_key = hash(paragraph_key)
            wiki_p_level_indexdb.document_length_table.add(hash_value_paragraph_key, added_terms_num)

            count += 1

        # if count >= 1000:
        #     break

    wiki_p_level_indexdb.save_to_file(file_name, memory_saving=True)


def whole_wiki_pages_title_raw_indexing_paragraph_level_to_indexdb():
    key_separator = '/'
    whole_tokenized_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    whole_tokenized_db_cursor.execute("SELECT * from unnamed")

    # wiki_p_level_indexdb = IndexDB()
    file_name = config.PDATA_ROOT / "reverse_indexing/wiki_p_level_persistent_indexdb.db"
    index_db = IndexingDB(file_name)
    index_db.create_tables()

    count = 0

    term_title_items_buffer_list: List[Tuple[str, str, int]] = []
    title_items_buffer_list: List[Tuple[str, int]] = []

    for key, value in tqdm(whole_tokenized_db_cursor, total=TOTAL_NUM_DOC):
        item = json.loads(value)
        article_title = item['title']
        article_clean_text = item['clean_text']
        article_poss = item['poss']

        title_term_list = []
        title_poss_list = []

        title_ngram = None

        assert len(article_clean_text) == len(article_poss)

        paragraph_term_title_dict: Dict[Tuple[str, str], int] = dict()
        paragraph_title_dict: Dict[str, int] = dict()

        for p_i, (paragraph_text, paragraph_poss) in enumerate(zip(article_clean_text, article_poss)):
            paragraph_term_list = []
            paragraph_poss_list = []
            for sent_text, sent_poss in zip(paragraph_text, paragraph_poss):
                if p_i == 0:  # In title.
                    title_term_list.extend(sent_text)
                    title_poss_list.extend(sent_poss)
                    continue  # If the terms are in title, we don't those terms in abstract and article term.
                else:  # p_i != 0
                    paragraph_term_list.extend(sent_text)
                    paragraph_poss_list.extend(sent_poss)

            if p_i == 0 and title_ngram is None:
                title_ngram = get_ngrams(title_term_list, title_poss_list, 1,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)
                continue

            paragraph_ngram = get_ngrams(paragraph_term_list, paragraph_poss_list, 1,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)

            if len(paragraph_ngram) == 0:
                continue

            added_terms_num = 0

            paragraph_key = key_separator.join((article_title, str(p_i)))

            for added_term in title_ngram + paragraph_ngram:
                paragraph_term_title_dict[(added_term, paragraph_key)] = \
                    paragraph_term_title_dict.get((added_term, paragraph_key), 0) + 1
                added_terms_num += 1

            paragraph_title_dict[paragraph_key] = added_terms_num
            count += 1

            if p_i >= 60:
                break

        if count >= 5000:
            break

        for (term, paragraph_key), ovalue in paragraph_term_title_dict.items():
            term_title_items_buffer_list.append((term, paragraph_key, ovalue))

        for paragraph_title, ovalue in paragraph_title_dict.items():
            title_items_buffer_list.append((paragraph_title, ovalue))

        if len(term_title_items_buffer_list) >= 1000:   # Flush
            index_db.insert_many_items(term_title_items_buffer_list)
            index_db.insert_many_articles(title_items_buffer_list)
            term_title_items_buffer_list = []
            title_items_buffer_list = []

    index_db.insert_many_items(term_title_items_buffer_list)
    index_db.insert_many_articles(title_items_buffer_list)
    index_db.close()


def whole_wiki_pages_title_raw_indexing_article_level_to_indexdb():
    whole_tokenized_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    whole_tokenized_db_cursor.execute("SELECT * from unnamed")

    # wiki_p_level_indexdb = IndexDB()
    file_name = config.PDATA_ROOT / "reverse_indexing/wiki_a_level_persistent_indexdb.db"
    index_db = IndexingDB(file_name)
    index_db.create_tables()

    count = 0

    term_title_items_buffer_list: List[Tuple[str, str, int]] = []
    title_items_buffer_list: List[Tuple[str, int]] = []

    for key, value in tqdm(whole_tokenized_db_cursor, total=TOTAL_NUM_DOC):
        article_term_title_dict: Dict[Tuple[str, str], int] = dict()
        article_title_dict: Dict[str, int] = dict()

        item = json.loads(value)
        article_title = item['title']
        article_clean_text = item['clean_text']
        article_poss = item['poss']

        title_term_list = []
        title_poss_list = []

        title_ngram = None

        article_ngram = []

        assert len(article_clean_text) == len(article_poss)

        for p_i, (paragraph_text, paragraph_poss) in enumerate(zip(article_clean_text, article_poss)):
            paragraph_term_list = []
            paragraph_poss_list = []
            for sent_text, sent_poss in zip(paragraph_text, paragraph_poss):
                if p_i == 0:  # In title.
                    title_term_list.extend(sent_text)
                    title_poss_list.extend(sent_poss)
                    continue  # If the terms are in title, we don't those terms in abstract and article term.
                else:  # p_i != 0
                    paragraph_term_list.extend(sent_text)
                    paragraph_poss_list.extend(sent_poss)

            if p_i == 0 and title_ngram is None:
                title_ngram = get_ngrams(title_term_list, title_poss_list, 2,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)
                continue

            paragraph_ngram = get_ngrams(paragraph_term_list, paragraph_poss_list, 2,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)

            if len(paragraph_ngram) == 0:
                continue

            article_ngram.extend(paragraph_ngram)

            if p_i >= 60:
                break

        added_terms_num = 0

        for added_term in title_ngram + article_ngram:
            article_term_title_dict[(added_term, article_title)] = \
                article_term_title_dict.get((added_term, article_title), 0) + 1
            added_terms_num += 1

        article_title_dict[article_title] = added_terms_num
        count += 1

        if count >= 200:
            break

        for (term, article_title), ovalue in article_term_title_dict.items():
            term_title_items_buffer_list.append((term, article_title, ovalue))

        for article_title, ovalue in article_title_dict.items():
            title_items_buffer_list.append((article_title, ovalue))

        if len(term_title_items_buffer_list) >= 1000:   # Flush
            index_db.insert_many_items(term_title_items_buffer_list)
            index_db.insert_many_articles(title_items_buffer_list)
            term_title_items_buffer_list = []
            title_items_buffer_list = []

    index_db.insert_many_items(term_title_items_buffer_list)
    index_db.insert_many_articles(title_items_buffer_list)
    index_db.close()


def whole_wiki_pages_title_raw_indexing_article_level(limited_terms=True):
    whole_tokenized_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    whole_tokenized_db_cursor.execute("SELECT * from unnamed")

    wiki_p_level_indexdb = IndexDB()
    file_name = config.PDATA_ROOT / "reverse_indexing/wiki_a_level_limited_gram_rindexdb"

    if limited_terms:
        limited_terms_set = load_wiki_abstract_terms(config.PRO_ROOT / "data/processed/wiki_abs_3gram_terms.txt")
    else:
        limited_terms_set = []

    limited_terms_set = set(limited_terms_set)

    count = 0

    for key, value in tqdm(whole_tokenized_db_cursor, total=TOTAL_NUM_DOC):
        item = json.loads(value)
        article_title = item['title']
        article_clean_text = item['clean_text']
        article_poss = item['poss']

        title_term_list = []
        title_poss_list = []

        title_ngram = None

        assert len(article_clean_text) == len(article_poss)

        # article_term_list = []
        # article_poss_list = []
        article_ngram = []

        for p_i, (paragraph_text, paragraph_poss) in enumerate(zip(article_clean_text, article_poss)):
            paragraph_term_list = []
            paragraph_poss_list = []
            for sent_text, sent_poss in zip(paragraph_text, paragraph_poss):
                if p_i == 0:  # In title.
                    title_term_list.extend(sent_text)
                    title_poss_list.extend(sent_poss)
                    continue  # If the terms are in title, we don't those terms in abstract and article term.
                else:  # p_i != 0
                    paragraph_term_list.extend(sent_text)
                    paragraph_poss_list.extend(sent_poss)

            if p_i == 0 and title_ngram is None:
                title_ngram = get_ngrams(title_term_list, title_poss_list, 1,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)
                continue

            paragraph_ngram = get_ngrams(paragraph_term_list, paragraph_poss_list, 1,
                                         filter_fn=partial(filter_ngram, mode='any'),
                                         included_tags=POS_INCLUDED)
            if len(paragraph_ngram) == 0:
                continue

            article_ngram.extend(paragraph_ngram)

            if p_i >= 80:
                break

        added_terms_num = 0

        for added_term in title_ngram + article_ngram:
            if added_term in limited_terms_set:
                wiki_p_level_indexdb.inverted_index.add(added_term, article_title)
                added_terms_num += 1
            elif ' ' not in added_term:
                wiki_p_level_indexdb.inverted_index.add(added_term, article_title)
                added_terms_num += 1

        wiki_p_level_indexdb.document_length_table.add(article_title, added_terms_num)

        count += 1

        # if count >= 5000:
        #     break

    wiki_p_level_indexdb.save_to_file(file_name)


def hash(token, num_buckets=None):
    """Unsigned 32 bit murmurhash for feature hashing."""
    if num_buckets is None:
        return murmurhash3_32(token, positive=True)
    else:
        return murmurhash3_32(token, positive=True) % num_buckets


if __name__ == '__main__':
    # abs_rindexdb = IndexDB()
    # abs_rindexdb.load_from_file(config.PDATA_ROOT / "reverse_indexing/abs_rindexdb")
    # print(len(abs_rindexdb))
    # query = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
    # whole_wiki_pages_title_raw_indexing_paragraph_level()
    # whole_wiki_pages_title_raw_indexing_paragraph_level(limited_terms=True)
    # whole_wiki_pages_title_raw_indexing_paragraph_level_unigram()
    # whole_wiki_pages_title_raw_indexing_paragraph_level_unigram_size_limited(2 ** 24)
    whole_wiki_pages_title_raw_indexing_paragraph_level_unigram_size_limited_memory_saving()
    # g_score_dict = dict()
    # load_from_file(g_score_dict,
    #                config.PDATA_ROOT / "reverse_indexing/wiki_p_level_unigram_rindexdb_hash_size_limited/scored_db/default-tf-idf.score.txt",
                   # config.PDATA_ROOT / "reverse_indexing/wiki_p_level_unigram_rindexdb/inverted_index.txt",
                   # with_int_type=True, memory_efficient=True)
    # print(g_score_dict)
    # whole_wiki_pages_title_raw_indexing_article_level()
    # whole_wiki_pages_title_raw_indexing_paragraph_level_to_indexdb()
    # whole_wiki_pages_title_raw_indexing_article_level_to_indexdb()


    # whole_wiki_pages_title_raw_indexing()
