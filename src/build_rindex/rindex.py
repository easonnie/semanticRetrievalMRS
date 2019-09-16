from inspect_wikidump import init_inspect
from utils import common
import config
import json
from tqdm import tqdm
import spacy
import spacy.tokens
from sqlitedict import SqliteDict
import numpy as np


# We only need documents.
nlp = spacy.load('en')
nlp.remove_pipe('parser')
nlp.remove_pipe('ner')


def spacy_get_pos(tokens):
    doc = spacy.tokens.doc.Doc(
                nlp.vocab, words=tokens)

    for name, proc in nlp.pipeline:
        proc(doc)

    return [token.pos_ for token in doc]


def get_sentence_tokens(texts, charoffsets):
    whole_text = "".join(texts)
    tokens = []
    sentence_offsets = []

    start_t = 0
    end_t = 0
    for offset_list in charoffsets:
        end_t = start_t
        for start, end in offset_list:
            cur_token = whole_text[start:end]
            if len(cur_token) > 0:
                tokens.append(cur_token)
                end_t += 1
        sentence_offsets.append((start_t, end_t))
        start_t = end_t
    return tokens, sentence_offsets


def iterative_abs(debug_num=None):
    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0

    with open(config.ABS_WIKI_FILE, 'rb') as abs_file:
        for line in tqdm(abs_file, total=total_doc_num):
            item = json.loads(line)
            # print(item.keys())
            # print()
            tokens, sent_offset = get_sentence_tokens(item['text'], item['charoffset'])
            poss = spacy_get_pos(tokens)
            assert len(tokens) == len(poss)
            print(tokens)
            print(sent_offset)
            # print(poss)


def iterative_abs_save_info(debug_num=None):
    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0

    with open(config.ABS_WIKI_FILE, 'rb') as abs_file:
        with SqliteDict(str(config.ABS_PROCESS_FOR_RINDEX_DB), encode=json.dumps, decode=json.loads) as abs_rindex_db:
            for line in tqdm(abs_file, total=total_doc_num):
                item = json.loads(line)
                # print(item.keys())
                # print()
                if item['title'] in abs_rindex_db:
                    continue

                tokens, sent_offset = get_sentence_tokens(item['text'], item['charoffset'])
                poss = spacy_get_pos(tokens)
                assert len(tokens) == len(poss)
                # print(tokens)
                # print(sent_offset)
                abs_rindex_db[item['title']] = {
                    'tokens': tokens,
                    'poss': poss,
                    'sentence_offset': sent_offset
                }
                cur_count += 1

                if cur_count % 5000 == 0:
                    abs_rindex_db.commit()

            abs_rindex_db.commit()
            abs_rindex_db.close()


def iterative_abs_save_random_batching(batch_size=10000):
    total_doc_num = init_inspect.TOTAL_NUM_DOC

    with open(config.ABS_WIKI_FILE, 'rb') as abs_file:
        lines = []
        for line in tqdm(abs_file, total=total_doc_num):
            lines.append(line)
            # if len(lines) == 100000:
            #     break

    random_per = range(len(lines))
    # random_per = np.random.permutation(len(lines))
    # random.shuffle(lines)

    # existing_title_set = set()

    batch_list = []

    with SqliteDict(str(config.ABS_PROCESS_FOR_RINDEX_DB), encode=json.dumps, decode=json.loads) as abs_rindex_db:
        for index in tqdm(random_per):
            item = json.loads(lines[index])
            # print(item.keys())
            # print()
            if item['title'] in abs_rindex_db:
                continue

            tokens, sent_offset = get_sentence_tokens(item['text'], item['charoffset'])
            poss = spacy_get_pos(tokens)
            assert len(tokens) == len(poss)
            # print(tokens)
            # print(sent_offset)
            rindex_item = {
                'tokens': tokens,
                'poss': poss,
                'sentence_offset': sent_offset
            }

            batch_list.append((item['title'], rindex_item))

            if len(batch_list) == batch_size:
                for title, rindex_item in batch_list:
                    abs_rindex_db[title] = rindex_item
                abs_rindex_db.commit()
                batch_list = []

        # Commit last one
        for title, rindex_item in batch_list:
            abs_rindex_db[title] = rindex_item

        abs_rindex_db.commit()
        abs_rindex_db.close()


if __name__ == '__main__':
    # iterative_abs()
    # iterative_abs_save_info()
    iterative_abs_save_random_batching()