import re

import config
import json
from sqlitedict import SqliteDict
import sqlite3
from tqdm import tqdm

from build_rindex.rindex import spacy_get_pos
from inspect_wikidump.whole_abs_cross_checking import TOTAL_ARTICLE_COUNT
from wiki_util.hyperlink import item_get_clean_text
from wiki_util.wiki_db_tool import get_cursor


def iterative_build():
    # wiki_abs_db_cursor = get_cursor(str(config.ABS_WIKI_DB))
    wiki_whole_db_cursor = get_cursor(str(config.WHOLE_WIKI_DB))
    wiki_whole_db_cursor.execute("SELECT * from unnamed")
    total_count = 0
    cur_count = 0

    with SqliteDict(str(config.WHOLE_PROCESS_FOR_RINDEX_DB), encode=json.dumps, decode=json.loads) as whole_rindex_db:
        for key, value in tqdm(wiki_whole_db_cursor, total=TOTAL_ARTICLE_COUNT):
            cur_item = json.loads(value)
            # print(cur_item)
            clean_text = item_get_clean_text(cur_item)
            # print(clean_text)
            new_item = dict()
            new_item['title'] = cur_item['title']

            flatten_article_tokens = []

            for p_i, paragraph in enumerate(clean_text):
                # flatten_paragraph_tokens = []
                # paragraph_poss = []
                for s_i, sentence in enumerate(paragraph):
                    flatten_article_tokens.extend(sentence)

                # flatten_article_tokens.extend(flatten_paragraph_tokens)

            flatten_article_poss = spacy_get_pos(flatten_article_tokens)

            cur_ptr = 0
            article_poss = []
            for p_i, paragraph in enumerate(clean_text):
                paragraph_poss = []
                for s_i, sentence in enumerate(paragraph):
                    sentence_poss = []
                    for _ in sentence:
                        sentence_poss.append(flatten_article_poss[cur_ptr])
                        cur_ptr += 1
                    paragraph_poss.append(sentence_poss)
                article_poss.append(paragraph_poss)

            new_item['clean_text'] = clean_text
            new_item['poss'] = article_poss
            whole_rindex_db[new_item['title']] = new_item

            cur_count += 1

            if cur_count % 5000 == 0:
                whole_rindex_db.commit()

        whole_rindex_db.commit()
        whole_rindex_db.close()


anchor_pattern = r"<(?:a\b[^>]*>|/a>)"


def remove_all_anchor(input_str):
    return re.sub(anchor_pattern, "", input_str)


def get_raw_text(item):
    # assert len(item['text']) == len(item["charoffset"])
    article_list = []  # article level hyperlinks

    # Article level
    for paragraph in item['text']:  # Paragraph level
        # paragraph is a list of strings (each indicates one sentence), and paragraph_offsets is a list of tuples.
        paragraph_sent_list = []

        for sentence in paragraph:
            paragraph_sent_list.append(remove_all_anchor(sentence))

        article_list.append(paragraph_sent_list)  # End article level
    return article_list


def create_raw_text_tables(db_path, table_name='raw_text'):
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    c = cursor
    c.execute(f"CREATE TABLE {table_name} ("
              f"article_title TEXT NOT NULL, "
              f"p_num INTEGER NOT NULL, "
              f"value TEXT NOT NULL);")

    c.execute(f"CREATE INDEX article_title_index ON {table_name}(article_title);")
    c.execute(f"CREATE INDEX paragraph_index ON {table_name}(article_title, p_num);")


def insert_many_raw_text_table(cursor: sqlite3.Cursor, data, table_name='raw_text'):
    cursor.executemany(f"INSERT INTO {table_name} (article_title, p_num, value) VALUES (?, ?, ?)", data)


def iterative_build_raw_text():
    wiki_whole_db_cursor = get_cursor(str(config.WHOLE_WIKI_DB))
    wiki_whole_db_cursor.execute("SELECT * from unnamed")
    total_count = 0
    cur_count = 0

    with SqliteDict(str(config.WHOLE_WIKI_RAW_TEXT), encode=json.dumps, decode=json.loads) as whole_rindex_db:
        for key, value in tqdm(wiki_whole_db_cursor, total=TOTAL_ARTICLE_COUNT):
            cur_item = json.loads(value)
            raw_text = get_raw_text(cur_item)

            new_item = dict()
            new_item['title'] = cur_item['title']
            new_item['raw_text'] = raw_text
            whole_rindex_db[new_item['title']] = new_item

            cur_count += 1

            if cur_count % 5000 == 0:
                whole_rindex_db.commit()
                # break

        whole_rindex_db.commit()
        whole_rindex_db.close()


if __name__ == '__main__':
    # iterative_build()
    # iterative_build_raw_text()
    # create_raw_text_tables(config.WHOLE_WIKI_RAW_TEXT)
    db_path = config.WHOLE_WIKI_RAW_TEXT
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    insert_many_raw_text_table(cursor, [('aaa', 3, "['a', 'b']"), ('aaa', 3, "['a', 'a', 'b']")])
    conn.commit()
    conn.close()