import re

import config
import json
from sqlitedict import SqliteDict
import sqlite3
from tqdm import tqdm

from inspect_wikidump.whole_abs_cross_checking import TOTAL_ARTICLE_COUNT
from wiki_util.wiki_db_tool import get_cursor


anchor_pattern = r"<(?:a\b[^>]*>|/a>)"


def remove_all_anchor(input_str):
    return re.sub(anchor_pattern, "", input_str)


def get_raw_text(item):
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


def query_raw_text(cursor, article_title, p_num=-1, table_name='raw_text'):
    if p_num <= -1:
        cursor.execute(f"SELECT * FROM {table_name} WHERE article_title=?", (article_title, ))
    else:
        # print(f"SELECT * FROM {table_name} WHERE article_title={article_title} AND p_num={p_num}")
        cursor.execute(f"SELECT * FROM {table_name} WHERE article_title=? AND p_num=?", (article_title, p_num))
    r_list = []
    for element in cursor:
        r_list.append(element)
    return r_list


def iterative_build_raw_text():
    wiki_whole_db_cursor = get_cursor(str(config.WHOLE_WIKI_DB))
    wiki_whole_db_cursor.execute("SELECT * from unnamed")
    total_count = 0
    insert_data_list = []

    db_path = config.WHOLE_WIKI_RAW_TEXT
    conn = sqlite3.connect(str(db_path))
    saving_cursor = conn.cursor()

    for key, value in tqdm(wiki_whole_db_cursor, total=TOTAL_ARTICLE_COUNT):
        cur_item = json.loads(value)
        raw_text = get_raw_text(cur_item)

        article_title = cur_item['title']

        for p_num, paragraph in enumerate(raw_text):
            assert isinstance(paragraph, list)
            p_str = json.dumps(paragraph)
            insert_data_list.append((article_title, p_num, p_str))
            total_count += 1
            conn.commit()

        if len(insert_data_list) >= 5000:
            insert_many_raw_text_table(saving_cursor, insert_data_list)
            insert_data_list = []

        # if total_count >= 10000:
        #     break

    print(total_count)
    insert_many_raw_text_table(saving_cursor, insert_data_list)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    # create_raw_text_tables(config.WHOLE_WIKI_RAW_TEXT)
    # iterative_build_raw_text()
    # pass
    # iterative_build()
    # iterative_build_raw_text()
    # create_raw_text_tables(config.WHOLE_WIKI_RAW_TEXT)
    db_path = config.WHOLE_WIKI_RAW_TEXT
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    r_list = query_raw_text(cursor, 'China')
    print(r_list)
    # # insert_many_raw_text_table(cursor, [('aaa', 3, "['a', 'b']"), ('aaa', 3, "['a', 'a', 'b']")])
    # insert_many_raw_text_table(cursor, [('aaa', 4, "['a12', 'b']"), ('aaa', 5, "['a43', 'a', 'b']")])
    # conn.commit()
    # # conn.close()
    #
    # print(query_raw_text(cursor, 'aaa', 4))