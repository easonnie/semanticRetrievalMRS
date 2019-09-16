import config
import json
from sqlitedict import SqliteDict
import sqlite3
from tqdm import tqdm

TOTAL_ARTICLE_COUNT = 5_233_329


def get_cursor(save_path):
    conn = sqlite3.connect(save_path)
    cursor = conn.cursor()
    return cursor


if __name__ == '__main__':
    wiki_abs_db_cursor = get_cursor(str(config.ABS_WIKI_DB))
    wiki_abs_db_cursor.execute("SELECT * from unnamed")
    wiki_whole_db_cursor = get_cursor(str(config.WHOLE_WIKI_DB))
    total_count = 0
    for key, value in tqdm(wiki_abs_db_cursor, total=TOTAL_ARTICLE_COUNT):
        # print(key, value)
        cur_abs_item = json.loads(value)
        wiki_whole_db_cursor.execute("SELECT * from unnamed WHERE key=(?)", (key,))

        cur_key, cur_whole_value = wiki_whole_db_cursor.fetchall()[0]
        cur_whole_items = json.loads(cur_whole_value)
        # print(cur_whole_items.keys())
        # print(cur_abs_item.keys())
        abs_first_para = ''.join(cur_abs_item['text_with_links'])
        # print(abs_first_para)
        whole_first_para = ""
        for para in cur_whole_items['text']:
            cur_para = ''.join(para)
            if len(cur_para) >= 50:
                whole_first_para = cur_para
                break

        total_count += 1
        # print(whole_first_para)
        if abs_first_para != whole_first_para:
            print(cur_whole_items['title'], cur_whole_items['id'])
            print("ABS:", abs_first_para)
            print("WHL:", whole_first_para)
        # print(whole_items)

    print(total_count)


    # with SqliteDict(str(config.ABS_WIKI_DB), flag='r', encode=json.dumps, decode=json.loads) as wiki_abs_db, \
    #         SqliteDict(str(config.WHOLE_WIKI_DB), flag='r', encode=json.dumps, decode=json.loads) as wiki_whole_db:
    #     for key, _ in wiki_abs_db.iteritems():
    #         abs_item = wiki_abs_db[key]
    #         whole_item = wiki_whole_db[key]
    #         print(abs_item)
    #         print(wiki_whole_db)
    #         exit(0)

