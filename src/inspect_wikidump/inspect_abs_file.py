import config
import json
from tqdm import tqdm
from inspect_wikidump import init_inspect
from urllib.parse import unquote
from lxml import etree
from collections import Counter
from sqlitedict import SqliteDict


def iterative_checking(check_func_dict, debug_num=None, verbose=False):
    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0

    with open(config.ABS_WIKI_FILE, 'rb') as in_f:
        for line in tqdm(in_f, total=total_doc_num):
            if debug_num is not None and debug_num == cur_count:
                break

            item = json.loads(line)
            print(item)
            print(item.keys())
            print(item['text_with_links'])
            print(item['charoffset_with_links'])
            cur_count += 1

    print("Total Count:", cur_count)


# We will need to do this later to check the consistency
# TODO do cross checking for 'whole' and 'abs'
def iterative_cross_checking_abs_whole(debug_num=None):
    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0
    error_count = 0

    with open(config.WHOLE_WIKI_FILE, 'rb') as in_whole_f, open(config.ABS_WIKI_FILE, 'rb') as in_abs_f:

        while True:
            if debug_num is not None and debug_num == cur_count:
                break
            whl_line = next(in_whole_f)
            abs_line = next(in_abs_f)

            if whl_line and abs_line:
                whl_item = json.loads(whl_line)
                abs_item = json.loads(abs_line)

                cur_count += 1

                the_para = None
                for whl_para in whl_item['text'][1:]:
                    print(whl_para)
                    if len(''.join(whl_para)) > 50:
                        the_para = whl_para
                        break

                if the_para != abs_item['text_with_links']:
                    print(abs_item['title'])
                    print(whl_item['title'])
                    print(the_para)
                    print(whl_item['text'])
                    print(abs_item['text_with_links'])
                    # print(whl_item['text'][1])
                    error_count += 1
                    raise Exception()

    print(error_count)
    print(cur_count)
    print(error_count / cur_count)


def iterative_cross_checking_abs_whole_from_db(debug_num=None):
    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0
    error_count = 0

    with SqliteDict(str(config.ABS_WIKI_DB), flag='r', encode=json.dumps, decode=json.loads) as abs_db, \
            SqliteDict(str(config.WHOLE_WIKI_DB), flag='r', encode=json.dumps, decode=json.loads) as whole_db:

        titles = []
        for title in tqdm(abs_db.iterkeys(), total=len(abs_db)):
            titles.append(title)

        for title in tqdm(titles):
            abs_item = abs_db[title]
            if title not in whole_db:
                print(f"Title: {title} not in whole_db")
                return
            else:
                whl_item = whole_db[title]

            cur_count += 1

            the_para = None
            for whl_para in whl_item['text'][0:]:
                # print(whl_para)
                if len(''.join(whl_para)) >= 50:
                    the_para = whl_para
                    break

            if the_para != abs_item['text_with_links']:
                print(abs_item['title'])
                print(whl_item['title'])
                print(the_para)
                print(whl_item['text'])
                print(the_para)
                print(abs_item['text_with_links'])
                # print(whl_item['text'][1])
                error_count += 1
                raise Exception()

    print(error_count)
    print(cur_count)
    print(error_count / cur_count)


if __name__ == '__main__':
    # iterative_checking(None, debug_num=1)
    # iterative_cross_checking_abs_whole(100)
    iterative_cross_checking_abs_whole_from_db(100)
