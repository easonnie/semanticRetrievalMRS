import config
import bz2
import json
from tqdm import tqdm
import json
from sqlitedict import SqliteDict

# This file is important to convert wiki dump to one file "whole_file.jsonl" that contains all the wiki pages.


TOTAL_NUM_DOC = 5_486_212

# Whole: 5_486_211
# ABS: 5_233_329


def get_all_wiki_doc(wiki_path):
    wfile_list = []

    def append_wiki_file_list(current_path, file_list):
        if current_path.is_file():
            file_list.append(current_path)
        else:
            for subdir in current_path.iterdir():
                append_wiki_file_list(subdir, file_list)

    append_wiki_file_list(wiki_path, wfile_list)
    for wiki_file in wfile_list:
        if wiki_file.suffix != '.bz2':
            print("Potential wrong file type!")

    return wfile_list


def file2json_items(file_path_name):
    items = []
    with bz2.open(file_path_name, mode='rb') as in_f:
        for line in in_f:
            item = json.loads(line, encoding='utf-8')
            items.append(item)

    return items


def write_to_file(file_path_list, out_file_path_name):
    with open(out_file_path_name, mode='wb') as out_f:
        for i, file_path_name in tqdm(enumerate(file_path_list)):
            with bz2.open(file_path_name, mode='rb') as in_f:
                # print(f"Writing {i} file from", file_path_name, "to", out_file_path_name)
                for line in in_f:
                    out_f.write(line)


def write_to_db(file_path_list, out_file_path_name):
    with SqliteDict(str(out_file_path_name), encode=json.dumps, decode=json.loads) as db_dict:
        for i, file_path_name in tqdm(enumerate(file_path_list)):
            with bz2.open(file_path_name, mode='rb') as in_f:
                # print(f"Writing {i} file from", file_path_name, "to", out_file_path_name)
                for line in in_f:
                    item = json.loads(line)
                    title = item['title']
                    db_dict[title] = item

        db_dict.commit()
        db_dict.close()


def file_to_db(file_pathname, db_path_name):
    with open(file_pathname, encoding='utf-8', mode='r') as in_f, \
            SqliteDict(str(db_path_name), encode=json.dumps, decode=json.loads) as db_dict:
        for line in tqdm(in_f):
            item = json.loads(line)
            title = item['title']
            db_dict[title] = item

        db_dict.commit()
        db_dict.close()


if __name__ == '__main__':
    wiki_file_list = get_all_wiki_doc(config.WHOLE_WIKI_DUMP_PATH)
    wiki_file_list.sort()
    print(len(wiki_file_list))
    # write_to_file(wiki_file_list, config.PDATA_ROOT / "whole_file.jsonl")
    write_to_db(wiki_file_list, config.PDATA_ROOT / "whole_file.db")
    #
    #
    # wiki_file_list = get_all_wiki_doc(config.ABS_WIKI_DUMP_PATH)
    # wiki_file_list.sort()
    # print(len(wiki_file_list))
    # # write_to_file(wiki_file_list, config.PDATA_ROOT / "whole_abs_file.jsonl")
    # write_to_db(wiki_file_list, config.PDATA_ROOT / "whole_abs_file.db")

    # file_to_db(config.PDATA_ROOT / "whole_abs_file.jsonl", config.PDATA_ROOT / "whole_abs_file.db")
    # file_to_db(config.PDATA_ROOT / "whole_abs_file.jsonl", config.PDATA_ROOT / "whole_abs_file.db")


    # for n1, n2 in zip(wiki_file_list, wiki_file_list_0):
    # print(n1)
    # print(str(n1).replace("processed", "abstracts"))
    # print(str(n1).replace("processed", "abstracts") == str(n2))

    # print(wiki_file_list, wiki_file_list_0)

    # print(wiki_file_list)

    # wiki_file_list = wiki_file_list[:80]

    # doc_list = []
    # for cur_file in tqdm(wiki_file_list):
    #     doc_list.extend(file2json_items(cur_file))
    #
    # print(len(doc_list))
