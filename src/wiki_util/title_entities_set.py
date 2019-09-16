import config
from tqdm import tqdm
import re
import json
from wiki_util import wiki_db_tool
from utils import text_process_tool
# from tqdm import

title_entity_set = None

disambiguation_group = None

normalization_mapping = None

disamb_pattern = re.compile(r'(^.*)\s\(.*\)$')


# Code added on 2019-03-22, to save all the title from wiki abs file.
def iterative_save_all_title(debug_num=None):
    total_doc_num = wiki_db_tool.TOTAL_ARTICLE_NUMBER_ABS if debug_num is None else debug_num
    cur_count = 0
    a_cursor = wiki_db_tool.get_cursor(config.ABS_WIKI_DB)
    a_cursor.execute("SELECT * from unnamed")

    with open(config.WIKI_TITLE_SET_FILE, mode='w', encoding='utf-8') as out_f:
        for key, value in tqdm(a_cursor, total=total_doc_num):
            item = json.loads(value)
            cur_count += 1
            out_f.write(item['title'] + "\n")


def get_disamb_match(text_in):
    matches = list(disamb_pattern.finditer(text_in))
    if len(matches) == 0:
        return None, None
    else:
        match = matches[0]
        return match.group(0), match.group(1)


def get_title_entity_set():
    global title_entity_set
    global disambiguation_group
    global normalization_mapping
    if disambiguation_group is None:
        disambiguation_group = dict()
    if title_entity_set is None:
        title_entity_set = set()
        with open(config.WIKI_TITLE_SET_FILE, mode='r', encoding='utf-8') as in_f:
            print("Get title entity set from", config.WIKI_TITLE_SET_FILE)
            for line in tqdm(in_f):
                cur_title = line.strip()
                title_entity_set.add(cur_title)

                # Build disambiguation groups
                dis_whole, dis_org = get_disamb_match(cur_title)
                if dis_whole is not None:
                    if dis_org not in disambiguation_group:
                        disambiguation_group[dis_org] = set()
                    disambiguation_group[dis_org].add(dis_whole)

        for title in title_entity_set:
            if title in disambiguation_group:
                disambiguation_group[title].add(title)

    if normalization_mapping is None:
        normalization_mapping = dict()
        for title in title_entity_set:
            if title != text_process_tool.normalize(title):
                normalization_mapping[text_process_tool.normalize(title)] = title

        return title_entity_set
    else:
        return title_entity_set


if __name__ == "__main__":
    # iterative_save_all_title()
    title_set = get_title_entity_set()
    # print(disambiguation_group["Big Stone Gap"])
    for kw in tqdm(disambiguation_group):
        print(kw)

    # print(get_title_entity_set())
    # count = 0
    # for k, v in disambiguation_group.items():
    #     print(k)
    #     print(v)
    #     count += 1
    #     if count == 10:
    #         break
