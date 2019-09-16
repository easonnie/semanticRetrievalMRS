import config
import json
from sqlitedict import SqliteDict
import sqlite3
from tqdm import tqdm
import wiki_util.hyperlink
from inspect_wikidump.inspect_whole_file import get_first_paragraph_index
# from wiki_util.title_entities_set import get_title_entity_set

TOTAL_ARTICLE_NUMBER_ABS = 5_233_329

# Total title vocabulary = 1_821_696
# Total word count:
# Counter({2: 2225063, 3: 1118728, 1: 625115, 4: 569823, 5: 319777, 6: 174183, 7: 90339, 8: 42826, 9: 27853, 10: 19489, 11: 10030, 12: 5250, 13: 2591, 14: 1166, 15: 623, 16: 264, 17: 86, 18: 42, 20: 26, 19: 24, 23: 7, 21: 6, 22: 5, 28: 3, 25: 2, 31: 2, 27: 2, 24: 2, 30: 1, 40: 1})


def item_get_clean_text(item, hyperlinks=None):
    assert len(item['text']) == len(item["charoffset"])
    article_tokens = []  # article level hyperlinks

    # Article level
    for paragraph, paragraph_offsets in zip(item['text'], item["charoffset"]):  # Paragraph level
        # paragraph is a list of strings (each indicates one sentence), and paragraph_offsets is a list of tuples.
        assert len(paragraph) == len(paragraph_offsets)
        paragraph_text = ''.join(paragraph)  # Text of the whole paragraph

        paragraph_tokens = []  # paragraph level tokens

        for sentence, sentence_offsets in zip(paragraph, paragraph_offsets):
            # sentence is str, sentece_offset: list of tuple_list (indicating token range)
            cur_sent_tokens = []  # This is with all the hyperlink-tag deleted.

            for start, end in sentence_offsets:
                cur_token = paragraph_text[start:end]

                if cur_token.startswith("<a href=\""):
                    continue
                elif cur_token == "</a>" or (cur_token == "»" and paragraph_text[end - 4:end] == "</a»"):
                    continue

                if len(cur_token) != 0:
                    cur_sent_tokens.append(cur_token)  # We keep all the tokens.

            paragraph_tokens.append(cur_sent_tokens)

        article_tokens.append(paragraph_tokens)  # End article level
    return article_tokens


def get_cursor(save_path):
    if not isinstance(save_path, str):
        save_path = str(save_path)
    conn = sqlite3.connect(save_path)
    cursor = conn.cursor()
    return cursor


def get_item_by_key(cursor, key):
    cursor.execute("SELECT * from unnamed WHERE key=(?)", (key,))
    fetched_data = cursor.fetchall()
    if len(fetched_data) == 0:
        return None
    else:
        cur_key, cur_whole_value = fetched_data[0]
    return json.loads(cur_whole_value)


def get_first_paragraph_hyperlinks(cursor, key):
    hyperlinks = None
    item = get_item_by_key(cursor, key)
    item_hyperlinks = wiki_util.hyperlink.extract_hyperlink_and_append(item)
    frist_p_index = get_first_paragraph_index(item)
    return item_hyperlinks[frist_p_index]


def get_first_paragraph_hyperlinks_from_item(item):
    item_hyperlinks = wiki_util.hyperlink.extract_hyperlink_and_append(item)
    frist_p_index = get_first_paragraph_index(item)
    return item_hyperlinks[frist_p_index]


def get_paragraph_text(cursor, key, only_abs=False, flatten=False):
    item = get_item_by_key(cursor, key)
    if item is None:
        print(key, "No values!")
    clean_text = item_get_clean_text(item)
    if only_abs:
        index = get_first_paragraph_index(item)
        clean_text = clean_text[index]

    if flatten:
        all_para_text = []
        for sentence in clean_text:
            all_para_text.extend(sentence)
        return all_para_text
    else:
        return clean_text


def item_to_paragraph_list(item, flatten_paragraph=False):
    all_paragraphs = item['clean_text']
    paragraph_list = []
    for i, paragraph in enumerate(all_paragraphs):
        if len(paragraph) > 0:
            if flatten_paragraph:
                flatten_paragraph_token_list = []
                for sentence in paragraph:
                    flatten_paragraph_token_list.extend(sentence)
                paragraph_list.append((i, flatten_paragraph_token_list))
            else:
                paragraph_list.append((i, paragraph))

    return paragraph_list


def get_first_paragraph_index_from_clean_text_item(item, text_field_name='clean_text', skip_first=False):
    first_para_index = -1
    text_field_name = 'text' if text_field_name is None else text_field_name
    for i, para in enumerate(item[text_field_name]):
        if skip_first and i == 0:
            continue
        flatten_para = []
        for sent in para:
            flatten_para.extend(sent)
        # print(flatten_para)
        cur_para = ' '.join(flatten_para)
        # if len(cur_para) >= 50:
        if len(cur_para) >= 50:
            first_para_index = i
            break
    return first_para_index


def get_first_paragraph_from_clean_text_item(t_item, text_field_name='clean_text', flatten_to_paragraph=True, skip_first=False):
    # if fix_index is None:
    #     index = get_first_paragraph_index_from_clean_text_item(t_item, text_field_name)
    # else:
    #     index = fix_index
    index = get_first_paragraph_index_from_clean_text_item(t_item, text_field_name, skip_first)
    the_para = t_item['clean_text'][index]

    if flatten_to_paragraph:
        flatten_para_list = []
        # print(the_para)
        for sent in the_para:
            flatten_para_list.extend(sent)

        return flatten_para_list
    else:
        return the_para


if __name__ == '__main__':
    w_cursor = get_cursor(config.WHOLE_WIKI_DB)
    t_cursor = get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)
    # a_cursor = get_cursor(config.ABS_WIKI_DB)
    # ner_set = get_title_entity_set()

    # item = get_item_by_key(w_cursor, "Ulmus parvifolia 'Ed Wood'")
    item = get_item_by_key(w_cursor, "Kimber (name)")
    # item = get_item_by_key(w_cursor, "Chinese Elm")
    print(item)
    # print("Chinese Elm" in ner_set)

    # links = get_first_paragraph_hyperlinks(w_cursor, "Ulmus parvifolia 'Ed Wood'")
    # links = get_first_paragraph_hyperlinks(w_cursor, "Ulmus parvifolia 'Ed Wood'")
    # print(links)
    # for hl in links:
    #     print(hl, links)
    #
    text_item = get_item_by_key(t_cursor, key='Adam Collis')
    print(get_first_paragraph_from_clean_text_item(text_item))
    # index = get_first_paragraph_index_from_clean_text_item(text_item, 'clean_text')

    # print(text_item['clean_text'][index])

    # print(text_item)
    # first
    # paragraph_index = wiki_db_tool.get_first_paragraph_index(text_item)

    # c_text = get_paragraph_text(w_cursor, 'Mahershala Ali', only_abs=True, flatten=True)
    # c_text = get_paragraph_text(w_cursor, 'Adam Collis', only_abs=True, flatten=True)
    # c_text = get_paragraph_text(w_cursor, 'Ed Wood (film)', only_abs=True, flatten=True)
    # c_text = get_paragraph_text(w_cursor, 'Tyler Bates', only_abs=True, flatten=True)
    # c_text = get_paragraph_text(w_cursor, 'Doctor Strange (2016 film)', only_abs=True, flatten=True)
    # c_text = get_paragraph_text(w_cursor, 'Doctor Strange (2016 film)', only_abs=True, flatten=True)
    # print(' '.join(c_text))
    # print(c_text)
    # print(get_first_paragraph_hyperlinks(w_cursor, 'Mahershala Ali'))


    # print(item)
    # item = get_item_by_key(a_cursor, 'American')
    # print(item)
