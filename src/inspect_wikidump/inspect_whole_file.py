import config
import json
from tqdm import tqdm
from inspect_wikidump import init_inspect
from urllib.parse import unquote
from lxml import etree
from collections import Counter


# Total: 5,486,212
# Error offset/boundary 660     # We ignore all of them
# ETC: 14 mins

# Number of tile that didn't match first paragraph 9,648    We just use the title.


def check_identifier(item):
    if item['title'] == ''.join(item['text'][0]):
        return True
    else:
        return False


def check_sent_para_stats(item):
    total_sentence_num = 0
    total_paragraph_num = 0
    sentence_nums_in_paragraph = []

    for paragraph in item['text']:  # Paragraph level
        total_paragraph_num += 1
        cur_paragraph_sent_num = 0

        for _ in paragraph:
            total_sentence_num += 1
            cur_paragraph_sent_num += 1

        sentence_nums_in_paragraph.append(cur_paragraph_sent_num)

    return total_paragraph_num, total_sentence_num, sentence_nums_in_paragraph


def check_token_stats(item):
    token_per_a = []
    token_per_p = []
    token_per_s = []

    token_in_a = 0
    for paragraphs, paragraph_offsets in zip(item['text'], item["charoffset"]):  # Paragraph level
        token_in_p = 0
        for sentence, sentence_offsets in zip(paragraphs, paragraph_offsets):  # sentence
            token_in_s = 0
            for start, end in sentence_offsets:  # each token
                token_in_s += 1
                token_in_p += 1
                token_in_a += 1

            token_per_s.append(token_in_s)
        token_per_p.append(token_in_p)
    token_per_a.append(token_in_a)

    return token_per_a, token_per_p, token_per_s


def iterative_counting_token(debug_num=None):
    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0

    a_counter = Counter()
    p_counter = Counter()
    s_counter = Counter()

    with open(config.WHOLE_WIKI_FILE, 'rb') as in_f:
        for line in tqdm(in_f, total=total_doc_num):

            if debug_num is not None and debug_num == cur_count:
                break

            item = json.loads(line, encoding='utf-8')
            if not check_boundary(item):
                continue

            cur_count += 1

            token_per_a, token_per_p, token_per_s = check_token_stats(item)

            a_counter.update(token_per_a)
            p_counter.update(token_per_p)
            s_counter.update(token_per_s)

    with open("t_a_counter.json", encoding='utf-8', mode='w') as out_f:
        json.dump(a_counter, out_f)

    with open("t_p_counter.json", encoding='utf-8', mode='w') as out_f:
        json.dump(p_counter, out_f)

    with open("t_s_counter.json", encoding='utf-8', mode='w') as out_f:
        json.dump(s_counter, out_f)

    print(cur_count)
    print(len(a_counter))


def iterative_counting_sent_para(debug_num=None):
    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0

    total_sent_counter = Counter()
    total_para_counter = Counter()
    sent_per_para_counter = Counter()

    with open(config.WHOLE_WIKI_FILE, 'rb') as in_f:
        for line in tqdm(in_f, total=total_doc_num):

            if debug_num is not None and debug_num == cur_count:
                break

            item = json.loads(line, encoding='utf-8')
            if not check_boundary(item):
                continue

            cur_count += 1

            total_paragraph_num, total_sentence_num, sentence_nums_in_paragraph = check_sent_para_stats(item)

            total_para_counter.update([total_paragraph_num])
            total_sent_counter.update([total_sentence_num])
            sent_per_para_counter.update(sentence_nums_in_paragraph)

    with open("total_para_counter.json", encoding='utf-8', mode='w') as out_f:
        json.dump(total_para_counter, out_f)

    with open("total_sent_counter.json", encoding='utf-8', mode='w') as out_f:
        json.dump(total_sent_counter, out_f)

    with open("sent_per_para_counter.json", encoding='utf-8', mode='w') as out_f:
        json.dump(sent_per_para_counter, out_f)

    print(cur_count)


def paragraph_text_fixing(text: str) -> str:
    text = text.replace("</a»", "</a>")
    return text


def check_hyperlink_match_text(item):
    assert len(item['text']) == len(item["charoffset"])
    # Article level
    for paragraphs, paragraph_offsets in zip(item['text'], item["charoffset"]):  # Paragraph level
        assert len(paragraphs) == len(paragraph_offsets)
        paragraph_text = ''.join(paragraphs)  # Text of the whole paragraph
        # paragraph_text_fixing(paragraph_text)

        in_hyperlink = False
        start_hyperlink_index = 0
        start_inner_hyperlink_index = 0
        end_hypoerlink_index = 0
        end_inner_hyperlink_index = 0

        # if item['title'] == 'Franz Gürtner':
        #     what = 0

        for sentence, sentence_offsets in zip(paragraphs, paragraph_offsets):
            # sentence is str, sentece_offset: list of tuple_list
            cur_sent_tokens = []
            results_hyperlinks = []

            for start, end in sentence_offsets:
                cur_token = paragraph_text[start:end]
                cur_sent_tokens.append(cur_token)

                if cur_token.startswith("<a href=\""):
                    if "<a href=\"http%3A//" in cur_token or "<a href=\"https%3A//" in cur_token\
                            or "<a href=\"//" in cur_token:  # Ignore external links.
                        continue

                    if in_hyperlink:
                        # We didn't find a correct "</a>" to close
                        cur_hyperlink_start_token = paragraph_text[start_hyperlink_index:start_inner_hyperlink_index]
                        hl_head = etree.fromstring(cur_hyperlink_start_token + "</a>")
                        hl_href = hl_head.get('href')
                        # Remember here to append another indices

                        # in_hyperlink = True

                        # print(item)
                        # print(item['title'])
                        # print(sentence)
                        # print(cur_sent_tokens)
                        # print(cur_token)
                        print(hl_href)
                        print("Potential Error. Check This.")
                        # raise ValueError("Hyperlink Parsing Error!")

                    in_hyperlink = True
                    start_hyperlink_index = start
                    start_inner_hyperlink_index = end

                elif cur_token == "</a>" or (cur_token == "»" and paragraph_text[end-4:end] == "</a»"):
                    if not in_hyperlink:
                        continue  # Fail to reveal the start.

                    in_hyperlink = False
                    end_hypoerlink_index = end
                    end_inner_hyperlink_index = start

                    cur_hyperlink_start_token = paragraph_text[start_hyperlink_index:start_inner_hyperlink_index]
                    cur_hyperlink = paragraph_text[start_hyperlink_index:end_hypoerlink_index]
                    cur_hyperlink_inner_text = paragraph_text[start_inner_hyperlink_index:end_inner_hyperlink_index]

                    results_hyperlinks.append(cur_hyperlink)
                    # print(cur_hyperlink)
                    try:
                        hl = etree.fromstring(cur_hyperlink)
                        assert hl.text == cur_hyperlink_inner_text

                    except:
                        hl_head = etree.fromstring(cur_hyperlink_start_token + "</a>")

    return True


def check_boundary(item):  # Let's just delete all the boundary errors.
    assert len(item['text']) == len(item["charoffset"])
    # Article level
    for paragraph, paragraph_offsets in zip(item['text'], item["charoffset"]):  # Paragraph level
        assert len(paragraph) == len(paragraph_offsets)
        paragraph_text = ''.join(paragraph)  # Text of the whole paragraph

        for sentence, sentence_offsets in zip(paragraph, paragraph_offsets):
            # sentence is str, sentece_offset: list of tuple_list
            sentence_start = sentence_offsets[0][0]
            sentence_end = sentence_offsets[-1][-1]
            # Remember to strip the sentence to exclude any space in the original text.
            if sentence.strip() != paragraph_text[sentence_start:sentence_end]:
                return False

            # Important 2019/1/11 we just ignore the boundary error containing pages.
            # Total 660 boundary errors.
    return True


# This method is important to be applied on the whole_wiki_items.
def get_first_paragraph_index(item, text_field_name=None):
    first_para_index = -1
    text_field_name = 'text' if text_field_name is None else text_field_name
    for i, para in enumerate(item[text_field_name]):
        cur_para = ''.join(para)
        if len(cur_para) >= 50:
            first_para_index = i
            break
    return first_para_index


# This one is important, we will need to use this method during releasing.
def iterative_save_all_title(debug_num=None):
    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0
    # title_set = set()

    with open(config.WHOLE_WIKI_FILE, 'rb') as in_f, open("title_set.txt", mode='w', encoding='utf-8') as out_f:
        for line in tqdm(in_f, total=total_doc_num):

            if debug_num is not None and debug_num == cur_count:
                break

            item = json.loads(line, encoding='utf-8')
            cur_count += 1

            out_f.write(item['title'] + "\n")


def iterative_checking(check_func_dict, debug_num=None, verbose=False):
    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0

    error_count_dict = {}
    for key, value in check_func_dict.items():
        error_count_dict[f"{key}_count"] = 0

    with open(config.WHOLE_WIKI_FILE, 'rb') as in_f:
        for line in tqdm(in_f, total=total_doc_num):

            if debug_num is not None and debug_num == cur_count:
                break

            item = json.loads(line, encoding='utf-8')
            cur_count += 1

            # print(item)

            if not check_boundary(item):
                continue

            for key, vfunc in check_func_dict.items():
                if not vfunc(item):
                    error_count_dict[f"{key}_count"] = error_count_dict[f"{key}_count"] + 1
                    if verbose and key == 'title_match_first_para':
                        print('title:', item['title'])
                        print('text_0:', ' '.join(item['text'][0]))

    for key, value in error_count_dict.items():
        print(key, value)

    print("Total Count:", cur_count)


if __name__ == '__main__':
    check_func_dict = {
        # 'title_match_first_para': check_identifier,
        # 'offset_boundary_check': check_boundary
        'check_hyperlink_match_text': check_hyperlink_match_text,
    }
    
    iterative_checking(check_func_dict, debug_num=1, verbose=True)
    # iterative_counting_sent_para(None)
    # iterative_counting_token(100)
    # iterative_save_all_title(None)
