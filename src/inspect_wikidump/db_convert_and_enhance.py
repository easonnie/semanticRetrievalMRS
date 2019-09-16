import config
import json
from tqdm import tqdm
from inspect_wikidump import init_inspect
from urllib.parse import unquote
from lxml import etree
from collections import Counter
import spacy
import spacy.tokens
import urllib.parse
from sqlitedict import SqliteDict
import re

from inspect_wikidump.inspect_whole_file import check_boundary

nlp = spacy.load('en')

extract_pattern = re.compile(r'<a href="(.+)">(.*)</a>')
backup_extract_pattern = re.compile(r'<a href="(.+)">')


def extract_href(h_text):
    matches = list(extract_pattern.finditer(h_text))
    if len(matches) != 0:
        return matches[0].group(1), matches[0].group(2)
    else:
        matches = list(backup_extract_pattern.finditer(h_text))
        if len(matches) == 0:
            return None, None
        return matches[0].group(1), ""


def spacy_get_pos(tokens):
    doc = spacy.tokens.doc.Doc(
        nlp.vocab, words=tokens)

    for name, proc in nlp.pipeline:
        if name == 'tagger':
            proc(doc)

    return [token.pos_ for token in doc]


def convert_current_item(item):
    assert len(item['text']) == len(item["charoffset"])
    # Article level

    processed_item = {
        'sentences': [],
        'paragraph_tags': [],
        'hyperlinks': [],
        'poss': []
    }

    paragraph_num = 0

    aricle_valid_paragraph_tags = []
    aricle_valid_poss = []
    aricle_valid_sentences = []
    aricle_valid_hyperlinks = []
    aricle_valid_raw_tokens = []

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
            valid_tokens = []
            valid_hyperlinks = []
            valid_raw_tokens = []

            cur_sent_tokens = []
            results_hyperlinks = []

            hyper_link_start_token_num = -1
            hyper_link_end_token_num = -1

            for token_i, (start, end) in enumerate(sentence_offsets):

                cur_token = paragraph_text[start:end]
                valid_raw_tokens.append(cur_token)
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
                    hyper_link_start_token_num = token_i
                    start_hyperlink_index = start
                    start_inner_hyperlink_index = end
                    continue

                elif cur_token == "</a>" or (cur_token == "»" and paragraph_text[end-4:end] == "</a»"):
                    if not in_hyperlink:
                        continue  # Fail to reveal the start.

                    in_hyperlink = False
                    end_hypoerlink_index = end
                    end_inner_hyperlink_index = start
                    hyper_link_end_token_num = token_i

                    cur_hyperlink_start_token = paragraph_text[start_hyperlink_index:start_inner_hyperlink_index]
                    cur_hyperlink = paragraph_text[start_hyperlink_index:end_hypoerlink_index]
                    cur_hyperlink_inner_text = paragraph_text[start_inner_hyperlink_index:end_inner_hyperlink_index]

                    results_hyperlinks.append(cur_hyperlink)
                    # print(cur_hyperlink)

                    raw_title, h_inner_text = extract_href(cur_hyperlink)

                    if raw_title is not None:
                        if len(h_inner_text) > 0:
                            assert h_inner_text == cur_hyperlink_inner_text
                        link_title = urllib.parse.unquote(raw_title)
                        if hyper_link_start_token_num == -1 and len(aricle_valid_hyperlinks) > 0:
                            # append to last sentence
                            aricle_valid_hyperlinks[-1].append((len(aricle_valid_sentences[-1]) - 1,
                                                                len(aricle_valid_sentences[-1]) - 1,
                                                                link_title))
                        else:
                            valid_hyperlinks.append((hyper_link_start_token_num, hyper_link_end_token_num, link_title))
                    # we reach the end of a hyperlink
                    continue

                    # try:
                    #     hl = etree.fromstring(cur_hyperlink)
                    #     assert hl.text == cur_hyperlink_inner_text
                    #     raw_title = hl.get('href')
                    #     link_title = urllib.parse.unquote(raw_title)
                    # except:
                    #     hl_head = etree.fromstring(cur_hyperlink_start_token + "</a>")
                    #     raw_title = hl_head.get('href')
                    #     link_title = urllib.parse.unquote(raw_title)
                valid_tokens.append(cur_token)
                # token end

            aricle_valid_sentences.append(valid_tokens)
            # aricle_valid_poss.append(spacy_get_pos(valid_tokens))
            aricle_valid_hyperlinks.append(valid_hyperlinks)
            aricle_valid_paragraph_tags.append(paragraph_num)
            aricle_valid_raw_tokens.append(valid_raw_tokens)
            # sentence end

        paragraph_num += 1
        # paragraph end

    processed_item['sentences'] = aricle_valid_sentences
    processed_item['poss'] = aricle_valid_poss
    processed_item['hyperlinks'] = aricle_valid_hyperlinks
    processed_item['paragraph_tags'] = aricle_valid_paragraph_tags

    return processed_item


def iterative_checking_from_db(debug_num=None):

    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0

    # error_count_dict = {}
    # for key, value in check_func_dict.items():
    #     error_count_dict[f"{key}_count"] = 0

    with SqliteDict(str(config.WHOLE_WIKI_DB), flag='r', encode=json.dumps, decode=json.loads) as whole_db:

        titles = []
        for title in tqdm(whole_db.iterkeys(), total=len(whole_db)):
            titles.append(title)

        for title in tqdm(titles):
            item = whole_db[title]
            if debug_num is not None and debug_num == cur_count:
                whole_db.close()
                break

            # item = json.loads(line, encoding='utf-8')
            cur_count += 1

            # print(item)

            # Important, we don't check boundary error here to accelerate but might give problem in the future.
            # if not check_boundary(item):
            #     continue

            processed_item = convert_current_item(item)

    # for key, value in error_count_dict.items():
    #     print(key, value)
    #
    print("Total Count:", cur_count)


if __name__ == '__main__':
    iterative_checking_from_db(1000)