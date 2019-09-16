from collections import namedtuple
from lxml import etree
import urllib.parse
from inspect_wikidump import init_inspect
import config
from tqdm import tqdm
import json

from inspect_wikidump.inspect_whole_file import check_boundary

Hyperlink = namedtuple("Hyperlink", ["inner_text", "href", "start_position", "end_position",
                                     "start_inner_text_position", "end_inner_text_position"])


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


def extract_hyperlink_and_append(item):
    # This function will return one more field (append to item later) in which the extracted hyperlink lies.
    # The new field will be as structure as text, each sentence will have a list of hyperlinks.
    assert len(item['text']) == len(item["charoffset"])
    article_hyperlinks = []  # article level hyperlinks

    # Article level
    for paragraph, paragraph_offsets in zip(item['text'], item["charoffset"]):  # Paragraph level
        # paragraph is a list of strings (each indicates one sentence), and paragraph_offsets is a list of tuples.
        assert len(paragraph) == len(paragraph_offsets)
        paragraph_text = ''.join(paragraph)  # Text of the whole paragraph

        paragraph_hyperlinks = []  # paragraph level hyperlinks

        in_hyperlink = False
        start_hyperlink_index = 0
        start_inner_hyperlink_index = 0
        end_hypoerlink_index = 0
        end_inner_hyperlink_index = 0

        for sentence, sentence_offsets in zip(paragraph, paragraph_offsets):
            # sentence is str, sentece_offset: list of tuple_list (indicating token range)
            cur_sent_tokens = []  # This is with all the hyperlink-tag none-deleted.
            results_hyperlinks_text = []  # This variable will be useless, not for probing.
            sentence_hyperlinks = []
            # sentence-level hyperlinks, (according to the end position.) this will be fixed laster.

            for start, end in sentence_offsets:
                cur_token = paragraph_text[start:end]
                cur_sent_tokens.append(cur_token)  # We keep all the tokens.

                if cur_token.startswith("<a href=\""):
                    if "<a href=\"http%3A//" in cur_token or "<a href=\"https%3A//" in cur_token \
                            or "<a href=\"//" in cur_token:  # Ignore external links.
                        continue

                    if in_hyperlink:
                        # We didn't find a correct "</a>" to close the last encountered hyperlink
                        cur_hyperlink_start_token = paragraph_text[start_hyperlink_index:start_inner_hyperlink_index]
                        hl_head = etree.fromstring(cur_hyperlink_start_token + "</a>")
                        hl_href = hl_head.get('href')

                        # Here is an important addition of empty text hyperlink
                        # We only append the token head here.
                        sentence_hyperlinks.append(Hyperlink(inner_text="", href=hl_href,
                                                             start_position=start_hyperlink_index,
                                                             end_position=start_inner_hyperlink_index,
                                                             start_inner_text_position=start_inner_hyperlink_index,
                                                             end_inner_text_position=start_inner_hyperlink_index))

                        # in_hyperlink = True

                        # print(item)
                        # print(item['title'])
                        # print(sentence)
                        # print(cur_sent_tokens)
                        # print(cur_token)
                        print(hl_href)
                        print("Potential Error. We didn't find a correct '</a>' to close the last hyperlink.")

                    in_hyperlink = True
                    start_hyperlink_index = start
                    start_inner_hyperlink_index = end

                elif cur_token == "</a>" or (cur_token == "»" and paragraph_text[end - 4:end] == "</a»"):
                    if not in_hyperlink:
                        continue  # Fail to reveal the start.   We just ignore.

                    in_hyperlink = False
                    end_inner_hyperlink_index = start
                    end_hypoerlink_index = end

                    cur_hyperlink_start_token = paragraph_text[start_hyperlink_index:start_inner_hyperlink_index]
                    cur_hyperlink = paragraph_text[start_hyperlink_index:end_hypoerlink_index]
                    cur_hyperlink_inner_text = paragraph_text[start_inner_hyperlink_index:end_inner_hyperlink_index]

                    results_hyperlinks_text.append(cur_hyperlink)
                    # print(cur_hyperlink)
                    try:
                        hl = etree.fromstring(cur_hyperlink)
                        assert hl.text == cur_hyperlink_inner_text
                        hl_href = urllib.parse.unquote(hl.get('href'))
                    except:
                        hl_head = etree.fromstring(cur_hyperlink_start_token + "</a>")
                        hl_href = urllib.parse.unquote(hl_head.get('href'))

                    # Append the (hyperlink + [start:end])  w.r.t the current paragraph.
                    sentence_hyperlinks.append(Hyperlink(inner_text=cur_hyperlink_inner_text, href=hl_href,
                                                         start_position=start_hyperlink_index,
                                                         end_position=end_hypoerlink_index,
                                                         start_inner_text_position=start_inner_hyperlink_index,
                                                         end_inner_text_position=end_inner_hyperlink_index))
                    # sentence_hyperlinks.append((Hyperlink(cur_hyperlink_inner_text, hl_href),
                    #                             [start_hyperlink_index, end_hypoerlink_index,
                    #                              start_inner_hyperlink_index, end_inner_hyperlink_index]))

            paragraph_hyperlinks.append(sentence_hyperlinks)  # End paragraph level

        article_hyperlinks.append(paragraph_hyperlinks)  # End article level

    # Hyperlink offset fixing.
    assert len(article_hyperlinks) == len(item["charoffset"])
    """
    The process above extract the hyperlinks and append them to the sentences.
    However, which sentence the hyperlink appended to is positioned according to the end position of the hyperlink.
    But, we want to append the hyperlink according to the start position of inner text. (Just like how human read webpage.)
    """

    fixed_article_hyperlinks = []
    # fixed_paragraph_hyperlinks = None
    for i, (paragraph_hyperlinks, paragraph_offsets) in enumerate(zip(article_hyperlinks, item["charoffset"])):
        fixed_paragraph_hyperlinks = []
        all_paragraph_hyperlinks = []
        paragraph_text = ''.join(item['text'][i])  # Text of the whole paragraph

        for sentence_hyperlinks in paragraph_hyperlinks:
            all_paragraph_hyperlinks.extend(sentence_hyperlinks)
        #
        assert len(paragraph_hyperlinks) == len(paragraph_offsets)  # Number of sentences

        for sentence, sentence_offsets in zip(paragraph_hyperlinks, paragraph_offsets):
            # sentence is str, sentece_offset: list of tuple_list (indicating token range)
            fixed_sentence_hyperlinks = []

            sentence_start = sentence_offsets[0][0]
            sentence_end = sentence_offsets[-1][-1]
            for hyperlink in all_paragraph_hyperlinks:
                cur_hlink_inner_text_start = hyperlink.start_inner_text_position
                cur_hlink_inner_text_end = hyperlink.end_inner_text_position

                if cur_hlink_inner_text_start >= sentence_start and cur_hlink_inner_text_end <= sentence_end:
                    fixed_sentence_hyperlinks.append(hyperlink)

            # print(paragraph_text[sentence_start:sentence_end])
            # print(fixed_sentence_hyperlinks)
            fixed_paragraph_hyperlinks.append(fixed_sentence_hyperlinks)

        fixed_article_hyperlinks.append(fixed_paragraph_hyperlinks) if fixed_paragraph_hyperlinks is not None else None

    return fixed_article_hyperlinks


def iterative_processing(debug_num=None):
    total_doc_num = init_inspect.TOTAL_NUM_DOC if debug_num is None else debug_num
    cur_count = 0

    with open(config.WHOLE_WIKI_FILE, 'rb') as in_f:
        for line in tqdm(in_f, total=total_doc_num):

            if debug_num is not None and debug_num == cur_count:
                break

            item = json.loads(line, encoding='utf-8')
            cur_count += 1

            if not check_boundary(item):
                continue

            item_hyperlinks = extract_hyperlink_and_append(item)
            print(item)
            print(item_hyperlinks)

    print("Total Count:", cur_count)


if __name__ == '__main__':
    iterative_processing(10)
