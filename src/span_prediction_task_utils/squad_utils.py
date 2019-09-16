from pytorch_pretrained_bert import BertTokenizer

from utils import common
import config
import span_prediction_task_utils.common_utils as span_utils
import collections
# span prediction object
import uuid
from tqdm import tqdm

"""
Follow the interface of this script the fitem and eitem, s.t. 
the input data can be correctly handled by downstream pipeline modules.
"""


# TODO write a document later to explain the API.

SPECIAL_TOKEN_LIST = ('[CLS]', 'yes', 'no')

# Deprecated
def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


# Deprecated
def preprocssing_span_prediction_item(item, tokenizer, is_training,
                                      max_tokens_for_doc, doc_stride):
    ready_training_list = []

    c_tokens, c2w_index, w2c_index, c_start_position, c_end_position = span_utils.w_tokens2c_tokens(
        item['w_token_context'], item['answer_start'], item['answer_end'],
        item['gt_answer_text'], tokenizer, item['no_answer'], is_training
    )

    no_answer = item['no_answer']

    _DocSpan = collections.namedtuple(
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(c_tokens):
        length = len(c_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(c_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        start_position = None
        end_position = None

        fctoken_to_wtoken_map = {}  # we need a map to the original w-tokens
        for i in range(doc_span.length):
            # The index offset is 0.
            # the_index = doc_span.start + i
            fctoken_to_wtoken_map[i] = c2w_index[doc_span.start + i]

        if is_training and not item['no_answer']:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (c_start_position >= doc_start and c_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                no_answer = True
                start_position = -1
                end_position = -1
            else:
                # doc_offset = len(query_tokens) + 2
                doc_offset = 0
                start_position = c_start_position - doc_start + doc_offset
                end_position = c_end_position - doc_start + doc_offset
                # We change this from original code base, here the doc_offset will only w.r.t. the context.

        if is_training and item['no_answer']:
            no_answer = True
            start_position = -1
            end_position = -1

        if not is_training:
            # no_answer = True
            start_position = -2  # -2 indicates that the start and end position is hidden.
            end_position = -2

        fid = str(uuid.uuid4())  # This is unique forward id for each of the doc that will be feeded into net.
        uid = item['uid']

        # This the the code from Google BERT where we keep track of the score of current token.
        # This is only for sliding window approach.
        token_is_max_context = {}

        # We do this for all the document. Bc this is not that time-consuming. 2019-03-14
        # if not is_training:     # We only need to do this during inference  # But maybe not. Check laters
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[i] = is_max_context

        fc_token = c_tokens[doc_span.start:doc_span.start + doc_span.length]  # actual forward c-tokens
        f_item = dict()
        f_item['fid'] = fid
        f_item['uid'] = uid
        f_item['doc_span_index'] = doc_span_index
        f_item['context_w_tokens'] = item['w_token_context']
        f_item['context_c_tokens'] = fc_token
        f_item['no_answer'] = no_answer
        f_item['start_position'] = start_position
        f_item['end_position'] = end_position
        f_item['fctoken_to_wtoken_map'] = fctoken_to_wtoken_map
        f_item['token_is_max_context'] = token_is_max_context

        # The item will not include query.
        ready_training_list.append(f_item)
        # each item in the list don't have query, we append the query latter outside the function.

    return ready_training_list


# Deprecated
def eitems_to_fitems(eitem_list, tokenizer, is_training, max_tokens_for_doc=384, doc_stride=128):
    fitem_dict = {}
    fitem_list = []  # The output of all fitems
    for item in tqdm(eitem_list):
        f_items = preprocssing_span_prediction_item(item, tokenizer, is_training,
                                                    max_tokens_for_doc,
                                                    doc_stride)
        for fitem in f_items:
            query = item['query']
            fitem['o_query'] = query
            fitem['gt_answer_text'] = item['gt_answer_text']
            fitem_dict[fitem['fid']] = fitem
            fitem_list.append(fitem)

    return fitem_dict, fitem_list


def preprocessing_squad(data_set, is_training=True):
    article_list = data_set['data']
    d_list = []
    for article in article_list:
        # print(article.keys()) # title, paragraphs
        title = article['title']
        for paragraph in article['paragraphs']:
            # print(paragraph.keys())
            context = paragraph['context']  # context, qas
            # Firstly, we do whitespace tokenization
            context_w_tokens, w_tokens_char_to_word_offset = span_utils.w_processing(context)
            for question in paragraph['qas']:
                uid = question['id']  # id, question, answers
                query = question['question']
                answers = question['answers']
                no_answer = question['is_impossible'] if 'is_impossible' in question else False
                is_yes_no_question = False
                # Randomly, selection one answer
                if len(answers) < 1:
                    assert no_answer
                    # continue    # if there is no answer, we just ignore.

                if not no_answer:
                    answer = answers[0]  # answer_start, text
                    answer_start = answer['answer_start']
                    answer_len = len(answer['text'])
                    answer_text = answer['text']
                    answer_end = answer_start + answer_len
                else:
                    answer = None
                    answer_start = -1
                    answer_end = -1
                    answer_len = 0
                    answer_text = ""
                # no_answer = is_impossible

                _, adjusted_answer_start, adjusted_answer_end = span_utils.pair_w_tokens_with_ground_truth_span(
                    context_w_tokens, w_tokens_char_to_word_offset,
                    answer_text, answer_start, no_answer, is_yes_no_question, is_training
                )  # The start and end are both inclusive.

                if adjusted_answer_start is None and not no_answer:
                    continue
                    # The question should be answerable but
                    # we can not find answer span even from whitespace tokenized sequence, there might be a problem

                if adjusted_answer_start == -1:
                    no_answer = True
                # print(adjusted_answer_start, adjusted_answer_end)

                # Then we have what we need.
                item = dict()
                item['article_title'] = title
                item['uid'] = uid
                item['w_token_context'] = context_w_tokens
                item['gt_answer_text'] = answer_text
                item['query'] = query
                item['answer_start'] = adjusted_answer_start
                item['answer_end'] = adjusted_answer_end
                item['no_answer'] = no_answer
                item['is_yes_no_question'] = is_yes_no_question

                d_list.append(item)

    return d_list


def get_squad_question_selection_forward_list(data_set):
    article_list = data_set['data']
    d_list = []
    for article in article_list:
        # print(article.keys()) # title, paragraphs
        title = article['title']
        for paragraph in article['paragraphs']:
            # print(paragraph.keys())
            context = paragraph['context']  # context, qas
            # Firstly, we do whitespace tokenization
            for question in paragraph['qas']:
                uid = question['id']  # id, question, answers
                query = question['question']
                answers = question['answers']
                no_answer = question['is_impossible'] if 'is_impossible' in question else False
                is_yes_no_question = False
                # Randomly, selection one answer
                fitem = dict()
                fitem['qid'] = str(uid)  # query id
                fid = str(uuid.uuid4())
                fitem['fid'] = fid  # forward id
                fitem['query'] = query

                fitem['context'] = context
                fitem['element'] = "Not used"
                fitem['s_labels'] = 'true' if not no_answer else 'false'

                d_list.append(fitem)

    return d_list


def get_squad_question_answer_list(data_set):
    article_list = data_set['data']
    d_list = []
    random_dict = dict()
    for article in article_list:
        # print(article.keys()) # title, paragraphs
        title = article['title']
        for paragraph in article['paragraphs']:
            # print(paragraph.keys())
            context = paragraph['context']  # context, qas
            # Firstly, we do whitespace tokenization
            for question in paragraph['qas']:
                uid = question['id']  # id, question, answers
                query = question['question']
                answers = question['answers']
                no_answer = question['is_impossible'] if 'is_impossible' in question else False
                is_yes_no_question = False
                # Randomly, selection one answer
                answer_text = answers[0]['text']
                fitem = dict()
                fitem['qid'] = str(uid)  # query id
                fid = str(uuid.uuid4())
                fitem['answers'] = answer_text

                random_dict[str(uid)] = answer_text

                d_list.append(fitem)

    return d_list, random_dict


if __name__ == '__main__':
    squad_v11 = common.load_json(config.SQUAD_TRAIN_1_1)
    squad_pos_fitems = get_squad_question_selection_forward_list(squad_v11)
    print(len(squad_pos_fitems))
