import uuid

import span_prediction_task_utils.common_utils as span_utils
import collections
from tqdm import tqdm
from utils import list_dict_data_tool

SPECIAL_TOKEN_LIST = ('[CLS]', 'yes', 'no')


def _check_is_max_context(doc_spans, cur_span_index, position):
    # cf. hugging face.
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


def preprocssing_span_prediction_item_paired(item, tokenizer, is_training,
                                             max_tokens_for_doc, max_query_length, doc_stride,
                                             appended_special_token_list=SPECIAL_TOKEN_LIST):
    ready_training_list = []

    c_tokens, c2w_index, w2c_index, c_start_position, c_end_position = span_utils.w_tokens2c_tokens(
        item['w_token_context'], item['answer_start'], item['answer_end'],
        item['gt_answer_text'], tokenizer, item['no_answer'], item['is_yes_no_question'], is_training
    )

    if c_tokens is None and c2w_index is None and w2c_index is None and c_start_position is None and c_end_position is None:
        return ready_training_list

    special_mapping = dict()
    for i, token_str in enumerate(appended_special_token_list):
        special_mapping[i] = token_str

    query = item['query']
    query_ctokens = tokenizer.tokenize(query)

    if len(query_ctokens) > max_query_length:
        query_ctokens = query_ctokens[0:max_query_length]

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
        # We first append special tokens and query tokens
        fc_token = []
        segment_ids = []
        for token in appended_special_token_list:
            fc_token.append(token)  # [CLS] yes no
            segment_ids.append(0)
        # Then we append all query tokens
        for token in query_ctokens:
            fc_token.append(token)
            segment_ids.append(0)
        fc_token.append("[SEP]")
        segment_ids.append(0)
        # query token append finished.

        start_position = None
        end_position = None

        fctoken_to_wtoken_map = {}  # we need a map to the original w-tokens

        # This the the code from Google BERT where we keep track of the score of current token.
        # This is only for sliding window approach.
        token_is_max_context = {}

        # We do this for all the document. Bc this is not that time-consuming. 2019-03-14
        # if not is_training:     # We only need to do this during inference  # But maybe not. Check laters

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            fctoken_to_wtoken_map[len(fc_token)] = c2w_index[split_token_index]
            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(fc_token)] = is_max_context
            fc_token.append(c_tokens[split_token_index])
            segment_ids.append(1)
            # The index offset is 0.
            # the_index = doc_span.start + i
            # fctoken_to_wtoken_map[i] = c2w_index[doc_span.start + i]
            # split_token_index = doc_span.start + i
            # is_max_context = _check_is_max_context(doc_spans, doc_span_index,
            #                                        split_token_index)
            # token_is_max_context[i] = is_max_context

        fc_token.append("[SEP]")
        segment_ids.append(1)

        if is_training and not item['no_answer'] and not item['is_yes_no_question']:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (c_start_position >= doc_start and c_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                no_answer = True
                start_position = 0
                end_position = 0  # if out of span then, no_answers
            else:
                # doc_offset = len(query_tokens) + 2
                no_answer = False
                doc_offset = len(appended_special_token_list) + len(query_ctokens) + 1
                # special CLS, yes, no, [query], seq
                start_position = c_start_position - doc_start + doc_offset
                end_position = c_end_position - doc_start + doc_offset
                # We change this from original code base, here the doc_offset will only w.r.t. the context.

        if is_training and item['no_answer']:
            no_answer = True
            start_position = 0
            end_position = 0

        if is_training and 'is_yes_no_question' in item and item['is_yes_no_question']:
            no_answer = False
            if item['gt_answer_text'].lower() == 'yes':
                start_position = 1
                end_position = 1
            elif item['gt_answer_text'].lower() == 'no':
                start_position = 2
                end_position = 2
            else:
                raise ValueError('Answers of Yes/No Question can only be yes or no.')

        if not is_training:
            # no_answer = True
            start_position = -1  # -2 indicates that the start and end position is hidden.
            end_position = -1

        fid = str(uuid.uuid4())  # This is unique forward id for each of the doc that will be feeded into net.
        uid = item['uid']

        f_item = dict()
        f_item['fid'] = fid
        f_item['uid'] = uid
        f_item['doc_span_index'] = doc_span_index
        f_item['w_token_context'] = item['w_token_context']
        f_item['paired_c_tokens'] = fc_token
        f_item['segment_ids'] = segment_ids
        f_item['no_answer'] = no_answer
        f_item['start_position'] = start_position
        f_item['end_position'] = end_position
        f_item['fctoken_to_wtoken_map'] = fctoken_to_wtoken_map
        f_item['token_is_max_context'] = token_is_max_context
        f_item['special_position_mapping'] = special_mapping
        if 'additional_fields' in item:
            f_item['additional_fields'] = item['additional_fields']
        # if f_item['doc_span_index'] == 1:
        #     pass
        # The item will not include query.
        ready_training_list.append(f_item)
        # each item in the list don't have query, we append the query latter outside the function.
    return ready_training_list


def eitems_to_fitems(eitem_list, tokenizer, is_training, max_tokens_for_doc=320, max_query_length=64, doc_stride=128, debug=False):
    fitem_list = []  # The output of all fitems
    if debug:
        eitem_list = eitem_list[:100]

    for item in tqdm(eitem_list):
        f_items = preprocssing_span_prediction_item_paired(item, tokenizer, is_training,
                                                           max_tokens_for_doc, max_query_length,
                                                           doc_stride)
        fitem_list.extend(f_items)

    fitem_dict = list_dict_data_tool.list_to_dict(fitem_list, 'fid')

    return fitem_dict, fitem_list
