# The problem of span prediction is that we are given a context and we want to extract some span from the context
# that match the ground truth span based on some queries such as a question in SQuAD setting.
# Typical NLP models usually don't process the raw textual at character-level but rather tokenize the input text
# to a list of token and process it at some token-level depending on what the specific tokenizer it use.
# This tokenization process might cause some problems when the span of token can not exactly match the ground truth
# character-level span because of the boundaries mismatching.
# In this script, we try to solve this potential problem with a unified framework.


# The procedure for preprocess of span prediction can be explain the in the following several steps:
# (1). We first tokenize the context using whitespace-tokenization. Let's define it as w-tokens.
# The input test will only to split by white space so that some span of w-tokens will guarantee to
# cover the ground truth span.
# The output of the fist procedure will be:
#   1. A list of w-tokens.
#   2. A tuple of two integer indicating the start and end position of the ground truth span w.r.t w-tokens.
#   3. Ground truth span text (For later usage.)
#   4. The query text.

# (2). NLP models sometimes use different tokenizer for different task, so we should be able to handle customized
# tokenizer for this same task. Therefore, in this step, we will convert each w-token into sub-tokens with any tokenizer
# and meanwhile correctly output the correct start and end position in the new s-tokens.
# The output of this procedure will be:
#   1. A list of integer indicating the start indices (or start position) of each w-tokens in s-tokens list.
#   2. A list of integer indicating the indices (or position) of each s-tokens in w-tokens list.
#   3. A new tuple of two integer indicating the start and end position of the ground truth span w.r.t s-token list.
#   *. Potential improve the answer span with sub-token matching.

# Optional procedure. sliding window processing.
import collections
import logging
from typing import List, Tuple
import sys
import re

from pytorch_pretrained_bert import BasicTokenizer

s = ''.join(chr(c) for c in range(sys.maxunicode + 1))
all_white_space = set(re.findall(r'\s', s))

logger = logging.getLogger(__name__)


def w_processing(context):
    # This is the method of white-space tokenization.
    # We process the context to get w_tokens and char_to_word_offset
    # char_to_word_offset have length equals the total number of characters in the context: each element indicate the
    # position index of the word (in the w-tokens) to which the current character belongs to.
    paragraph_text = context
    doc_tokens = []  # Here doc_tokens indicate w-tokens
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    return doc_tokens, char_to_word_offset


def pair_w_tokens_with_ground_truth_span(w_tokens, w_char_to_word_offset, ground_truth_answer_text, span_offset_start,
                                         no_answer=False, is_yes_no_question=False, is_training=False,
                                         do_checking=False):
    """
    :param w_tokens:                    The white-space tokenzied input context.
    :param w_char_to_word_offset:       Total length is the character length of the context.
                                        Each element is the context char to word index in w-tokens.
    :param ground_truth_answer_text:    The GT answer span.
    :param span_offset_start:           The GT span start in the original context.
    :param no_answer:                   If the question is non-answerable.
    :param is_yes_no_question:
    :param is_training:
    :return:
    w_token: Same as input,
    start_position: The start position of the answer span in w.r.t. w-tokens. (Inclusive)
    end_position:   The end position of the answer span in w.r.t. w-tokens. (Inclusive)
    """
    # Here we just try to pair the w_tokens with the ground truth span
    # The output will be the w-tokens and adjusted start and end position of the ground truth span.
    # If we could not find the answer, we will return None.

    # If the answer is non-answerable, give the ground_truth_span = None
    doc_tokens = w_tokens
    char_to_word_offset = w_char_to_word_offset
    start_position = None
    end_position = None
    orig_answer_text = ground_truth_answer_text

    if no_answer or is_yes_no_question:
        # Non-answerable question
        start_position = -1
        end_position = -1
        orig_answer_text = ""

    elif orig_answer_text is not None and not no_answer and not is_yes_no_question and is_training:
        # if the orig_answer_text is None, which means the span is None or Non-answerable
        answer_offset = span_offset_start
        answer_length = len(orig_answer_text)
        start_position = char_to_word_offset[answer_offset]
        end_position = char_to_word_offset[answer_offset + answer_length - 1]
        # Only add answers where the text can be exactly recovered from the
        # document. If this CAN'T happen it's likely due to weird Unicode
        # stuff so we will just skip the example.
        #
        # Note that this means for training mode, every example is NOT
        # guaranteed to be preserved.
        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
        cleaned_answer_text = " ".join(
            whitespace_tokenize(orig_answer_text))

        if actual_text.find(cleaned_answer_text) == -1 and is_training:
            logger.warning("Can not find answer span from whitespace-tokenized context, there might be a problem")
            logger.warning("Could not find answer: '%s' vs. '%s'",
                           actual_text, cleaned_answer_text)
            # If we could not find answer, we return None
            if do_checking:
                return None, None, None
            else:
                return doc_tokens, start_position, end_position
    else:
        # Non-answerable question
        start_position = -1
        end_position = -1
        orig_answer_text = ""

    return doc_tokens, start_position, end_position


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == '\u200e' \
            or c in all_white_space or c == 'ِ' or c == '₿' or c == '':
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def w_tokens2c_tokens(w_tokens, w_start_position, w_end_position, gt_answer_text,
                      tokenizer, no_answer=False, is_yes_no_question=False, is_training=True):
    """
    This method convert w-tokens to c-tokens (customized tokenzied tokens.) depend on the given tokenizer.

    :param w_tokens:                The w-tokens
    :param w_start_position:        Start position of w.r.t. w-tokens
    :param w_end_position:          End position of w.r.t. w-tokens
    :param gt_answer_text:          Ground truth answer text
    :param tokenizer:               The tokenizer
    :param no_answer:               Whether the context contains the answer.
    :param is_training:
    :return:

        c_tokens,               The c-tokens on top of w-tokens with customized tokenizer.
        c_to_w_index,           The index from c-tokens to w-tokens.
        w_to_c_index,           The start index from w-tokens to c-tokens.
        c_start_position,       Start position of w.r.t. c-tokens
        c_end_position          End position of w.r.t. c-tokens
    """
    c_to_w_index = []  # Total length: len(c_tokens)
    w_to_c_index = []  # Total length: len(w_tokens)
    c_tokens = []

    # print(w_tokens)
    for (i, token) in enumerate(w_tokens):
        w_to_c_index.append(len(c_tokens))
        sub_tokens = tokenizer.tokenize(token)
        # print(w_tokens)
        # print(token)
        # print(sub_tokens)
        if len(sub_tokens) == 0:
            return None, None, None, None, None     # Error skip this example, change later?
            raise ValueError("Number of sub-tokens should be greater than 0.")
        for sub_token in sub_tokens:  # Make sure that token will not be empty
            c_to_w_index.append(i)
            c_tokens.append(sub_token)

    c_start_position = None
    c_end_position = None
    if is_training and no_answer:
        c_start_position = -1
        c_end_position = -1

    if is_training and is_yes_no_question:
        c_start_position = -1
        c_end_position = -1

    if is_training and not no_answer:
        c_start_position = w_to_c_index[w_start_position]
        if w_end_position < len(w_tokens) - 1:
            c_end_position = w_to_c_index[w_end_position + 1] - 1
        else:
            c_end_position = len(c_tokens) - 1
        (c_start_position, c_end_position) = _improve_answer_span(
            c_tokens, c_start_position, c_end_position, tokenizer,
            gt_answer_text)

    return c_tokens, c_to_w_index, w_to_c_index, c_start_position, c_end_position


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    # The code is originally from hugging face squad preprocessing.
    # We will need to use the code when we use tokenizer to obtain subtokens and rematch with ground truth answer span
    """Returns tokenized answer spans that better match the annotated answer."""
    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def logits_to_span_score(start_logits, end_logits, context_length, max_span_length=30):
    """
    This is batched operation that convert start logits and end logtis to a list of scored, span.
    Each example of the two logits will also depends on the context length.

    :param start_logits:        # [B, T]
    :param end_logits:          # [B, T]
    :param context_length:      # [B]
    :return:
    """
    # pass
    batch_size = start_logits.size(0)
    s_logits = start_logits.tolist()
    e_logits = end_logits.tolist()
    c_l = context_length.tolist()
    b_scored_span_list: List[List[Tuple[float, float, Tuple[int, int]]]] = []
    if max_span_length is None:
        max_span_length = 1000

    for b_i in range(batch_size):
        cur_l = c_l[b_i]
        cur_s_logits = s_logits[b_i][:cur_l]
        cur_e_logits = e_logits[b_i][:cur_l]
        cur_span_list: List[Tuple[float, float, Tuple[int, int]]] = []

        for s_i in range(len(cur_s_logits)):
            for e_i in range(s_i, min(len(cur_e_logits), s_i + max_span_length)):
                s_score = cur_s_logits[s_i]
                e_score = cur_e_logits[e_i]
                assert s_i <= e_i
                cur_span_list.append((s_score, e_score, (s_i, e_i)))

        cur_span_list = sorted(cur_span_list, key=lambda x: x[0] + x[1], reverse=True)  # Start + End logits
        b_scored_span_list.append(cur_span_list)

    return b_scored_span_list


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # This method is borrowed from hugging-face BERT codebase.

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def merge_predicted_fitem_to_eitem(fitem_dict, pre_etiem_dict=None, pred_no_answer=False, no_answer_threshold=0.0):
    """
    This method helps merge the forward item to prediction file.
    Because the sliding window natural, one context-question pair might be converted to multiple forward item passing through the network.
    We need to merge them before we do evaluation. This usually require some comparing method to choose the best prediction along all the forward outputs..

    This method can be generally useful to many scenario.
    :param fitem_dict:
    :return:
    """
    eitem_dict = {}
    for k, fitem in fitem_dict.items():
        uid = fitem['uid']
        if uid in eitem_dict:
            eitem_dict[uid]['top_spans'].extend(fitem['top_spans'])
        else:
            if pre_etiem_dict is not None:
                eitem_dict[uid].update(pre_etiem_dict[uid])
            else:
                eitem_dict[uid] = {}
                eitem_dict[uid]['top_spans'] = fitem['top_spans']

    eval_dict = {}
    for k, eitem in eitem_dict.items():
        eitem_dict[k]['top_spans'] = list(sorted(eitem['top_spans'], key=lambda x: (x.start_logit + x.end_logit),
                                                 reverse=True))
        # yes_item, no_item, no_answer_item = None, None, None
        no_answer_min_score = 1_000_000
        best_yes_score = -1_000_000
        best_no_score = -1_000_000
        for cur_span in eitem_dict[k]['top_spans']:
            # "text", "c_token_start", "c_token_end", "start_logit", "end_logit"
            if cur_span.text == "" and \
                    cur_span.c_token_start == cur_span.c_token_end and cur_span.c_token_start == 0:
                sum_score = cur_span.start_logit + cur_span.end_logit
                no_answer_min_score = min(sum_score, no_answer_min_score)
            elif cur_span.text == "yes" \
                    and cur_span.c_token_start == cur_span.c_token_end and cur_span.c_token_start == 1:
                best_yes_score = max(best_yes_score, cur_span.start_logit + cur_span.end_logit)
            elif cur_span.text == "no" \
                    and cur_span.c_token_start == cur_span.c_token_end and cur_span.c_token_start == 2:
                best_no_score = max(best_no_score, cur_span.start_logit + cur_span.end_logit)

        assert no_answer_min_score != 1_000_000  # no_answer_min_score need to be updated
        assert best_yes_score != -1_000_000  # yes_score need to be updated
        assert best_no_score != -1_000_000  # no_score need to be updated

        eitem_dict[k]['best_yes_score'] = best_yes_score
        eitem_dict[k]['best_no_score'] = best_no_score
        eitem_dict[k]['best_pred_span'] = eitem['top_spans'][0].text
        eitem_dict[k]['best_pred_span_score'] = eitem['top_spans'][0].start_logit + eitem['top_spans'][0].end_logit
        eitem_dict[k]['min_no_answer_score'] = no_answer_min_score

        if pred_no_answer and no_answer_min_score - eitem_dict[k]['best_pred_span_score'] > no_answer_threshold:
            eval_dict[k] = ""  # Predict no answer
        else:
            eval_dict[k] = eitem_dict[k]['best_pred_span']

    return eitem_dict, eval_dict


def write_to_predicted_fitem(start_logits, end_logits, context_length, b_fids, b_uids, gt_span,
                             fitem_dict, output_fitem_dict, do_lower_case, top_n_span=30):
    """
    We convert the output of networks together with the original fitem to new output fitem.
    This new output fitem will later to be mapped to original examples using "merge_predicted_fitem_to_eitem" method above.

    :param start_logits:            A batch of start logits.
    :param end_logits:              A batch of end logits.
    :param context_length:          A batch of int indicating the length of the context.
    :param b_fids:                  A batch of fids.
    :param b_uids:                  A batch of uids.
    :param gt_span:                 A batch of ground truth spans.

        All the parameter above should be order preserved.

    :param fitem_dict:              A the fitem we built before passing through the networks. This is just for reference.
    :param output_fitem_dict:       The output fitem dict we want to write to
    :return:
    """

    # b_fids = batch['fid']
    # b_uids = batch['uid']
    # print(gt_span)

    b_scored_spans = logits_to_span_score(start_logits, end_logits, context_length)
    # print(b_scored_spans[:5])

    batch_size = start_logits.size(0)

    NbestSpanPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestSpanPrediction", ["text", "c_token_start", "c_token_end", "start_logit", "end_logit"])

    for b_i in range(batch_size):
        new_fitem = {}

        fid = b_fids[b_i]
        uid = b_uids[b_i]
        assert uid == fitem_dict[fid]['uid']  # Check mapping correctness.

        scored_span = b_scored_spans[b_i]

        gt_c_start, gt_c_end = -1, -1

        if gt_span is not None:
            b_gt_span = gt_span[b_i]
            gt_c_start = int(b_gt_span[0])
            gt_c_end = int(b_gt_span[1])

        c_l = int(context_length[b_i])
        fitem = fitem_dict[fid]
        special_position_mapping = fitem['special_position_mapping']
        fctoken_to_wtoken_map = fitem['fctoken_to_wtoken_map']
        paired_c_tokens = fitem['paired_c_tokens']
        w_token_context = fitem['w_token_context']

        # We do the max context filtering here to prevent any further confusion.
        token_is_max_context = fitem['token_is_max_context']
        filtered_scored_span = []
        special_scored_span = []

        for start_s, end_s, (c_token_start, c_token_end) in scored_span:
            if c_token_start == c_token_end and c_token_start in special_position_mapping:
                special_scored_span.append((start_s, end_s, (c_token_start, c_token_end)))
                # We append (0,0), (1,1), (2,2)
                continue

            if token_is_max_context.get(c_token_start, False) \
                    and c_token_start in fctoken_to_wtoken_map and c_token_end in fctoken_to_wtoken_map:
                filtered_scored_span.append((start_s, end_s, (c_token_start, c_token_end)))

        scored_span = filtered_scored_span
        # Max context filtering end

        scored_span = sorted(scored_span, key=lambda x: x[0] + x[1], reverse=True)  # Start + End logits
        top30_scored_span = scored_span[:top_n_span] + special_scored_span

        output_top_spans = []
        seen_final_text = set()

        for cur_span in top30_scored_span:
            # top_span = top30_scored_span[0]
            start_logits = cur_span[0]
            end_logits = cur_span[1]
            c_token_start, c_token_end = cur_span[2]

            if c_token_start == c_token_end and c_token_start in special_position_mapping.keys():
                if c_token_start == 0:
                    final_text = ""
                    output_top_spans.append(
                        NbestSpanPrediction(final_text, c_token_start, c_token_end, start_logits, end_logits))
                    seen_final_text.add(final_text)
                elif c_token_start == 1:
                    final_text = "yes"
                    output_top_spans.append(
                        NbestSpanPrediction(final_text, c_token_start, c_token_end, start_logits, end_logits))
                    seen_final_text.add(final_text)
                elif c_token_end == 2:
                    final_text = "no"
                    output_top_spans.append(
                        NbestSpanPrediction(final_text, c_token_start, c_token_end, start_logits, end_logits))
                    seen_final_text.add(final_text)
                else:
                    raise ValueError(f"Unexpected special position value {c_token_start}.")
                continue
            # print(c_token_start, c_token_end)
            # fctoken_to_wtoken_map = fitem['fctoken_to_wtoken_map']

            c_token_pred_answer_span = paired_c_tokens[c_token_start:c_token_end + 1]
            w_token_start = fctoken_to_wtoken_map[c_token_start]
            w_token_end = fctoken_to_wtoken_map[c_token_end]
            w_token_pred_answer_span = w_token_context[w_token_start:w_token_end + 1]
            gt_c_answer_span = paired_c_tokens[gt_c_start:gt_c_end + 1]

            c_cat_text = " ".join(c_token_pred_answer_span)

            # De-tokenize WordPieces that have been split off.
            c_cat_text = c_cat_text.replace(" ##", "")
            c_cat_text = c_cat_text.replace("##", "")

            # Clean whitespace
            c_cat_text = c_cat_text.strip()
            c_cat_text = " ".join(c_cat_text.split())
            w_cat_text = " ".join(w_token_pred_answer_span)

            final_text = get_final_text(c_cat_text, w_cat_text, do_lower_case)

            # if final_text not in seen_final_text:
            output_top_spans.append(
                NbestSpanPrediction(final_text, c_token_start, c_token_end, start_logits, end_logits))
            seen_final_text.add(final_text)

        new_fitem['top_spans'] = output_top_spans
        new_fitem['fid'] = fitem['fid']
        new_fitem['uid'] = fitem['uid']
        if 'additional_fields' in fitem:
            new_fitem['additional_fields'] = fitem['additional_fields']

        # new_fitem.update(fitem)
        # assert fid not in new_fitem
        output_fitem_dict[fid] = new_fitem

    return output_fitem_dict


def check_pred_output(start_logits, end_logits, context_length, b_fids, b_uids, gt_span, fitem_dict):
    # b_fids = batch['fid']
    # b_uids = batch['uid']
    # print(gt_span)

    b_scored_spans = logits_to_span_score(start_logits, end_logits, context_length)
    # print(b_scored_spans[:5])

    batch_size = start_logits.size(0)
    for b_i in range(batch_size):
        fid = b_fids[b_i]
        scored_span = b_scored_spans[b_i]

        gt_c_start, gt_c_end = -1, -1

        if gt_span is not None:
            b_gt_span = gt_span[b_i]
            gt_c_start = int(b_gt_span[0])
            gt_c_end = int(b_gt_span[1])

        c_l = int(context_length[b_i])
        fitem = fitem_dict[fid]
        top10_scored_span = scored_span[:10]
        top_span = top10_scored_span[0]
        c_token_start, c_token_end = top_span[2]
        # print(c_token_start, c_token_end)
        context_c_tokens = fitem['context_c_tokens']
        context_w_tokens = fitem['context_w_tokens']
        fctoken_to_wtoken_map = fitem['fctoken_to_wtoken_map']
        token_is_max_context = fitem['token_is_max_context']

        c_token_pred_answer_span = context_c_tokens[c_token_start:c_token_end + 1]
        w_token_start = fctoken_to_wtoken_map[c_token_start]
        w_token_end = fctoken_to_wtoken_map[c_token_end]
        w_token_pred_answer_span = context_w_tokens[w_token_start:w_token_end + 1]
        gt_c_answer_span = context_c_tokens[gt_c_start:gt_c_end + 1]

        print("C Pred Answer:", c_token_pred_answer_span)
        print("W Pred Answer:", w_token_pred_answer_span)
        print("C GT Answer:", gt_c_answer_span)
        print("GT span:", gt_c_start, gt_c_end)
        print("Pred span:", c_token_start, c_token_end)
