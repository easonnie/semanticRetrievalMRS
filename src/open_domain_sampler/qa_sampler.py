import copy
import json
import random
import span_prediction_task_utils.common_utils as span_utils

from pytorch_pretrained_bert import BertTokenizer

from build_rindex import raw_text_db
from open_domain_sampler import od_sample_utils
from span_prediction_task_utils import span_preprocess_tool
from evaluation import ext_hotpot_eval
from hotpot_fact_selection_sampler.sampler_utils import select_top_k_and_to_results_dict
from span_prediction_task_utils.span_match_and_label import MatchObject, ContextAnswerMatcher, regex_match_and_get_span
from span_prediction_task_utils.span_preprocess_tool import eitems_to_fitems
from span_prediction_task_utils.squad_utils import preprocessing_squad
from utils import common, list_dict_data_tool
import config
from utils.text_process_tool import normalize
from wiki_util import wiki_db_tool


def format_convert(items, is_training):
    ready_items = []
    for item in items:
        ready_item = dict()
        answer_text = item['answer']
        answer_start = item['answer_start_index']
        no_answer = item['no_answer']
        context = item['context']
        if context is None or len(context) == 0:
            context = 'empty'
        is_yes_no_question = item['is_yes_no_question']

        context_w_tokens, w_tokens_char_to_word_offset = span_utils.w_processing(context)
        _, adjusted_answer_start, adjusted_answer_end = span_utils.pair_w_tokens_with_ground_truth_span(
            context_w_tokens, w_tokens_char_to_word_offset,
            answer_text, answer_start, no_answer, is_yes_no_question, is_training, do_checking=False
        )

        ready_item['uid'] = item['qid']
        ready_item['w_token_context'] = context_w_tokens
        ready_item['gt_answer_text'] = answer_text
        ready_item['query'] = item['query']
        ready_item['answer_start'] = adjusted_answer_start
        ready_item['answer_end'] = adjusted_answer_end
        ready_item['no_answer'] = no_answer
        ready_item['is_yes_no_question'] = is_yes_no_question

        if 'additional_fields' in item:
            ready_item['additional_fields'] = item['additional_fields']

        ready_items.append(ready_item)

    assert len(ready_items) == len(items)

    return ready_items


def build_open_qa_forword_item(p_level_results, d_list, is_training, db_cursor, match_type):
    o_forward_items = []
    for item in d_list:
        qid = item['question']
        query = item['question']
        # o_contexts = item['context']
        answers = item['answers'] if 'answers' in item else "##hidden##"
        # q_type = item['']

        retrieved_paragraph_indices = p_level_results['pred_p_list'][qid]  # remember the key here.
        selected_scored_paragraph = p_level_results['selected_scored_results'][qid]

        scored_p_dict = dict()
        for cur_score, (title, p_num) in selected_scored_paragraph:
            scored_p_dict[(title, p_num)] = cur_score

        for p_title, p_num in retrieved_paragraph_indices:
            # Loop for paragraph
            p_list = raw_text_db.query_raw_text(db_cursor, p_title, p_num=p_num)
            assert len(p_list) == 1
            std_title, p_num, p_sentences = p_list[0]
            paragraph_text = ' '.join(json.loads(p_sentences))

            context = paragraph_text

            for answer in answers:
                if len(answer) <= 2:
                    # print(answer)
                    # print(len(answer))
                    continue
                answer_start_list = regex_match_and_get_span(paragraph_text, answer, type=match_type)

                example_item = dict()
                example_item['query'] = query
                # if query == 'If Roman numerals were used, what would Super Bowl 50 have been called?':
                #     print("Found")

                example_item['qid'] = qid
                example_item['context'] = context
                example_item['no_answer'] = False
                example_item['is_yes_no_question'] = False

                example_item['answer'] = answer
                if is_training:
                    # Yes or no question:
                    if answer.lower() == 'yes':
                        example_item['no_answer'] = False
                        example_item['answer_start_index'] = -2
                        example_item['is_yes_no_question'] = True
                    elif answer.lower() == 'no':
                        example_item['no_answer'] = False
                        example_item['answer_start_index'] = -3
                        example_item['is_yes_no_question'] = True
                    elif len(answer_start_list) > 0:
                        example_item['no_answer'] = False
                        random_selected_index = random.choice(answer_start_list)
                        start = random_selected_index.span()[0]
                        end = random_selected_index.span()[1]
                        new_answer = context[start:end]
                        # print(context[start:end], answers)
                        example_item['answer_start_index'] = start
                        example_item['is_yes_no_question'] = False
                        example_item['answer'] = new_answer
                    elif len(answer_start_list) == 0:
                        example_item['no_answer'] = True
                        example_item['answer_start_index'] = -1
                        example_item['is_yes_no_question'] = False
                else:
                    example_item['no_answer'] = False
                    example_item['answer_start_index'] = random.choice(answer_start_list) if len(
                        answer_start_list) > 0 else -4
                    example_item['is_yes_no_question'] = False

                additional_fields = dict()
                # Important: Remember the field value!
                # (score, (p_title, p_num))
                additional_fields['p_level_scored_results'] = [scored_p_dict[(p_title, p_num)], [p_title, p_num]]

                example_item['additional_fields'] = additional_fields

                o_forward_items.append(example_item)

    return o_forward_items


def get_open_qa_item_with_upstream_paragraphs(d_list, cur_eval_results_list, is_training,
                                              tokenizer: BertTokenizer, max_context_length, max_query_length,
                                              doc_stride=128,
                                              debug_mode=False, top_k=10, filter_value=0.1, match_type='string'):
    t_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_RAW_TEXT)

    if debug_mode:
        d_list = d_list[:100]
        id_set = set([item['question'] for item in d_list])
        cur_eval_results_list = [item for item in cur_eval_results_list if item['qid'] in id_set]

    d_o_dict = list_dict_data_tool.list_to_dict(d_list, 'question')
    copied_d_o_dict = copy.deepcopy(d_o_dict)

    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_d_o_dict,
                                                          'qid', 'fid', check=True)

    cur_results_dict_top10 = od_sample_utils.select_top_k_and_to_results_dict(copied_d_o_dict,
                                                                              score_field_name='prob',
                                                                              top_k=top_k, filter_value=filter_value)

    forward_example_items = build_open_qa_forword_item(cur_results_dict_top10, d_list, is_training, t_cursor,
                                                       match_type)
    forward_example_items = format_convert(forward_example_items, is_training)
    fitems_dict, read_fitems_list = span_preprocess_tool.eitems_to_fitems(forward_example_items, tokenizer, is_training,
                                                                          max_context_length,
                                                                          max_query_length, doc_stride, False)

    return fitems_dict, read_fitems_list, cur_results_dict_top10['pred_p_list']


def inspect_upstream_eval():
    is_training = True
    debug_mode = True
    d_list = common.load_jsonl(config.OPEN_SQUAD_DEV_GT)
    in_file_name = config.PRO_ROOT / 'saved_models/05-12-08:44:38_mtr_open_qa_p_level_(num_train_epochs:3)/i(2000)|e(2)|squad|top10(0.6909176915799432)|top20(0.7103122043519394)|seed(12)_eval_results.jsonl'
    cur_eval_results_list = common.load_jsonl(in_file_name)
    top_k = 10
    filter_value = 0.1
    t_cursor = wiki_db_tool.get_cursor(config.WHOLE_WIKI_RAW_TEXT)
    match_type = 'string'

    if debug_mode:
        d_list = d_list[:100]
        id_set = set([item['question'] for item in d_list])
        cur_eval_results_list = [item for item in cur_eval_results_list if item['qid'] in id_set]

    d_o_dict = list_dict_data_tool.list_to_dict(d_list, 'question')
    copied_d_o_dict = copy.deepcopy(d_o_dict)

    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_d_o_dict,
                                                          'qid', 'fid', check=True)

    cur_results_dict_top10 = od_sample_utils.select_top_k_and_to_results_dict(copied_d_o_dict,
                                                                              score_field_name='prob',
                                                                              top_k=top_k, filter_value=filter_value)

    forward_example_items = build_open_qa_forword_item(cur_results_dict_top10, d_list, is_training, t_cursor,
                                                       match_type)

    print(forward_example_items)
    # print(len(cur_results_dict_top10))

    # list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
    #                                                                cur_results_dict_top10,
    #                                                                'qid', 'pred_p_list')
    #
    # t10_recall = open_domain_qa_eval.qa_paragraph_eval_v1(copied_dev_d_list, dev_list)
    # pass


def inspect_upstream_eval_v1():
    bert_model_name = "bert-base-uncased"
    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    do_lower_case = True

    max_pre_context_length = 315
    max_query_length = 64
    doc_stride = 128

    is_training = True
    debug_mode = True

    d_list = common.load_jsonl(config.OPEN_SQUAD_DEV_GT)
    in_file_name = config.PRO_ROOT / 'saved_models/05-12-08:44:38_mtr_open_qa_p_level_(num_train_epochs:3)/i(2000)|e(2)|squad|top10(0.6909176915799432)|top20(0.7103122043519394)|seed(12)_eval_results.jsonl'
    cur_eval_results_list = common.load_jsonl(in_file_name)
    top_k = 10
    filter_value = 0.1
    match_type = 'string'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                              cache_dir=bert_pretrain_path)

    fitems_dict, read_fitems_list, _ = get_open_qa_item_with_upstream_paragraphs(d_list, cur_eval_results_list, is_training,
                                                                                 tokenizer, max_pre_context_length, max_query_length, doc_stride,
                                                                                 debug_mode, top_k, filter_value, match_type)
    print(len(read_fitems_list))
    print(len(fitems_dict))


def inspect_sampler_squad_examples():
    bert_model_name = "bert-base-uncased"
    bert_pretrain_path = config.PRO_ROOT / '.pytorch_pretrained_bert'
    do_lower_case = True
    max_pre_context_length = 315
    max_query_length = 64
    doc_stride = 128
    debug = True

    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case,
                                              cache_dir=bert_pretrain_path)

    squad_train_v2 = common.load_json(config.SQUAD_TRAIN_2_0)

    train_eitem_list = preprocessing_squad(squad_train_v2)
    train_fitem_dict, train_fitem_list = eitems_to_fitems(train_eitem_list, tokenizer, is_training=False,
                                                          max_tokens_for_doc=max_pre_context_length,
                                                          doc_stride=doc_stride,
                                                          debug=debug)
    print(len(train_fitem_list))


if __name__ == '__main__':
    # inspect_sampler_squad_examples()
    # inspect_upstream_eval()
    inspect_upstream_eval_v1()
