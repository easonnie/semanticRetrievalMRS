import copy
import random
import span_prediction_task_utils.common_utils as span_utils

from pytorch_pretrained_bert import BertTokenizer
from span_prediction_task_utils import span_preprocess_tool
from evaluation import ext_hotpot_eval
from hotpot_fact_selection_sampler.sampler_utils import select_top_k_and_to_results_dict
from span_prediction_task_utils.span_match_and_label import MatchObject, ContextAnswerMatcher
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
        ready_items.append(ready_item)

    assert len(ready_items) == len(items)

    return ready_items


def build_qa_forword_item(sentence_level_results, d_list, is_training, db_cursor,
                          append_head=True, forward_type='random_one'):
    o_forward_items = []
    for item in d_list:
        qid = item['_id']
        query = item['question']
        # o_contexts = item['context']
        answer = item['answer'] if 'answer' in item else "##hidden##"
        # q_type = item['']

        retrieved_facts = sentence_level_results['sp'][qid]  # remember the key here.
        supporting_facts = []
        if is_training:
            supporting_facts = item['supporting_facts']

        additional_none_overlap_facts = []
        for fact in retrieved_facts:
            if fact not in supporting_facts:
                additional_none_overlap_facts.append(fact)

        selected_facts = additional_none_overlap_facts + supporting_facts

        # Retrieve all the documents:
        all_document = set([fact[0] for fact in selected_facts])
        sentid2sent_token_dict = dict()

        for doc in all_document:
            text_item = wiki_db_tool.get_item_by_key(db_cursor, key=doc)
            context = wiki_db_tool.get_first_paragraph_from_clean_text_item(text_item, flatten_to_paragraph=False,
                                                                            skip_first=True)
            for i, sentence_token in enumerate(context):
                # sentence_text = sentence_token
                if len(sentence_token) != 0:
                    sentid2sent_token_dict[(doc, i)] = sentence_token

        all_document = list(all_document)
        random.shuffle(all_document)
        # end

        shuffled_selected_facts = []
        selected_facts = sorted(selected_facts, key=lambda x: (x[0], x[1]))
        for doc in all_document:
            for fact in selected_facts:
                if fact[0] == doc:
                    shuffled_selected_facts.append(fact)

        assert len(shuffled_selected_facts) == len(selected_facts)

        cur_doc = None
        context_token_list = []
        for doc, i in shuffled_selected_facts:
            if (doc, i) not in sentid2sent_token_dict:
                print(f"Potential Error: {(doc, i)} not exists in DB.")
                continue
            # print((doc, i), sentid2sent_token_dict[(doc, i)])
            paragraph_token_list = sentid2sent_token_dict[(doc, i)]

            if cur_doc != doc and append_head and i != 0:
                for cur_token in doc.split(' ') + ['.'] + paragraph_token_list:
                    n_token = normalize(cur_token)
                    if n_token is not None and len(n_token) != 0:
                        context_token_list.append(n_token)
                # context_token_list = context_token_list + doc.split(' ') + ['.'] + paragraph_token_list
            else:
                for cur_token in paragraph_token_list:
                    n_token = normalize(cur_token)
                    if n_token is not None and len(n_token) != 0:
                        context_token_list.append(n_token)
                # context_token_list += paragraph_token_list

        context_matcher = ContextAnswerMatcher(context_token_list, uncase=True)
        context, answer_start_list = context_matcher.concate_and_return_answer_index(answer, match_type='left')

        example_item = dict()
        example_item['query'] = query
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
                example_item['answer_start_index'] = random.choice(answer_start_list)
                example_item['is_yes_no_question'] = False
            elif len(answer_start_list) == 0:
                example_item['no_answer'] = True
                example_item['answer_start_index'] = -1
                example_item['is_yes_no_question'] = False
        else:
            example_item['no_answer'] = False
            example_item['answer_start_index'] = random.choice(answer_start_list) if len(answer_start_list) > 0 else -4
            example_item['is_yes_no_question'] = False

        o_forward_items.append(example_item)

    # read_fitems = span_preprocess_tool.eitems_to_fitems(o_forward_items, bert_tokenizer, is_training, max_token_context,
    #                                                     max_token_query, doc_stride, False)

    return o_forward_items


def get_qa_item_with_upstream_sentence(d_list, sentence_level_results, is_training,
                                       tokenizer: BertTokenizer, max_context_length, max_query_length,
                                       doc_stride=128,
                                       debug_mode=False, top_k=5, filter_value=0.2):
    t_db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)

    if debug_mode:
        d_list = d_list[:100]
        id_set = set([item['_id'] for item in d_list])
        sentence_level_results = [item for item in sentence_level_results if item['qid'] in id_set]

    d_o_dict = list_dict_data_tool.list_to_dict(d_list, '_id')
    copied_d_o_dict = copy.deepcopy(d_o_dict)
    list_dict_data_tool.append_subfield_from_list_to_dict(sentence_level_results, copied_d_o_dict,
                                                          'qid', 'fid', check=True)

    cur_results_dict = select_top_k_and_to_results_dict(copied_d_o_dict, top_k=top_k,
                                                        score_field_name='prob',
                                                        filter_value=filter_value,
                                                        result_field='sp')

    forward_example_items = build_qa_forword_item(cur_results_dict, d_list, is_training, t_db_cursor)
    forward_example_items = format_convert(forward_example_items, is_training)
    fitems_dict, read_fitems_list = span_preprocess_tool.eitems_to_fitems(forward_example_items, tokenizer, is_training,
                                                                          max_context_length,
                                                                          max_query_length, doc_stride, False)

    return fitems_dict, read_fitems_list, cur_results_dict['sp']


def inspect_upstream_eval():
    dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, '_id')
    dev_eval_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/dev_s_level_bert_v1_results.jsonl")
    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    list_dict_data_tool.append_subfield_from_list_to_dict(dev_eval_results_list, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)

    # 0.5
    # cur_results_dict_v05 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
    #                                                         score_field_name='prob',
    #                                                         filter_value=0.5,
    #                                                         result_field='sp')

    cur_results_dict_v02 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
                                                            score_field_name='prob',
                                                            filter_value=0.2,
                                                            result_field='sp')

    # _, metrics_v5 = ext_hotpot_eval.eval(cur_results_dict_v05, dev_list, verbose=False)

    _, metrics_v2 = ext_hotpot_eval.eval(cur_results_dict_v02, dev_list, verbose=False)

    v02_sp_f1 = metrics_v2['sp_f1']
    v02_sp_recall = metrics_v2['sp_recall']
    v02_sp_prec = metrics_v2['sp_prec']

    v05_sp_f1 = metrics_v5['sp_f1']
    v05_sp_recall = metrics_v5['sp_recall']
    v05_sp_prec = metrics_v5['sp_prec']

    logging_item = {
        'label': 'ema',
        'v02': metrics_v2,
        # 'v05': metrics_v5,
    }

    print(logging_item)


def inspect_oracle_answer_text(append_head=True):
    dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    db_cursor = wiki_db_tool.get_cursor(config.WHOLE_PROCESS_FOR_RINDEX_DB)

    total, error = 0, 0

    for item in dev_list:
        qid = item['_id']
        query = item['question']
        answer = item['answer']
        o_contexts = item['context']
        supporting_facts = item['supporting_facts']

        # print(query)
        # print(answer)

        supporting_doc = set([fact[0] for fact in item['supporting_facts']])
        selected_fact = []
        sentid2sent_token_dict = dict()

        for doc in supporting_doc:
            # if doc in gt_doc:
            #     continue
            text_item = wiki_db_tool.get_item_by_key(db_cursor, key=doc)
            context = wiki_db_tool.get_first_paragraph_from_clean_text_item(text_item, flatten_to_paragraph=False,
                                                                            skip_first=True)
            for i, sentence_token in enumerate(context):
                # sentence_text = sentence_token
                if len(sentence_token) != 0:
                    selected_fact.append([doc, i])
                    sentid2sent_token_dict[(doc, i)] = sentence_token

        # shuffle doc ordering.
        supporting_doc = list(supporting_doc)
        random.shuffle(supporting_doc)
        # end

        shuffled_supporting_fact_list = []
        supporting_facts = sorted(supporting_facts, key=lambda x: (x[0], x[1]))
        for doc in supporting_doc:
            for fact in supporting_facts:
                if fact[0] == doc:
                    shuffled_supporting_fact_list.append(fact)

        assert len(shuffled_supporting_fact_list) == len(supporting_facts)

        # print(supporting_facts)
        # print(shuffled_supporting_fact_list)
        #
        # print("Sup Fact.")
        cur_doc = None
        context_token_list = []
        for doc, i in shuffled_supporting_fact_list:
            if (doc, i) not in sentid2sent_token_dict:
                print(f"Potential Error: {(doc, i)} not exists in DB.")
                continue
            # print((doc, i), sentid2sent_token_dict[(doc, i)])
            paragraph_token_list = sentid2sent_token_dict[(doc, i)]

            if cur_doc != doc and append_head and i != 0:
                context_token_list = context_token_list + doc.split(' ') + ['.'] + paragraph_token_list
            else:
                context_token_list += paragraph_token_list

        # print(context_token_list)
        context_matcher = ContextAnswerMatcher(context_token_list, uncase=True)
        context, answer_start_list = context_matcher.concate_and_return_answer_index(answer, match_type='left')

        if len(answer_start_list) > 1:
            error += 1

        if len(answer_start_list) == 0 and answer != 'yes' and answer != 'no':
            print("Error")
            print("Query:", query)
            print("Answer:", answer)
            print("Sp fact:", shuffled_supporting_fact_list)
            print("Context:", context)

            context_matcher = ContextAnswerMatcher(context_token_list, uncase=True)
            context, answer_start_list = context_matcher.concate_and_return_answer_index(answer)

        # print(sentid2sent_token_dict)

        # for title, number in supporting_facts:
        #     print(title, number)
        total += 1

    print(error, total)


if __name__ == '__main__':
    # inspect_upstream_eval()
    d_list = common.load_json(config.DEV_FULLWIKI_FILE)
    sentence_level_results = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_sentence_level/04-19-02:17:11_hotpot_v0_slevel_retri_(doc_top_k:2)/i(12000)|e(2)|v02_f1(0.7153646038858843)|v02_recall(0.7114645831323757)|v05_f1(0.7153646038858843)|v05_recall(0.7114645831323757)|seed(12)/dev_s_level_bert_v1_results.jsonl")
    bert_model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

    example_dict, examples_list = get_qa_item_with_upstream_sentence(d_list, sentence_level_results, is_training=True,
                                                                     tokenizer=tokenizer, max_context_length=320,
                                                                     max_query_length=64,
                                                                     filter_value=0.2, top_k=5,
                                                                     debug_mode=True)
    print(len(examples_list))
