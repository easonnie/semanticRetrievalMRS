from evaluation import ext_hotpot_eval
from utils import common
import config
from hotpot_doc_retri.retrieval_utils import RetrievedItem, RetrievedSet


def full_wiki_baseline_upperbound():
    dev_fullwiki = common.load_json(config.DEV_FULLWIKI_FILE)
    # dev_fullwiki = common.load_json(config.DEV_DISTRACTOR_FILE)
    upperbound_pred_file = dict()

    upperbound_pred_file['sp'] = dict()
    upperbound_pred_file['sp_doc'] = dict()
    upperbound_pred_file['p_answer'] = dict()

    # print(dev_fullwiki)
    for item in dev_fullwiki:
        qid = item['_id']
        answer = item['answer']
        contexts = item['context']
        supporting_facts = item['supporting_facts']
        # supporting_doc = set([fact[0] for fact in item['supporting_facts']])

        # retrieved_doc_dict = set([context[0] for context in contexts])
        retrieved_doc_dict = dict()

        for doc_title, context_sents in contexts:
            if doc_title not in retrieved_doc_dict:
                retrieved_doc_dict[doc_title] = dict()

            for i, sent in enumerate(context_sents):
                retrieved_doc_dict[doc_title][i] = sent

        upperbound_pred_doc = []
        upperbound_pred_sp = []

        found_answer = False
        for sp_doc, sp_fact_line_num in supporting_facts:
            if sp_doc in retrieved_doc_dict and sp_fact_line_num in retrieved_doc_dict[sp_doc]:
                upperbound_pred_doc.append(sp_doc)
                upperbound_pred_sp.append([sp_doc, sp_fact_line_num])
                if answer in retrieved_doc_dict[sp_doc][sp_fact_line_num]:
                    found_answer = True

        p_answer = answer if found_answer else ""

        upperbound_pred_file['sp'][qid] = upperbound_pred_sp
        upperbound_pred_file['sp_doc'][qid] = upperbound_pred_doc

        upperbound_pred_file['p_answer'][qid] = p_answer

        if all([gt_fact in upperbound_pred_sp for gt_fact in supporting_facts]):
            # If we find all the evidence, to add additional yes/no answer.
            upperbound_pred_file['p_answer'][qid] = answer

    ext_hotpot_eval.eval(upperbound_pred_file, dev_fullwiki)


def doc_retrie_v5_upperbound():
    dev_fullwiki = common.load_json(config.DEV_FULLWIKI_FILE)
    pred_dev = common.load_json(
        config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_v5_file.json")

    pred_v5_sp_doc = pred_dev['sp_doc']
    # dev_fullwiki = common.load_json(config.DEV_DISTRACTOR_FILE)
    upperbound_pred_file = dict()

    upperbound_pred_file['sp'] = dict()
    upperbound_pred_file['sp_doc'] = dict()
    upperbound_pred_file['p_answer'] = dict()

    # print(dev_fullwiki)
    for item in dev_fullwiki:
        qid = item['_id']
        answer = item['answer']
        contexts = item['context']
        supporting_facts = item['supporting_facts']

        v5_retrieved_doc = pred_v5_sp_doc[qid]
        # print(v5_retrieved_doc)
        # supporting_doc = set([fact[0] for fact in item['supporting_facts']])

        # retrieved_doc_dict = set([context[0] for context in contexts])
        retrieved_doc_dict = dict()

        for doc_title, context_sents in contexts:
            if doc_title not in retrieved_doc_dict:
                retrieved_doc_dict[doc_title] = dict()

            for i, sent in enumerate(context_sents):
                retrieved_doc_dict[doc_title][i] = sent

        upperbound_pred_doc = []
        upperbound_pred_sp = []

        found_answer = False
        for sp_doc, sp_fact_line_num in supporting_facts:
            if sp_doc in retrieved_doc_dict and sp_fact_line_num in retrieved_doc_dict[sp_doc]:
                upperbound_pred_doc.append(sp_doc)
                upperbound_pred_sp.append([sp_doc, sp_fact_line_num])
                if answer in retrieved_doc_dict[sp_doc][sp_fact_line_num]:
                    found_answer = True

            elif sp_doc in v5_retrieved_doc:
                upperbound_pred_doc.append(sp_doc)
                upperbound_pred_sp.append([sp_doc, sp_fact_line_num])
                # if answer in retrieved_doc_dict[sp_doc][sp_fact_line_num]:
                #     found_answer = True

        p_answer = answer if found_answer else ""

        upperbound_pred_file['sp'][qid] = upperbound_pred_sp
        upperbound_pred_file['sp_doc'][qid] = upperbound_pred_doc

        upperbound_pred_file['p_answer'][qid] = p_answer

        if all([gt_fact in upperbound_pred_sp for gt_fact in supporting_facts]):
            # If we find all the evidence, to add additional yes/no answer.
            upperbound_pred_file['p_answer'][qid] = answer

    ext_hotpot_eval.eval(upperbound_pred_file, dev_fullwiki)


def doc_retrie_v5_reimpl_tf_idf_upperbound():
    top_k = 10
    dev_fullwiki = common.load_json(config.DEV_FULLWIKI_FILE)

    pred_dev = common.load_json(
        # config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_with_hyperlinked_v5_file.json")
        # config.RESULT_PATH / "doc_retri_results/doc_raw_matching_file.json")
        config.RESULT_PATH / "doc_retri_results/doc_retrieval_debug_v6/doc_raw_matching_with_disamb_withiout_hyperlinked_v6_file_debug_4.json")
        # config.RESULT_PATH / "doc_retri_results/doc_raw_matching_with_disamb_withiout_hyperlinked_v5_file.json")

    tf_idf_dev_results = common.load_jsonl(
        config.RESULT_PATH / "doc_retri_results/term_based_methods_results/hotpot_tf_idf_dev.jsonl")

    tf_idf_scored_dict = dict()
    for item in tf_idf_dev_results:
        sorted_scored_list = sorted(item['doc_list'], key=lambda x: x[0], reverse=True)
        pred_list = [docid for _, docid in sorted_scored_list[:top_k]]
        qid = item['qid']
        tf_idf_scored_dict[qid] = pred_list

    pred_v5_sp_doc = pred_dev['sp_doc']
    # dev_fullwiki = common.load_json(config.DEV_DISTRACTOR_FILE)
    upperbound_pred_file = dict()

    upperbound_pred_file['sp'] = dict()
    upperbound_pred_file['sp_doc'] = dict()
    upperbound_pred_file['p_answer'] = dict()

    # print(dev_fullwiki

    for item in dev_fullwiki:
        qid = item['_id']
        answer = item['answer']
        contexts = item['context']
        supporting_facts = item['supporting_facts']

        tf_idf_docs = tf_idf_scored_dict[qid]

        v5_retrieved_doc = pred_v5_sp_doc[qid]
        # print(v5_retrieved_doc)
        supporting_doc = set([fact[0] for fact in item['supporting_facts']])

        # retrieved_doc_dict = set([context[0] for context in contexts])
        retrieved_doc_dict = dict()

        for doc_title, context_sents in contexts:
            if doc_title not in retrieved_doc_dict:
                retrieved_doc_dict[doc_title] = dict()

            for i, sent in enumerate(context_sents):
                retrieved_doc_dict[doc_title][i] = sent

        upperbound_pred_doc = []
        upperbound_pred_sp = []

        found_answer = False
        for sp_doc in tf_idf_docs:
            if sp_doc in supporting_doc:
                upperbound_pred_doc.append(sp_doc)
                for gt_sp_doc, sp_fact_line_num in supporting_facts:
                    if gt_sp_doc == sp_doc:
                        upperbound_pred_sp.append([sp_doc, sp_fact_line_num])
                    # if answer in retrieved_doc_dict[sp_doc][sp_fact_line_num]:
                        found_answer = True

        for sp_doc in v5_retrieved_doc:
            if sp_doc not in upperbound_pred_doc:
                if sp_doc in supporting_doc:
                    upperbound_pred_doc.append(sp_doc)
                    for gt_sp_doc, sp_fact_line_num in supporting_facts:
                        if gt_sp_doc == sp_doc:
                            upperbound_pred_sp.append([sp_doc, sp_fact_line_num])
                        # if answer in retrieved_doc_dict[sp_doc][sp_fact_line_num]:
                            found_answer = True


                # upperbound_pred_sp.append([sp_doc, sp_fact_line_num])
                # if answer in retrieved_doc_dict[sp_doc][sp_fact_line_num]:
                #     found_answer = True

        p_answer = answer if found_answer else ""

        upperbound_pred_file['sp'][qid] = upperbound_pred_sp
        upperbound_pred_file['sp_doc'][qid] = upperbound_pred_doc

        upperbound_pred_file['p_answer'][qid] = p_answer

        if all([gt_fact in upperbound_pred_sp for gt_fact in supporting_facts]):
            # If we find all the evidence, to add additional yes/no answer.
            upperbound_pred_file['p_answer'][qid] = answer

    ext_hotpot_eval.eval(upperbound_pred_file, dev_fullwiki)


def full_wiki_baseline_score():
    dev_fullwiki = common.load_json(config.DEV_FULLWIKI_FILE)
    # dev_fullwiki = common.load_json(config.DEV_DISTRACTOR_FILE)
    upperbound_pred_file = dict()

    # upperbound_pred_file['sp'] = dict()
    upperbound_pred_file['sp_doc'] = dict()
    # upperbound_pred_file['p_answer'] = dict()

    # print(dev_fullwiki)
    for item in dev_fullwiki:
        qid = item['_id']
        answer = item['answer']
        contexts = item['context']
        supporting_facts = item['supporting_facts']
        # supporting_doc = set([fact[0] for fact in item['supporting_facts']])

        # retrieved_doc_dict = set([context[0] for context in contexts])
        retrieved_doc_dict = dict()

        for doc_title, context_sents in contexts:
            if doc_title not in retrieved_doc_dict:
                retrieved_doc_dict[doc_title] = dict()

            for i, sent in enumerate(context_sents):
                retrieved_doc_dict[doc_title][i] = sent

        upperbound_pred_doc = []
        # upperbound_pred_sp = []

        # found_answer = False
        # for sp_doc, sp_fact_line_num in supporting_facts:
        #     if sp_doc in retrieved_doc_dict and sp_fact_line_num in retrieved_doc_dict[sp_doc]:
        #         upperbound_pred_doc.append(sp_doc)
        #         upperbound_pred_sp.append([sp_doc, sp_fact_line_num])
        #         if answer in retrieved_doc_dict[sp_doc][sp_fact_line_num]:
        #             found_answer = True
        #
        # p_answer = answer if found_answer else ""

        # upperbound_pred_file['sp'][qid] = upperbound_pred_sp
        upperbound_pred_file['sp_doc'][qid] = list(retrieved_doc_dict.keys())

        # upperbound_pred_file['p_answer'][qid] = p_answer

        # if all([gt_fact in upperbound_pred_sp for gt_fact in supporting_facts]):
            # If we find all the evidence, to add additional yes/no answer.
            # upperbound_pred_file['p_answer'][qid] = answer

    ext_hotpot_eval.eval(upperbound_pred_file, dev_fullwiki)


def append_gt_downstream_to_get_upperbound_from_doc_retri(results_dict, d_list):

    upper_bound_results_dict = {'sp': dict(), 'sp_doc': dict(), 'p_answer': dict()}

    for item in d_list:
        qid = item['_id']
        answer = item['answer']
        # contexts = item['context']
        supporting_facts = item['supporting_facts']
        supporting_doc = set([fact[0] for fact in item['supporting_facts']])

        found_answer = False

        retrieved_doc = results_dict['sp_doc'][qid]

        upperbound_pred_doc = []
        upperbound_pred_sp = []

        for sp_doc in retrieved_doc:
            if sp_doc in supporting_doc:
                upperbound_pred_doc.append(sp_doc)
                for gt_sp_doc, sp_fact_line_num in supporting_facts:
                    if gt_sp_doc == sp_doc:
                        upperbound_pred_sp.append([sp_doc, sp_fact_line_num])
                        found_answer = True

        p_answer = answer if found_answer else ""

        upper_bound_results_dict['sp'][qid] = upperbound_pred_sp
        upper_bound_results_dict['sp_doc'][qid] = upperbound_pred_doc

        upper_bound_results_dict['p_answer'][qid] = p_answer

        if all([gt_fact in upperbound_pred_sp for gt_fact in supporting_facts]):
            # If we find all the evidence, to add additional yes/no answer.
            upper_bound_results_dict['p_answer'][qid] = answer

    return upper_bound_results_dict
    # ext_hotpot_eval.eval(upperbound_pred_file, dev_fullwiki)


if __name__ == '__main__':
    # full_wiki_baseline_upperbound()
    # doc_retrie_v5_upperbound()
    # full_wiki_baseline_score()
    doc_retrie_v5_reimpl_tf_idf_upperbound()
    #


        # for context in contexts:
        #     doc_title = context[0]
        #     # print(doc_title)
        #     # print(len(context[1]))
        #
        # for supporting_fact in item['supporting_facts']:
        #     print(supporting_fact)

    # print(supporting_doc)

# 8769750168804862
# 8677920324105334