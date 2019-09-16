from evaluation import fever_scorer
from fever_sampler.fever_sampler_utils import select_top_k_and_to_results_dict
from utils import common, list_dict_data_tool
import config
import copy


def p_eval():
    dev_list = common.load_jsonl(config.FEVER_DEV)
    # common.save_jsonl(cur_eval_results_list, f"fever_p_level_{tag}_results.jsonl")
    cur_eval_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_fever/fever_paragraph_level/04-22-15:05:45_fever_v0_plevel_retri_(ignore_non_verifiable:True)/i(5000)|e(0)|v02_ofever(0.8947894789478947)|v05_ofever(0.8555355535553555)|seed(12)/fever_p_level_dev_results.jsonl"
    )

    dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, 'id')
    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    copied_dev_d_list = copy.deepcopy(dev_list)
    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)

    cur_results_dict_th0_5 = select_top_k_and_to_results_dict(copied_dev_o_dict,
                                                              score_field_name='prob',
                                                              top_k=5, filter_value=0.005)

    list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
                                                                   cur_results_dict_th0_5,
                                                                   'id', 'predicted_docids')
    # mode = {'standard': False, 'check_doc_id_correct': True}
    strict_score, pr, rec, f1 = fever_scorer.fever_doc_only(copied_dev_d_list, dev_list,
                                                            max_evidence=5)

    score_05 = {
        'ss': strict_score,
        'pr': pr, 'rec': rec, 'f1': f1,
    }

    print(score_05)


def p_eval_term():
    dev_list = common.load_jsonl(config.FEVER_DEV)
    # common.save_jsonl(cur_eval_results_list, f"fever_p_level_{tag}_results.jsonl")
    # cur_eval_results_list = common.load_jsonl(
    #     config.PRO_ROOT / "data/p_fever/fever_paragraph_level/04-22-15:05:45_fever_v0_plevel_retri_(ignore_non_verifiable:True)/i(5000)|e(0)|v02_ofever(0.8947894789478947)|v05_ofever(0.8555355535553555)|seed(12)/fever_p_level_dev_results.jsonl"
    # )
    #
    # dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, 'id')
    # copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    # copied_dev_d_list = copy.deepcopy(dev_list)
    # list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
    #                                                       'qid', 'fid', check=True)
    #
    # cur_results_dict_th0_5 = select_top_k_and_to_results_dict(copied_dev_o_dict,
    #                                                           score_field_name='prob',
    #                                                           top_k=5, filter_value=0.005)
    #
    # list_dict_data_tool.append_item_from_dict_to_list_hotpot_style(copied_dev_d_list,
    #                                                                cur_results_dict_th0_5,
    #                                                                'id', 'predicted_docids')
    copied_dev_d_list = common.load_jsonl(
        config.PRO_ROOT / "results/doc_retri_results/fever_results/merged_doc_results/m_doc_dev.jsonl"
    )

    for item1, item2 in zip(dev_list, copied_dev_d_list):
        assert item1['id'] == item2['id']
        item2['evidence'] = item1['evidence']
    # mode = {'standard': False, 'check_doc_id_correct': True}
    strict_score, pr, rec, f1 = fever_scorer.fever_doc_only(copied_dev_d_list, dev_list,
                                                            max_evidence=None)

    score_05 = {
        'ss': strict_score,
        'pr': pr, 'rec': rec, 'f1': f1,
    }

    print(score_05)


if __name__ == '__main__':
    # p_eval()
    p_eval_term()