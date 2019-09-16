from evaluation import ext_hotpot_eval
from hotpot_content_selection.bert_p_level_v1 import select_top_k_and_to_results_dict
from utils import list_dict_data_tool
from utils import common
import config
import copy


def eval_p_level():
    cur_eval_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpotqa_paragraph_level/04-10-17:44:54_hotpot_v0_cs/i(40000)|e(4)|t5_doc_recall(0.8793382849426064)|t5_sp_recall(0.879496479212887)|t10_doc_recall(0.888656313301823)|t5_sp_recall(0.8888325134240054)|seed(12)/dev_p_level_bert_v1_results.jsonl"
    )

    dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, '_id')

    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    list_dict_data_tool.append_subfield_from_list_to_dict(cur_eval_results_list, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)
    # Top_5
    cur_results_dict_top5 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5)


    _, metrics_top5 = ext_hotpot_eval.eval(cur_results_dict_top5, dev_list, verbose=False)

    print(metrics_top5)

if __name__ == '__main__':
    eval_p_level()