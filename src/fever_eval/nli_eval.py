import json
import config
from evaluation import fever_scorer
import copy

from fever_sampler.nli_new_sampler import get_nli_pair
from utils import common, list_dict_data_tool
import numpy as np


def eval_nli():
    dev_list = common.load_jsonl(config.FEVER_DEV)
    # prediction_file = config.PRO_ROOT / "data/p_fever/fever_nli/04-25-22:02:53_fever_v2_nli_th0.2/ema_i(20000)|e(3)|ss(0.7002700270027002)|ac(0.746024602460246)|pr(0.6141389138913633)|rec(0.8627362736273627)|f1(0.7175148212089147)|seed(12)/nli_dev_cp_results_th0.2.jsonl"
    # prediction_file = config.PRO_ROOT / "saved_models/04-15-00:15:59_fever_v1_nli/i(18000)|e(2)|ss(0.6154615461546155)|ac(0.6701170117011701)|pr(0.26657540754071885)|rec(0.8852385238523852)|f1(0.40975857963668794)|seed(12)_dev_nli_results.json"
    prediction_file = config.PRO_ROOT / "data/p_fever/non_sent_level/ema_i(32000)|e(4)|ss(0.5592059205920592)|ac(0.6104110411041104)|pr(0.2638851385138135)|rec(0.8928142814281428)|f1(0.4073667130110584)|seed(12)_dev_nli_results.json"
    pred_list = common.load_jsonl(prediction_file)
    mode = {'standard': True}
    strict_score, acc_score, pr, rec, f1 = fever_scorer.fever_score(pred_list, dev_list,
                                                                    mode=mode, max_evidence=5)
    logging_item = {
        'ss': strict_score, 'ac': acc_score,
        'pr': pr, 'rec': rec, 'f1': f1,
    }

    print(logging_item)
    fever_scorer.fever_confusion_matrix(pred_list, dev_list)


if __name__ == '__main__':
    eval_nli()
