import config
from fever_models.nli.bert_fever_nli_v2 import get_nli_pair
from utils import common

if __name__ == '__main__':
    debug_mode = False
    dev_sent_filtering_prob = 0.2
    train_sent_filtering_prob = 0.2
    test_sent_filtering_prob = 0.2
    top_k_sent = 5

    # Data dataset and upstream sentence results.
    dev_sent_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_fever/fever_sentence_level/04-24-00-11-19_fever_v0_slevel_retri_(ignore_non_verifiable-True)/fever_s_level_dev_results.jsonl")
    train_sent_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_fever/fever_sentence_level/04-24-00-11-19_fever_v0_slevel_retri_(ignore_non_verifiable-True)/fever_s_level_train_results.jsonl")
    test_sent_results_list = common.load_jsonl(
        config.PRO_ROOT / "data/p_fever/fever_sentence_level/04-24-00-11-19_fever_v0_slevel_retri_(ignore_non_verifiable-True)/fever_s_level_test_results.jsonl")

    dev_fitems, dev_list = get_nli_pair('dev', is_training=False,
                                        sent_level_results_list=dev_sent_results_list, debug=debug_mode,
                                        sent_top_k=top_k_sent, sent_filter_value=dev_sent_filtering_prob)
    train_fitems, train_list = get_nli_pair('train', is_training=True,
                                            sent_level_results_list=train_sent_results_list, debug=debug_mode,
                                            sent_top_k=top_k_sent, sent_filter_value=train_sent_filtering_prob)
    test_fitems, test_list = get_nli_pair('test', is_training=False,
                                          sent_level_results_list=test_sent_results_list, debug=debug_mode,
                                          sent_top_k=5, sent_filter_value=test_sent_filtering_prob)

    print(dev_fitems[0])

    common.save_jsonl(dev_fitems, config.PRO_ROOT / "data/p_fever/intermediate_sent_data/dev_fitems.jsonl")
    common.save_jsonl(dev_list, config.PRO_ROOT / "data/p_fever/intermediate_sent_data/dev_list.jsonl")

    common.save_jsonl(train_fitems, config.PRO_ROOT / "data/p_fever/intermediate_sent_data/train_fitems.jsonl")
    common.save_jsonl(train_list, config.PRO_ROOT / "data/p_fever/intermediate_sent_data/train_list.jsonl")

    common.save_jsonl(test_fitems, config.PRO_ROOT / "data/p_fever/intermediate_sent_data/test_fitems.jsonl")
    common.save_jsonl(test_list, config.PRO_ROOT / "data/p_fever/intermediate_sent_data/test_list.jsonl")

