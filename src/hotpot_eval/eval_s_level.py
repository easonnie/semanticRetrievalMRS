import copy

from evaluation import ext_hotpot_eval
from hotpot_fact_selection_sampler.sampler_utils import select_top_k_and_to_results_dict
from utils import list_dict_data_tool, common
import config


def eval_hotpot_s():
    cur_dev_eval_results_list_out = common.load_jsonl(
        config.PRO_ROOT / "data/p_hotpotqa/hotpot_p_level_effects/hotpot_s_level_dev_results_top_k_doc_100.jsonl"
    )
    dev_list = common.load_json(config.DEV_FULLWIKI_FILE)
    dev_o_dict = list_dict_data_tool.list_to_dict(dev_list, '_id')
    copied_dev_o_dict = copy.deepcopy(dev_o_dict)
    list_dict_data_tool.append_subfield_from_list_to_dict(cur_dev_eval_results_list_out, copied_dev_o_dict,
                                                          'qid', 'fid', check=True)
    # 0.5
    cur_results_dict_v05 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
                                                            score_field_name='prob',
                                                            filter_value=0.5,
                                                            result_field='sp')

    # cur_results_dict_v02 = select_top_k_and_to_results_dict(copied_dev_o_dict, top_k=5,
    #                                                         score_field_name='prob',
    #                                                         filter_value=0.2,
    #                                                         result_field='sp')

    _, metrics_v5 = ext_hotpot_eval.eval(cur_results_dict_v05, dev_list, verbose=False)

    # _, metrics_v2 = ext_hotpot_eval.eval(cur_results_dict_v02, dev_list, verbose=False)

    logging_item = {
        # 'v02': metrics_v2,
        'v05': metrics_v5,
    }

    print(logging_item)
    f1 = metrics_v5['sp_f1']
    em = metrics_v5['sp_em']
    pr = metrics_v5['sp_prec']
    rec = metrics_v5['sp_recall']

    print(em, pr, rec, f1)


if __name__ == '__main__':
    eval_hotpot_s()
    # 1 : {'score': {'em': 0.2676569885212694, 'f1': 0.35761304406758887, 'prec': 0.3706357972499972, 'recall': 0.37488595662236285, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.0, 'sp_f1': 0.5315983993498946, 'sp_prec': 0.8127954085077648, 'sp_recall': 0.40643660975531576, 'joint_em': 0.0, 'joint_f1': 0.2154158082861405, 'joint_prec': 0.33406534330661514, 'joint_recall': 0.17317959858190005}}
    # 2 :: 0.3986495611073599 0.7560072023407646 0.7114645831323757 0.7153646038858843
    # 2 : {'score': {'em': 0.46563133018230923, 'f1': 0.5873511080216877, 'prec': 0.6050648364571004, 'recall': 0.6094402870544784, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.3986495611073599, 'sp_f1': 0.7153646038858843, 'sp_prec': 0.7560072023407646, 'sp_recall': 0.7114645831323757, 'joint_em': 0.26603646185010127, 'joint_f1': 0.49093231337134974, 'joint_prec': 0.5253774659313448, 'joint_recall': 0.5079485256835794}}
    # 3 : {'score': {'em': 0.4363268062120189, 'f1': 0.5551132378123007, 'prec': 0.5710563823891405, 'recall': 0.5799864187107506, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.10533423362592843, 'sp_f1': 0.6391557710801651, 'sp_prec': 0.5954130092279911, 'sp_recall': 0.7413022089321899, 'joint_em': 0.0700877785280216, 'joint_f1': 0.4060834107701022, 'joint_prec': 0.38521215043814344, 'joint_recall': 0.49204666190367213}}
    # 4 :: 0.06549628629304524 0.5060139545352211 0.7445873123050727 0.5798462682189831
    # 4 : {'score': {'em': 0.4041863605671843, 'f1': 0.5199367436963342, 'prec': 0.5359877779147572, 'recall': 0.5431043074065478, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.06549628629304524, 'sp_f1': 0.5798462682189831, 'sp_prec': 0.5060139545352211, 'sp_recall': 0.7445873123050727, 'joint_em': 0.04226873733963538, 'joint_f1': 0.3456981146625427, 'joint_prec': 0.3084495115934355, 'joint_recall': 0.45975383302534856}}
    # 5 : {'score': {'em': 0.39027683997299123, 'f1': 0.5035603196116694, 'prec': 0.5185525448024294, 'recall': 0.5251810545198358, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.04686022957461175, 'sp_f1': 0.5375693074274968, 'sp_prec': 0.4527301372946293, 'sp_recall': 0.7359856596250945, 'joint_em': 0.029709655638082377, 'joint_f1': 0.3114380668885575, 'joint_prec': 0.2691991870995185, 'joint_recall': 0.438901956382869}}
    # 6 :: 0.03700202565833896 0.4222417285617933 0.7183891193209239 0.5103626893025789
    # 6 : {'score': {'em': 0.37353139770425386, 'f1': 0.4840686120109792, 'prec': 0.498051640826311, 'recall': 0.5073695762362997, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.03700202565833896, 'sp_f1': 0.5103626893025789, 'sp_prec': 0.4222417285617933, 'sp_recall': 0.7183891193209239, 'joint_em': 0.02363268062120189, 'joint_f1': 0.28661976657433663, 'joint_prec': 0.24372833824221796, 'joint_recall': 0.41625486594096656}}
    # 7 : {'score': {'em': 0.3690749493585415, 'f1': 0.47703072416284925, 'prec': 0.4901630455133688, 'recall': 0.4996082067632491, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.031330182309250505, 'sp_f1': 0.4914993963947209, 'sp_prec': 0.4021809588116251, 'sp_recall': 0.7034574450982292, 'joint_em': 0.020526671168129642, 'joint_f1': 0.27485780404853444, 'joint_prec': 0.23098594993083463, 'joint_recall': 0.406038306485717}}
    # 8 :: 0.0287643484132343 0.38766824217872 0.6927606829362395 0.4782798857680527
    # 8 : {'score': {'em': 0.36286293045239704, 'f1': 0.46992446291345835, 'prec': 0.48303403318484933, 'recall': 0.4935198812718969, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.0287643484132343, 'sp_f1': 0.4782798857680527, 'sp_prec': 0.38766824217872, 'sp_recall': 0.6927606829362395, 'joint_em': 0.018771100607697502, 'joint_f1': 0.26584984709540677, 'joint_prec': 0.22127202438542365, 'joint_recall': 0.3985134968917774}}
    # 9 : {'score': {'em': 0.35705604321404455, 'f1': 0.46308064826425166, 'prec': 0.47649971772069494, 'recall': 0.48457334491601906, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.025658338960162053, 'sp_f1': 0.4680920909685097, 'sp_prec': 0.37666666666667903, 'sp_recall': 0.6845869907719996, 'joint_em': 0.01688048615800135, 'joint_f1': 0.25767193037265773, 'joint_prec': 0.21321488253534018, 'joint_recall': 0.38874021375878137}}
    # 10 ::0.023092505064145848 0.3674116587891192 0.6767811324394699 0.45935325084547
    # 10 : {'score': {'em': 0.3507089804186361, 'f1': 0.4574134733047023, 'prec': 0.4684838965394071, 'recall': 0.4804095175212662, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.023092505064145848, 'sp_f1': 0.45935325084547, 'sp_prec': 0.3674116587891192, 'sp_recall': 0.6767811324394699, 'joint_em': 0.0149898717083052, 'joint_f1': 0.25049993339486915, 'joint_prec': 0.2055609052268519, 'joint_recall': 0.38212293708242334}}
    # 12 :: 0.018906144496961513 0.35372496061221276 0.6660435355776327 0.4467203626892874
    # 11 : {'score': {'em': 0.349628629304524, 'f1': 0.45563231593734604, 'prec': 0.46814734220516846, 'recall': 0.4759113751804497, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.020931802835921675, 'sp_f1': 0.4521303319682635, 'sp_prec': 0.35973891514743606, 'sp_recall': 0.6703752290923104, 'joint_em': 0.013909520594193113, 'joint_f1': 0.24649518590211567, 'joint_prec': 0.2018308394073517, 'joint_recall': 0.3760424756802191}}
    # 12 : {'score': {'em': 0.3408507765023633, 'f1': 0.44742559582964236, 'prec': 0.46018488248027645, 'recall': 0.46928894391191306, 'doc_em': 0.0, 'doc_f1': 0.0, 'doc_prec': 0.0, 'doc_recall': 0.0, 'sp_em': 0.018906144496961513, 'sp_f1': 0.4467203626892874, 'sp_prec': 0.35372496061221276, 'sp_recall': 0.6660435355776327, 'joint_em': 0.012153950033760972, 'joint_f1': 0.23991281623319693, 'joint_prec': 0.19567155716943618, 'joint_recall': 0.3698419874872381}}


    # 100 :: 0.005806887238352465 0.29570110285845064 0.6071078100382615 0.38864329833268607
