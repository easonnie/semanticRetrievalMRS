from utils import common
import config
import random


def get_sample_data(size=-1):
    qa_gt_s = common.load_json(config.FEVER_DATA_ROOT / "qa_aug" / "squad_train_turker_groundtruth.json")
    # print(len(qa_gt_s))
    qa_aug_rnei = common.load_json(
        config.FEVER_DATA_ROOT / "qa_aug" / "squad_train_refutes_bytype_3x_claim_stoch_answspan_stoch.json")
    # print(len(qa_aug_rnei))
    random.shuffle(qa_aug_rnei)
    for item in qa_aug_rnei:
        sv = random.random()
        if sv > 0.5:
            item['label'] = "REFUTES"
        else:
            item['label'] = "NOT ENOUGH INFO"

    balanced_aug_data = qa_gt_s + qa_aug_rnei[:len(qa_gt_s) * 2]
    print("Total balanced size:", len(balanced_aug_data))
    random.shuffle(balanced_aug_data)
    if size != -1:
        return balanced_aug_data[:size]
    else:
        return balanced_aug_data


if __name__ == '__main__':
    aug_data = get_sample_data(3000)
    print(len(aug_data))
    r_count = 0
    e_count = 0
    s_count = 0
    for item in aug_data:
        if item['label'] == "NOT ENOUGH INFO":
            e_count += 1
            continue
        if item['label'] == "REFUTES":
            r_count += 1
            continue
        s_count += 1
    print(r_count, e_count, s_count)
