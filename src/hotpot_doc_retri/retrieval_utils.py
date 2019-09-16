from utils import common
import json
from typing import List, Dict, Union
import random

"""
The results of the retrieval is a set of documents with each retrieved document have many attributes including
the method used to retrieve the document, the subgroup that should be ordered, and the group score.
"""


class RetrievedItem(common.JsonableObj):
    def __init__(self, item_id: str, retrieval_method: str) -> None:
        super().__init__()
        self.item_id = item_id  # The item identifier.
        self.scores_dict: Dict[str, float] = dict()
        # self.types: List = []             # The type of this item.
        self.method = retrieval_method  # The top level method that retrieve this item

    def __repr__(self) -> str:
        return common.json_dumps(self)

    def set_score(self, score: float, namespace: str):
        self.scores_dict[namespace] = score

    def __eq__(self, other):
        if self.item_id == other.item_id:
            return True
        return False


def get_sorted_top_k(d_l, top_k, key=None):
    if top_k >= len(d_l):
        return d_l[:top_k]
    value = d_l[top_k - 1] if key is None else key(d_l[top_k - 1])
    init_list = d_l[:top_k]     # First select top_k
    for i in range(top_k, len(d_l)):
        cur_val = d_l[i] if key is None else key(d_l[i])
        if cur_val == value:
            init_list.append(d_l[i])
        else:
            break

    return init_list


class RetrievedSet(common.JsonableObj):
    def __init__(self) -> None:
        super().__init__()
        self.retrieved_dict: Dict[str, RetrievedItem] = dict()

    def add_item(self, retri_item: RetrievedItem):
        if retri_item.item_id in self.retrieved_dict.keys():
            return None
        else:
            self.retrieved_dict[retri_item.item_id] = retri_item
            return retri_item

    def score_item(self, item_id: str, score: float, namespace: str):
        if item_id not in self.retrieved_dict.keys():
            return None
        else:
            self.retrieved_dict[item_id].set_score(score, namespace)

    def remove_item(self, item_id: str):
        if item_id not in self.retrieved_dict.keys():
            return None
        else:
            del self.retrieved_dict[item_id]

    def sort_and_filter(self, namespace: str, methods: Union[str, List[str]] = '_all_',
                        top_k: int = None, filter_value: float = None, strict_mode=False):
        # Sort all the item by the score in the give namespace, with specific methods.
        if not isinstance(methods, list):
            methods = [methods]

        # We only select those meet the method and contain the score namespace.
        all_item_meet_requirements: List[RetrievedItem] = []
        for key, item in self.retrieved_dict.items():
            if item.method in methods or methods == ['_all_'] and namespace in item.scores_dict:    # important.
                all_item_meet_requirements.append(item)

        sorted_item_list = sorted(all_item_meet_requirements, key=lambda x: (x.scores_dict[namespace], x.item_id),
                                  reverse=True) # sort, we use id as second key to eliminate randomness.
        if filter_value is not None:
            sorted_item_list = list(filter(lambda x: x.scores_dict[namespace] >= filter_value, sorted_item_list))
            # filter by value
        if strict_mode:
            sorted_item_list = sorted_item_list[:top_k] if top_k is not None else sorted_item_list
        else:
        # Important update 2019-04-06 to keep the data consistancy after sorting, we just keep all the value with same score.
            sorted_item_list = get_sorted_top_k(sorted_item_list, top_k, key=lambda x: x.scores_dict[namespace]) if top_k is not None else sorted_item_list
        # filter by count

        for aitem in all_item_meet_requirements:
            if aitem in sorted_item_list:
                pass
            else:
                self.remove_item(aitem.item_id)

    def to_id_list(self):
        return list(self.retrieved_dict.keys())

    def __len__(self):
        return len(self.retrieved_dict)

    def __repr__(self):
        return common.json_dumps(self)


common.register_class(RetrievedItem)
common.register_class(RetrievedSet)

if __name__ == '__main__':
    random.seed(6)
    item1 = RetrievedItem("abs-0", "key_word_match")
    item2 = RetrievedItem("abs-1", "key_word_match")
    item3 = RetrievedItem("abs-2", "key_word_match")
    item4 = RetrievedItem("abs-3", "key_word_match_dis")
    item5 = RetrievedItem("abs-4", "key_word_match_dis")
    item6 = RetrievedItem("abs-5", "key_word_match_2de")
    item7 = RetrievedItem("abs-6", "key_word_match_2de")

    rset = RetrievedSet()
    for item in [item1, item2, item3, item4, item5, item6, item7]:
        rset.add_item(item)

    # for item in [item1, item2, item3, item4, item5, item6, item7]:
    for item in [item1, item2, item3, item4]:
        rset.score_item(item.item_id, score=random.randint(0, 100), namespace='global_space')

    rset.sort_and_filter(namespace='global_space', top_k=1, filter_value=None)

    print(rset)

    # print(item in [item2])
    # print(item == item2)
    #
    # item_str = common.json_dumps(item)
    # print(item_str)
    # new_item_raw = json.loads(item_str)
    # print(new_item_raw)
    # new_item = common.json_loads(item_str)
    # # print(type(new_item))
    # print(new_item)
    # # print()
    # # print(registered_classes)
