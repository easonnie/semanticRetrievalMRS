import json
import config
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def flatten_counter_info(counter_dict, range_min=None, range_max=None):
    max_key = max(counter_dict.keys()) if range_max is None else range_max
    min_key = min(counter_dict.keys()) if range_min is None else range_min

    print(f"Range from {min_key} to {max_key}.")

    x = list(range(min_key, max_key + 1))
    y = []
    for i in x:
        if i in counter_dict:
            y.append(counter_dict[i])
        else:
            y.append(0)

    return x, y


if __name__ == '__main__':
    # with open(config.PDATA_ROOT / "stats_info/sent_per_para_counter.json") as in_f:
    # with open(config.PDATA_ROOT / "stats_info/total_para_counter.json") as in_f:
    # with open(config.PDATA_ROOT / "stats_info/total_sent_counter.json") as in_f:
    # with open(config.PDATA_ROOT / "stats_info/t_a_counter.json") as in_f:
    with open(config.PDATA_ROOT / "stats_info/t_p_counter.json") as in_f:
        # with open(config.PDATA_ROOT / "stats_info/t_s_counter.json") as in_f:
        pain_info = json.load(in_f)
        stats_info = {}
        for k, v in pain_info.items():
            stats_info[int(k)] = v

        print(sorted(stats_info, reverse=True))

        x, y = flatten_counter_info(stats_info, range_min=0, range_max=80)

        print(x)
        print(y)
        print(max(y))

        total_num = 0
        total_count = 0
        for count, num in zip(y, x):
            total_num += count * num
            total_count += count

        print("Ave:", total_num / total_count)

        # exit(-1)

        plt.figure(dpi=200)
        sns.barplot(x, y, color="lightskyblue")
        sns.set(rc={"font.size": 0.01, "axes.labelsize": 5})
        # # Decoration
        # plt.title('Number of Sentences Per Paragraph')
        # plt.title('Number of Sentences Per Article')
        # plt.title('Number of Paragraph Per Article')
        # plt.title('Number of Token Per Article')
        # plt.title('Number of Token Per Paragraph')
        plt.title('Number of Token Per Sentence')
        # plt.legend()
        plt.show()
        # plt.savefig("fig.pdf")
