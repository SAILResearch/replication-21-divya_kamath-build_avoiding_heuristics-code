import matplotlib.pyplot as plt

from batching import RiskTopN
from utils import line_styles, project_list, color_list
from learning import IncrementalLearningModel

import sys

sys.setrecursionlimit(15000)


def get_number(prj):
    l = IncrementalLearningModel(prj['name'], 'RF', 30, 1)
    y_proba, y_test = l.get_predicted_data()
    # l.print_scores()

    b = RiskTopN(y_proba, y_test, top_n=2)
    x, y = b.get_num_of_exec_per_batch_size()

    print(b.lowest_num_of_exec)

    return x, y


def main():
    for idx, prj in enumerate(project_list):
        # print(prj['name'])
        x, y = get_number(prj)
        # plt.plot(x, y, line_styles[idx % len(line_styles)], color=color_list[idx % len(color_list)])
        plt.plot(x, y, line_styles[idx % len(line_styles)])
        # print()

    plt.legend([prj['name'] for prj in project_list], bbox_to_anchor=(1, 1))
    plt.xticks(range(1, 20 + 1))
    plt.ylim(0, 70)
    plt.grid(axis='x')
    plt.xlabel('Batch Size')
    plt.ylabel('Improvement %')
    plt.title('RiskTopN')
    plt.savefig("result_risk_top_2.png", bbox_inches='tight')


main()
