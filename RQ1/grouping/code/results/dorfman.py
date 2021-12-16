import matplotlib.pyplot as plt

from batching import BatchBisect
from utils import line_styles, project_list, color_list
from learning import IncrementalLearningModel

import sys


def get_number(prj):
    l = IncrementalLearningModel(prj['name'], 'RF', 30, 1)
    y_proba, y_test = l.get_predicted_data()

    b = BatchBisect(y_test, no_bisect=True)
    x, y = b.get_num_of_exec_per_batch_size()

    print(b.max_improvement)

    return x, y


def main():
    for idx, prj in enumerate(project_list):
        # print(prj['name'])
        x, y = get_number(prj)
        plt.plot(x, y, line_styles[idx % len(line_styles)])
        # print()
    #
    plt.legend([prj['name'] for prj in project_list], bbox_to_anchor=(1, 1))
    plt.xticks(range(1, 20 + 1))
    plt.ylim(0, 70)
    plt.grid(axis='x')
    plt.xlabel('Batch Size')
    plt.ylabel('Improvement %')
    # plt.title('BatchBisect and Batch4')
    plt.axvline(4, linestyle='-', color='#911010', linewidth=2)
    # plt.show()
    plt.savefig("result_dorfman.png", bbox_inches='tight')
#

main()
