import matplotlib.pyplot as plt

from batching import RiskIsolate
from utils import line_styles, project_list, color_list, num_of_learning_days_list
from learning import IncrementalLearningModel


def get_number(prj, num_of_learn_days):
    l = IncrementalLearningModel(prj, 'DecisionTree', num_of_learn_days, 1)
    y_proba, y_test = l.get_predicted_data()

    b = RiskIsolate(y_proba, y_test, test_all_after_size_4=False)
    b.get_num_of_exec_per_batch_size()
    return b.get_max_improvement()


def main():
    for idx, prj in enumerate(project_list):
        x = []
        y = []

        for num_of_learn_days in num_of_learning_days_list:
            x.append(str(num_of_learn_days))

            print(prj)
            print('Num of Learn Days: ', num_of_learn_days)
            max_improvement = get_number(prj, num_of_learn_days)
            y.append(max_improvement)
            print()

        plt.plot(x, y, line_styles[idx % len(line_styles)], color=color_list[idx % len(color_list)])

    plt.legend(project_list)

    # plt.xticks(num_of_learning_days_list)
    # plt.ylim(0, 24)
    plt.grid(axis='x')
    plt.xlabel('Number of Learn Days')
    plt.ylabel('Max Improvement %')
    plt.gcf().tight_layout(rect=(0, 0, 1, 1))
    plt.axhline(0, linestyle='dotted', color='black')
    plt.title('Window Batching')
    plt.show()


main()
