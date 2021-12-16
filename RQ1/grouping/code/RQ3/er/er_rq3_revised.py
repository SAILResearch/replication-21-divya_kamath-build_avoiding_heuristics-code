import operator

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import line_styles, er_project_list, get_predicted_data

TOP_N = 2


def get_number_of_runs_per_batch_size(gh_project_name):
    def batch_test(y_test, y_pred):
        num_of_exec = 1

        if len(y_test) == 1:
            return num_of_exec

        for _ in range(TOP_N):
            if not len(y_pred):
                break

            top_n_idx, top_n_value = max(enumerate(y_pred), key=operator.itemgetter(1))

            if top_n_value > 0:
                num_of_exec += 1
                y_test = np.delete(y_test, top_n_idx)
                y_pred = np.delete(y_pred, top_n_idx)

        no_failure = True

        for item in y_test:
            if item == 1:
                no_failure = False
                break

        if no_failure:
            return num_of_exec

        first_half = batch_test(y_test[:len(y_test) // 2], y_pred[:len(y_pred) // 2])
        second_half = batch_test(y_test[len(y_test) // 2:], y_pred[len(y_pred) // 2:])

        return num_of_exec + first_half + second_half

    dataset = pd.read_csv('../../data/{}.csv'.format(gh_project_name))

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 8].values

    for i in range(0, len(x)):
        for j in range(0, 7):
            if isinstance(x[i][j], str):
                x[i][j] = x[i][j].replace(',', '')

    y_pred, y_test = get_predicted_data(x, y, 'LogisticRegression')

    x = list()
    y = list()

    for batch_size in range(1, 21):
        print(batch_size)
        number_of_runs = 0

        for i in range(0, len(y_test), batch_size):
            if i + batch_size > len(y_test):
                selected_y_test = y_test[i:]
                selected_y_pred = y_pred[i:]
            else:
                selected_y_test = y_test[i:i + batch_size]
                selected_y_pred = y_pred[i:i + batch_size]

            number_of_runs += batch_test(selected_y_test, selected_y_pred)

        improvement = (1 - number_of_runs / len(y_test)) * 100
        print(improvement)

        x.append(batch_size)
        y.append(improvement)

    return x, y


for idx, prj in enumerate(er_project_list):
    print(prj)
    x, y = get_number_of_runs_per_batch_size(prj)
    plt.plot(x, y, line_styles[idx % len(line_styles)])
    print()

# plt.legend(project_list, bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
plt.legend(er_project_list)

plt.xticks(range(1, 21))
plt.ylim(0, 80)
plt.grid(axis='x')
plt.xlabel('Batch Size')
plt.ylabel('Improvement %')
plt.gcf().tight_layout(rect=(0, 0, 1, 1))
# plt.axhline(0, linestyle='dotted', color='black')
plt.show()