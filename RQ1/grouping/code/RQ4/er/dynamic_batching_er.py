import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd

from learning import IncrementalLearningModel, ERLearningModel
from utils import line_styles, er_project_list, color_list


def get_number_of_runs_per_batch_size(gh_project_name):
    def batch_test(y_test):
        num_of_exec = 1

        if len(y_test) == 1:
            return num_of_exec

        no_failure = True

        for item in y_test:
            if item == 1:
                no_failure = False
                break

        if no_failure:
            return num_of_exec

        if len(y_test) <= 4:
            return len(y_test)

        first_half = batch_test(y_test[:len(y_test) // 2])
        second_half = batch_test(y_test[len(y_test) // 2:])

        return num_of_exec + first_half + second_half

    df = pd.read_csv('../../data/{}.csv'.format(prj))
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    for i in range(0, len(x)):
        for j in range(0, 7):
            if isinstance(x[i][j], str):
                x[i][j] = x[i][j].replace(',', '')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # min_max_scaler = MinMaxScaler(feature_range=(0, 10000))
    # min_max_scaler.fit(x_train)
    # x_train = min_max_scaler.transform(x_train)
    #
    classifier = LogisticRegression()

    # classifier = DecisionTreeClassifier(criterion="entropy",
    #                                     min_samples_split=2,
    #                                     min_samples_leaf=1,
    #                                     random_state=0,
    #                                     max_leaf_nodes=100,
    #                                     splitter='best')

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    y_pred = classifier.predict_proba(x_test)[:, 0]

    x = list()
    y = list()

    for risk_threshold in range(1, 30):
        risk_threshold /= 10

        number_of_runs = 0
        i = 0

        print(risk_threshold)

        while i < len(y_test):
            batch_size = 0
            cumulative_risk = 0

            while cumulative_risk <= risk_threshold and i + batch_size < len(y_pred):
                cumulative_risk += y_pred[i + batch_size]
                batch_size += 1

            if i + batch_size > len(y_test):
                selected_y_test = y_test[i:]
            else:
                selected_y_test = y_test[i:i + batch_size]

            number_of_runs += batch_test(selected_y_test)

            i += batch_size

        improvement = (1 - number_of_runs / len(y_test)) * 100
        print(improvement)

        x.append(risk_threshold * 100)
        y.append(improvement)

    return x, y


for idx, prj in enumerate(er_project_list):
    print(prj)
    x, y = get_number_of_runs_per_batch_size(prj)
    plt.plot(x, y, line_styles[idx % len(line_styles)], color=color_list[idx % len(color_list)])
    print()

# plt.legend(project_list, bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
plt.legend(er_project_list)

plt.xticks(range(0, 300, 20))
# plt.ylim(0, 4)
plt.grid(axis='x')
plt.xlabel('Cumulative Risk Threshold %')
plt.ylabel('Improvement %')
plt.gcf().tight_layout(rect=(0, 0, 1, 1))
# plt.axhline(0, linestyle='dotted', color='black')
plt.show()
