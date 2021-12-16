import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from batching import RiskTopN, RiskIsolate
from utils import line_styles, er_project_list, color_list
from learning import ERLearningModel, IncrementalLearningModel
import pandas as pd

import sys

sys.setrecursionlimit(15000)


def get_number(prj):
    # l = LearningModel(prj, 'DT')
    # l = ERLearningModel(prj, 'LR')
    # y_proba, y_test = l.get_predicted_data()

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
    y_proba = classifier.predict_proba(x_test)[:, 0]

    # print(metrics.classification_report(y_test, y_pred))

    print(y_proba)

    b = RiskTopN(y_proba, y_test, top_n=1)
    # b = WindowBatching(y_proba, y_test)
    x, y = b.get_num_of_exec_per_batch_size()
    # b.plot_batch_size_histogram()

    return x, y


def main():
    for idx, prj in enumerate(er_project_list):
        print(prj)
        x, y = get_number(prj)
        plt.plot(x, y, line_styles[idx % len(line_styles)], color=color_list[idx % len(color_list)])
        print()

    plt.legend(er_project_list)

    plt.xticks(range(1, 21))
    plt.ylim(0, 75)
    plt.grid(axis='x')
    plt.xlabel('Batch Size')
    plt.ylabel('Improvement %')
    plt.gcf().tight_layout(rect=(0, 0, 1, 1))
    plt.axhline(0, linestyle='dotted', color='black')
    plt.show()


main()
