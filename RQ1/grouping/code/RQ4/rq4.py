"""
Dataset: https://travistorrent.testroots.org/page_access/

"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

RECURSIVE_BISECTION = True
DYNAMIC_BATCH_SIZE = True
BATCH_SIZE = 10

cols = [
    'gh_num_commits_in_push',
    'gh_team_size',
    'git_num_all_built_commits',
    'gh_num_issue_comments',
    'gh_num_commit_comments',
    'gh_num_pr_comments',
    'git_diff_src_churn',
    'git_diff_test_churn',
    'gh_diff_files_added',
    'gh_diff_files_deleted',
    'gh_diff_files_modified',
    'gh_diff_tests_added',
    'gh_diff_tests_deleted',
    'gh_diff_src_files',
    'gh_diff_doc_files',
    'gh_diff_other_files',
    'gh_num_commits_on_files_touched',
    'gh_sloc',
    'gh_asserts_cases_per_kloc',
    'gh_description_complexity',
    'tr_log_status'
]


def evaluate(classifier, x_test, y_test):
    y_pred = classifier.predict_proba(x_test)[:, 0]
    # y_pred = classifier.predict(x_test)

    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # print('Accuracy: ', accuracy)
    #
    # recall = metrics.recall_score(y_test, y_pred, average=None)
    # print('Recall:', recall)
    #
    # f_score = metrics.f1_score(y_test, y_pred, average=None)
    # print('F Score: ', f_score)
    #
    # precision = metrics.precision_score(y_test, y_pred, average=None)
    # print('Precision: ', precision)

    return y_pred


def learn(gh_project_name):
    dataset = pd.read_csv('../../data/{}.csv'.format(gh_project_name), usecols=cols)

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 20].values

    for i in range(0, len(y)):
        if y[i] != "ok":
            y[i] = "fail"

    # Handle Missing value
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imputer = imputer.fit(x)
    x = imputer.transform(x)

    # Splitting the dataset inti the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Decision Tree
    # t1 = time.time()
    classifier = DecisionTreeClassifier(criterion="entropy",
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        random_state=0,
                                        max_leaf_nodes=1500)
    classifier.fit(x_train, y_train)

    # t2 = time.time()
    # print('Time: ', t2 - t1)

    y_pred = evaluate(classifier, x_test, y_test)

    return y_pred, y_test


def batch_test(y_test, idx, batch_size):
    num_of_exec = 1

    if batch_size == 1:
        return num_of_exec

    if idx + batch_size >= len(y_test):
        upper_bound = len(y_test)
    else:
        upper_bound = idx + batch_size

    has_failed = False

    for i in range(idx, upper_bound):
        if y_test[i] != 'ok':
            has_failed = True
            break

    if has_failed:
        if RECURSIVE_BISECTION:
            num_of_exec += (
                    batch_test(y_test, idx, int(batch_size / 2)) +
                    batch_test(y_test, idx + int(batch_size / 2), int(batch_size / 2)))
        else:
            num_of_exec += batch_size

    return num_of_exec


def plot_risk_distribution(builds, project_name):
    failure_distribution = dict()

    for item in builds:
        k = int(item * 100)
        if k in failure_distribution:
            failure_distribution[k] += 1
        else:
            failure_distribution[k] = 1

    plt.bar(range(len(failure_distribution)), list(failure_distribution.values()), align='center')
    # plt.plot(range(len(failure_distribution)), list(failure_distribution.values()))
    plt.xticks(range(0, 110, 10))
    plt.xlabel('Predicted Risk (%)')
    plt.ylabel('Number of jobs')
    plt.title('Risk Distribution of Project {}'.format(project_name))
    plt.show()


def plot_recommended_batch_size_frequency(batch_size_list, project_name):
    x = list()
    y = list()

    for batch_size in range(1, max(batch_size_list)):
        x.append(batch_size)
        y.append(batch_size_list.count(batch_size))

    plt.bar(x, y, align='center')
    # plt.plot(range(len(failure_distribution)), list(failure_distribution.values()))
    plt.xticks(x)
    plt.xlabel('Recommended Batch Size')
    plt.ylabel('Frequency')
    plt.title('Recommended Batch Size Frequency of Project {}'.format(project_name))
    plt.show()


def calc_improvement(gh_project_name):
    y_pred, y_test = learn(gh_project_name)

    # Failure Rate
    # print(np.count_nonzero(y_test == 'fail') / len(y_test) * 100)

    # plot_risk_distribution(y_pred, gh_project_name)

    batch_size_list = []
    batch_min_max_diff_list = []

    risk = 0.1

    x = list()
    y = list()

    while risk <= 1:

        i = 0
        number_of_runs = 0

        while i < len(y_pred):

            if DYNAMIC_BATCH_SIZE:
                batch_size = 0
                cumulative_risk = 0

                while cumulative_risk <= risk and i + batch_size < len(y_pred):
                    cumulative_risk += y_pred[i + batch_size]
                    batch_size += 1

                batch_size_list.append(batch_size)

            else:
                batch_size = BATCH_SIZE

            number_of_runs += batch_test(y_test, i, batch_size)

            batch_min_max_diff = max(y_pred[i: i+batch_size]) - min(y_pred[i: i+batch_size])
            batch_min_max_diff_list.append(batch_min_max_diff)
            i += batch_size

        if DYNAMIC_BATCH_SIZE:
            # plot_recommended_batch_size_frequency(batch_size_list, gh_project_name)
            plt.hist(batch_min_max_diff_list)
            plt.show()

        improvement = (1 - number_of_runs / len(y_pred)) * 100
        y.append(improvement)
        # return improvement

        risk += 0.01
        x.append(int(risk * 100))

    max_value_index = y.index(max(y))
    x_max = x[max_value_index]
    y_max = y[max_value_index]

    plt.plot(x, y, label=gh_project_name)
    plt.yticks(range(10, 40, 5))
    plt.annotate('max (risk={}%)'.format(x_max), xy=(x_max, y_max),
                 xytext=(x_max, y_max + 2),
                 arrowprops=dict(facecolor='black', shrink=0),
                 )


def plot_risk_threshold():
    plt.xlabel('Risk Threshold (%)')
    plt.ylabel('Improvement (%) in number of executions')
    plt.legend()
    plt.show()


calc_improvement('jruby')
calc_improvement('rollbar-gem')
calc_improvement('rspec-rails')

plot_risk_threshold()


# todo: plot min max diff in each batch
# todo: use window to select commits for each batch
