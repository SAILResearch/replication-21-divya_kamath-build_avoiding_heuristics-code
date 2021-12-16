from batching import RiskTopN
from utils import project_list
from learning import IncrementalLearningModel


import sys

sys.setrecursionlimit(15000)


def get_results(prj):
    l = IncrementalLearningModel(prj['name'], 'RF', 30, 1)
    y_proba, y_test = l.get_predicted_data()

    b = RiskTopN(y_proba, y_test, top_n=2, batch_size_max=8)
    b.get_num_of_exec_per_batch_size()

    return b


def main():
    for idx, prj in enumerate(project_list):
        print(prj['name'])

        results = get_results(prj)

        # RiskTopN num of execution
        print(results.lowest_num_of_exec)

        # RiskTopN optimum batch size
        print(results.best_batch_size)


main()
