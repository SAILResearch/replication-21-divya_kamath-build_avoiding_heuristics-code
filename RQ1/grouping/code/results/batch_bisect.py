from statistics import mean

from batching import BatchBisect
from utils import project_list
from learning import IncrementalLearningModel


def get_results(prj):
    l = IncrementalLearningModel(prj['name'], 'RF', 30, 1)
    y_proba, y_test = l.get_predicted_data()

    b = BatchBisect(y_test, stop_at_4=False, batch_size_max=20)
    b.get_num_of_exec_per_batch_size()

    return b


def main():
    l = list()
    for idx, prj in enumerate(project_list):
        # print(prj['name'])

        results = get_results(prj)

        # BatchBisect num of execution
        # print(results.lowest_num_of_exec)

        # BatchBisect optimum batch size
        print(results.best_batch_size)
        l.append(results.best_batch_size)

    print()
    print(mean(l))

main()
