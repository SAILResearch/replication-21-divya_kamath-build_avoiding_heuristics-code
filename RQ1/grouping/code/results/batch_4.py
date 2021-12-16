from batching import BatchBisect
from utils import project_list
from learning import IncrementalLearningModel


def get_results(prj):
    l = IncrementalLearningModel(prj['name'], 'RF', 30, 1)
    y_proba, y_test = l.get_predicted_data()

    b = BatchBisect(y_test, stop_at_4=True, batch_size_max=8)
    b.get_num_of_exec_per_batch_size()

    return b


def main():
    for idx, prj in enumerate(project_list):
        # print(prj['name'])

        results = get_results(prj)

        # Batch4 num of execution
        print(results.batch4_num_of_exec)


main()
