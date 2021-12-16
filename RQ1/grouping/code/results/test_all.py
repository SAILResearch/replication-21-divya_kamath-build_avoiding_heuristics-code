from utils import project_list
from learning import IncrementalLearningModel


def get_testing_dataset_size(prj):
    l = IncrementalLearningModel(prj['name'], 'RF', 30, 1)
    y_proba, y_test = l.get_predicted_data()

    return len(y_test)


def main():
    for idx, prj in enumerate(project_list):
        print(prj['name'])

        # TestAll num of execution
        print(get_testing_dataset_size(prj))


main()
