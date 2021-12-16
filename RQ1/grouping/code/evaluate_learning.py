from learning import IncrementalLearningModel
from utils import project_list


def compare_classifiers(prj):
    for _ in ['DT', 'RF', 'NB', 'MLP', 'LR', 'SGD']:
        l = IncrementalLearningModel(prj['name'], _, 30, 1, use_cache=False, hyper_params={})
        l.get_predicted_data()
        l.print_scores()


def find_optimum_hyper_params_for_random_forest(prj):
    for n_estimators in [10, 50, 100, 200, 400]:
        hyper_params = {'n_estimators': n_estimators}
        for _ in ['RF']:
            l = IncrementalLearningModel(prj, _, 30, 1, use_cache=False, hyper_params=hyper_params)
            l.get_predicted_data()
            l.print_scores()

    if prj['params']['n_estimators'] == 10:
        return

    for max_depth in [10, 20, 50, 100, 200, None]:
        hyper_params = {'max_depth': max_depth, 'n_estimators': prj['params']['n_estimators']}
        for _ in ['RF']:
            l = IncrementalLearningModel(prj['name'], _, 30, 1, use_cache=False, hyper_params=hyper_params)
            l.get_predicted_data()
            l.print_scores()

    for criterion in ['gini', 'entropy']:
        hyper_params = {'criterion': criterion, 'n_estimators': prj['params']['n_estimators']}
        for _ in ['RF']:
            l = IncrementalLearningModel(prj['name'], _, 30, 1, use_cache=False, hyper_params=hyper_params)
            l.get_predicted_data()
            l.print_scores()

    for min_samples_split in [2, 5, 10, 20, 50, 100]:
        hyper_params = {'min_samples_split': min_samples_split, 'n_estimators': prj['params']['n_estimators']}
        for _ in ['RF']:
            l = IncrementalLearningModel(prj['name'], _, 30, 1, use_cache=False, hyper_params=hyper_params)
            l.get_predicted_data()
            l.print_scores()

    for min_samples_leaf in [1, 2, 5, 10, 20, 50, 100]:
        hyper_params = {'min_samples_leaf': min_samples_leaf, 'n_estimators': prj['params']['n_estimators']}
        for _ in ['RF']:
            l = IncrementalLearningModel(prj['name'], _, 30, 1, use_cache=False, hyper_params=hyper_params)
            l.get_predicted_data()
            l.print_scores()


def compare_with_default_hyper_params(prj):
    l = IncrementalLearningModel(prj['name'], 'RF', 30, 1, use_cache=False, hyper_params=prj['params'])
    l.get_predicted_data()
    l.print_scores()

    l = IncrementalLearningModel(prj['name'], 'RF', 30, 1, use_cache=False, hyper_params={})
    l.get_predicted_data()
    l.print_scores()

    print()


def main():
    for idx, prj in enumerate(project_list):
        compare_with_default_hyper_params(prj)
        find_optimum_hyper_params_for_random_forest(prj)
        compare_classifiers(prj)


main()
