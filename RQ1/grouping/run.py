from code.RQ4.dynamic_batching import risk_batch_get_number_of_runs_per_batch_size
from batching import BatchBisect, RiskTopN
from learning import IncrementalLearningModel
from utils import project_list


def test_all():
    output = "\n{:28} | {}\n".format('Project', 'Number of build execution')
    output += '-' * 56
    output += '\n'

    print('Creating learning model for each project ...')

    for prj in project_list:
        l = IncrementalLearningModel(prj['name'], 'RF', 30, 1, hyper_params={'n_estimators': 10})
        y_proba, y_test = l.get_predicted_data()
        #print(y_proba, y_test)
        output += '{:28} | {}\n'.format(prj['name'], str(len(y_test)).center(25))

    print(output)


def batch_bisect():
    output = "\n{:28} | {} | {}\n".format('Project', 'Number of build execution', 'Improvement over TestAll')
    output += '-' * 83
    output += '\n'

    print('Creating learning model for each project ...')

    for prj in project_list:
        l = IncrementalLearningModel(prj['name'], 'RF', 0, 0, hyper_params={'n_estimators': 10})
        y_proba, y_test = l.get_predicted_data()

        b = BatchBisect(y_test, prj['name'], stop_at_4=False, batch_size_max=16)
        b.get_num_of_exec_per_batch_size()

        output += '{:28} | {} | {} | {}\n'.format(
            prj['name'],
            str(b.lowest_num_of_exec).center(25),
            '{:.2f} %'.format(b.max_improvement).center(22),
            str(b.accuracy).center(25)
        )

    #print(output)


def batch_4():
    output = "\n{:28} | {} | {}\n".format('Project', 'Number of build execution', 'Improvement over TestAll')
    output += '-' * 83
    output += '\n'

    print('Creating learning model for each project ...')

    for prj in project_list:
        l = IncrementalLearningModel(prj['name'], 'RF', 30, 1, hyper_params={'n_estimators': 10})
        y_proba, y_test= l.get_predicted_data()

        b = BatchBisect(y_test, prj['name'], stop_at_4=True, batch_size_max=4)
        b.get_num_of_exec_per_batch_size()

        output += '{:28} | {} | {} | {} \n'.format(
            prj['name'],
            str(b.batch4_num_of_exec).center(25),
            '{:.2f} %'.format(b.batch4_improvement).center(22),
            str(b.accuracy).center(25)
        )

    #print(output)


def batch_stop_4():
    output = "\n{:28} | {} | {}\n".format('Project', 'Number of build execution', 'Improvement over TestAll')
    output += '-' * 83
    output += '\n'

    print('Creating learning model for each project ...')

    for prj in project_list:
        l = IncrementalLearningModel(prj['name'], 'RF', 30, 1, hyper_params={'n_estimators': 10})
        y_proba, y_test = l.get_predicted_data()

        b = BatchBisect(y_test, prj['name'], stop_at_4=True, batch_size_max=16)
        b.get_num_of_exec_per_batch_size()

        output += '{:28} | {} | {} | {} \n'.format(
            prj['name'],
            str(b.lowest_num_of_exec).center(25),
            '{:.2f} %'.format(b.max_improvement).center(22),
            str(b.accuracy).center(25)
        )

    #print(output)


def risk_top_n():
    output = "\n{:28} | {} | {}\n".format('Project', 'Number of build execution', 'Improvement over TestAll')
    output += '-' * 83
    output += '\n'

    print('Creating learning model for each project ...')

    for prj in project_list:
        l = IncrementalLearningModel(prj['name'], 'RF', 30, 1, hyper_params={'n_estimators': 10})
        y_proba, y_test = l.get_predicted_data()

        b = RiskTopN(y_proba, y_test, top_n=2, batch_size_max=8)
        b.get_num_of_exec_per_batch_size()

        output += '{:28} | {} | {} \n'.format(
            prj['name'],
            str(b.lowest_num_of_exec).center(25),
            '{:.2f} %'.format(b.max_improvement).center(22)
        )

    print(output)


def risk_batch():
    output = "\n{:28} | {} | {}\n".format('Project', 'Number of build execution', 'Improvement over TestAll')
    output += '-' * 83
    output += '\n'

    print('Creating learning model for each project ...')

    for prj in project_list:
        batch_size, improvement, num_of_exec, bb = risk_batch_get_number_of_runs_per_batch_size(prj)

        output += '{:28} | {} | {} \n'.format(
            prj['name'],
            str(min(num_of_exec)).center(25),
            '{:.2f} %'.format(max(improvement)).center(22)
        )

    print(output)


while True:
    option = int(input("""Please choose an option (1-6):
1- TestAll
2- BatchBisect
3- Batch4
4- BatchStop4
5- RiskTopN
6- RiskBatch
"""))

    if option == 1:
        test_all()
    elif option == 2:
        batch_bisect()
    elif option == 3:
        batch_4()
    elif option == 4:
        batch_stop_4()
    elif option == 5:
        risk_top_n()
    elif option == 6:
        risk_batch()

    option = input('To continue press c or press any other key to exit\n')

    if option != 'c':
        print('Bye!')
        exit(1)
