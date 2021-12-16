import matplotlib.pyplot as plt

from batching import RiskBatch
from utils import line_styles, project_list, color_list, get_y_test_y_proba

BATCH_SIZE_LIMIT = 20


def save_batch_size_list_csv(lav, file_name):
    s = ''

    for i in range(3200):
        line = ''
        c = True
        for item in lav:
            if len(item) > i:
                c = False
                line += str(item[i])

            line += ','

        if c:
            break

        line = line[:-1] + '\n'
        s += line

    with open('dynamic_batch_sizes.csv', 'w') as f:
        f.write(s)


def risk_batch_get_number_of_runs_per_batch_size(prj):
    def batch_test(batch):
        if len(batch) == 1:
            return 1

        no_failure = True

        for item in batch:
            if not item:
                no_failure = False
                break

        if no_failure:
            return 1

        if len(batch) <= 4:
            return 1 + len(batch)

        first_half = batch_test(batch[:len(batch) // 2])
        second_half = batch_test(batch[len(batch) // 2:])

        return 1 + first_half + second_half

    y_test, y_proba = get_y_test_y_proba(prj)

    x = list()
    y = list()
    z = list()
    bb = list()

    max_improvement = 0
    optimum_threshold = 0

    for risk_threshold in range(1, 20):
        risk_threshold /= 10

        number_of_runs = 0
        i = 0

        batch_size_list = []

        while i < len(y_test):
            batch_size = 0
            cumulative_risk = 0

            while cumulative_risk <= risk_threshold and i + batch_size < len(y_proba):
                cumulative_risk += y_proba[i + batch_size]
                batch_size += 1

                if batch_size >= BATCH_SIZE_LIMIT:
                    break

            if i + batch_size > len(y_test):
                selected_y_test = y_test[i:]
            else:
                selected_y_test = y_test[i:i + batch_size]

            batch_size_list.append(len(selected_y_test))
            number_of_runs += batch_test(selected_y_test)

            i += batch_size

        # plt.hist(batch_size_list, len(set(batch_size_list)), facecolor='blue', alpha=0.5)
        # plt.show()

        improvement = (1 - number_of_runs / len(y_test)) * 100

        if improvement > max_improvement:
            max_improvement = improvement
            bb = batch_size_list
            optimum_threshold = risk_threshold * 100

        # improvement = (1 - (number_of_runs / batch4)) * 100

        # print(improvement)

        x.append(risk_threshold * 100)
        y.append(improvement)
        z.append(number_of_runs)

    # print(len(bb))
    # print(bb)

    # print(optimum_threshold)

    return x, y, z, bb


def main():
    lav = list()

    for idx, prj in enumerate(project_list):
        av = [prj['name']]

        # print(prj['name'])
        x, y, z, bb = risk_batch_get_number_of_runs_per_batch_size(prj)
        y_test, y_proba = get_y_test_y_proba(prj)
        # b = DynamicBatching(y_proba, y_test)
        # x, y = b.get_num_of_exec_per_threshold()

        # av.extend(b.optimum_batch_size_list)

        # print(av)

        # print('Maximum Improvement', '{:.2f}%'.format(max(y)))
        # print('Lowest Num of Exec', min(z))
        # print(min(z))
        # print('{:.2f}%'.format(max(y)))
        # plt.plot(x, y, line_styles[idx % len(line_styles)], color=color_list[idx % len(color_list)])
        plt.plot(x, y, line_styles[idx % len(line_styles)])
        # print()
        # break

        lav.append(av)

        save_batch_size_list_csv(lav, '')

    # plt.hist(batch_size_list, len(set(batch_size_list)), facecolor='blue', alpha=0.5)
    # print(min(batch_size_list))
    # print(max(batch_size_list))
    # plt.title('Suggested Batch Size Histogram')
    # plt.xlabel('Batch Size')
    # plt.ylabel('Frequency')
    # plt.show()

    # plt.legend([prj['name'] for prj in project_list], bbox_to_anchor=(1, 1))
    plt.legend([prj['name'].split('--')[1].split('-')[0] for prj in project_list], ncol=3, loc=4)
    plt.xticks(range(10, 200, 20))
    plt.ylim(0, 70)
    plt.grid(axis='x')
    plt.xlabel('Cumulative Risk Threshold %')
    plt.ylabel('Improvement over TestAll %')
    plt.title('RiskBatch')
    # plt.show()
    plt.savefig("result_risk_batch.png", bbox_inches='tight')


# main()
