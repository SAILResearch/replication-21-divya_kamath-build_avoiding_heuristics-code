from math import log2

import matplotlib.pyplot as plt

MAX_BATCH_SIZE = 11


def get_number_of_exec():
    x_data = list()
    y_data = list()
    y2_data = list()
    y3_data = list()
    y5_data = [3, 4]
    y4_data = list()

    # x_data.append(1)
    # y_data.append(1)
    # y2_data.append(1)
    # y3_data.append(1)
    y4_data.append(0)

    for batch_size in range(2, MAX_BATCH_SIZE):
        x = batch_size
        y = 2 * log2(x) + 1
        test_list = [False]
        test_list += [True for _ in range(batch_size - 1)]

        x_data.append(x)
        y_data.append(y)
        y2_data.append(x + 1)
        y3_data.append(2 * x - 1)
        y4_data.append(0)

    for batch_size in range(4, MAX_BATCH_SIZE):
        x = batch_size
        y5_data.append(1.5 * x - 1)

    fig, ax = plt.subplots(1)

    ax.plot(x_data, y_data, 'g--', label='BatchBisect/BatchStop4 minimum (2 * log2(n) + 1)')
    ax.plot(x_data, y2_data, 'b-', label='Dorfman - Batching without bisection (n + 1)')
    ax.plot(x_data, y3_data, 'r:', label='BatchBisect Maximum (2 * n - 1)')
    ax.plot(x_data, y5_data, '--', color='purple' ,label='BatchStop4 Maximum (1.5 * n - 1)')
    ax.plot(4, 5, marker='o', color='#871912', label='Batch4 (4 + 1)', ls='', ms=10, mfc='none', mew=2)

    plt.xlim(1, MAX_BATCH_SIZE)
    plt.ylim(1, MAX_BATCH_SIZE)
    plt.xticks(range(1, MAX_BATCH_SIZE, 1))
    plt.yticks(range(1, MAX_BATCH_SIZE + 1, 1))
    plt.xlabel('Batch Size')
    plt.ylabel('Number of executions')
    plt.grid(True)
    plt.legend()
    plt.savefig("logn.png", format="png", dpi=300)
    plt.show()


if __name__ == "__main__":
    get_number_of_exec()
