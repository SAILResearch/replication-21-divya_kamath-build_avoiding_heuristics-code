import matplotlib.pyplot as plt

from batching import RiskTopN, BatchBisect
from utils import line_styles, project_list
from learning import IncrementalLearningModel

import sys

sys.setrecursionlimit(15000)

MAX_BATCH_SIZE = 20
RISK_TOP_N = 2


def get_number(prj, approach):
    l = IncrementalLearningModel(prj['name'], 'RF', 30, 1)
    y_proba, y_test = l.get_predicted_data()

    if approach == 'BatchBisect':
        b = BatchBisect(y_test, stop_at_4=False, batch_size_max=MAX_BATCH_SIZE)
    elif approach == 'BatchStop4':
        b = BatchBisect(y_test, stop_at_4=True, batch_size_max=MAX_BATCH_SIZE)
    elif approach == 'RiskTopN':
        b = RiskTopN(y_proba, y_test, top_n=RISK_TOP_N, batch_size_max=MAX_BATCH_SIZE)
    else:
        raise NotImplementedError

    x, y = b.get_num_of_exec_per_batch_size()

    print(b.lowest_num_of_exec)

    return x, y


def main():
    fig = plt.figure(figsize=(13, 7))

    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)

    for approach in ['BatchBisect', 'BatchStop4', 'RiskTopN']:
        for idx, prj in enumerate(project_list):
            x, y = get_number(prj, approach)

            if approach == 'BatchBisect':
                ax1.plot(x, y, line_styles[idx % len(line_styles)])
                ax1.grid(axis='x')
                ax1.set_title('BatchBisect')

            elif approach == 'BatchStop4':
                ax2.plot(x, y, line_styles[idx % len(line_styles)])
                ax2.grid(axis='x')
                ax2.set_title('BatchStop4 and Batch4')
                ax2.axvline(4, linestyle='-', color='#911010', linewidth=2)

            elif approach == 'RiskTopN':
                ax3.plot(x, y, line_styles[idx % len(line_styles)])
                ax3.grid(axis='x')
                ax3.set_title('RiskTop{}'.format(RISK_TOP_N))

    plt.setp(
        (ax1, ax2, ax3),
        xticks=range(0, MAX_BATCH_SIZE + 1, 2),
        xlim=(1, 20),
        ylim=(0, 70),
        xlabel='Batch Size',
        ylabel='Improvement %'
    )

    # fig.legend([prj['name'].split('--')[1].split('-')[0] for prj in project_list], ncol=3, loc="upper left", bbox_to_anchor=(1,1))
    lgd = fig.legend([prj['name'].split('--')[1].split('-')[0] for prj in project_list], ncol=9, loc=8,
                     bbox_to_anchor=(0.5, 0.2))

    # fig.subplots_adjust(bottom=0.25)
    fig.tight_layout()

    fig.show()
    fig.savefig("results.png", dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')


main()
