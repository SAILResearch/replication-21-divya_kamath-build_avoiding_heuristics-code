import operator

import matplotlib.pyplot as plt
import numpy as np


class BatchBisect:
    stop_at_4 = True
    batch_size_min = 1
    batch_size_max = 20

    batch_size_list = []
    num_of_exec_list = []
    num_of_merge_list = []
    improvement_list = []

    batch2_num_of_exec = 0
    batch2_num_of_merge = 0
    batch2_improvement = 0

    batch4_num_of_exec = 0
    batch4_num_of_merge = 0
    batch4_improvement = 0

    batch6_num_of_exec = 0
    batch6_num_of_merge = 0
    batch6_improvement = 0

    batch8_num_of_exec = 0
    batch8_num_of_merge = 0
    batch8_improvement = 0

    batch10_num_of_exec = 0
    batch10_num_of_merge = 0
    batch10_improvement = 0

    y_proba = []
    y_test = []

    num_of_exec = 0
    num_of_merge = 0
    max_improvement = -1000
    best_batch_size = 1
    lowest_num_of_exec = 0
    optimum_num_of_merge = 0

    def __init__(self, y_test, stop_at_4=True, batch_size_max=20, no_bisect=False):
        self.y_test = y_test
        self.num_of_exec = 0
        self.num_of_merge = 0
        self.num_of_merge_list = []
        self.num_of_exec_list = []
        self.batch_size_list = []
        self.improvement_list = []
        self.stop_at_4 = stop_at_4
        self.batch_size_max = batch_size_max
        self.no_bisect = no_bisect

    def test_batch(self, partial_y_test):
        self.num_of_exec += 1

        if len(partial_y_test) == 1:
            return

        self.num_of_merge += 1

        no_failure = True

        for item in partial_y_test:
            if item != 'passed':
                no_failure = False
                break

        if no_failure:
            return

        if self.no_bisect:
            self.num_of_exec += len(partial_y_test)
            return

        if self.stop_at_4 and len(partial_y_test) <= 4:
            self.num_of_exec += len(partial_y_test)
            return

        self.test_batch(partial_y_test[:len(partial_y_test) // 2])
        self.test_batch(partial_y_test[len(partial_y_test) // 2:])

    def get_num_of_exec_per_batch_size(self):

        for batch_size in range(self.batch_size_min, self.batch_size_max + 1):
            self.num_of_exec = 0
            self.num_of_merge = 0

            for i in range(0, len(self.y_test), batch_size):
                if i + batch_size > len(self.y_test):
                    selected_y_test = self.y_test[i:]
                else:
                    selected_y_test = self.y_test[i:i + batch_size]

                self.test_batch(selected_y_test)

            improvement = (1 - self.num_of_exec / len(self.y_test)) * 100

            if improvement > self.max_improvement:
                self.max_improvement = improvement
                self.best_batch_size = batch_size
                self.lowest_num_of_exec = self.num_of_exec
                self.optimum_num_of_merge = self.num_of_merge

            self.batch_size_list.append(batch_size)
            self.num_of_exec_list.append(self.num_of_exec)
            self.num_of_merge_list.append(self.num_of_merge)
            self.improvement_list.append(improvement)

            if batch_size == 2:
                self.batch2_num_of_exec = self.num_of_exec
                self.batch2_num_of_merge = self.num_of_merge
                self.batch2_improvement = improvement

            if batch_size == 4:
                self.batch4_num_of_exec = self.num_of_exec
                self.batch4_num_of_merge = self.num_of_merge
                self.batch4_improvement = improvement

            if batch_size == 6:
                self.batch6_num_of_exec = self.num_of_exec
                self.batch6_num_of_merge = self.num_of_merge
                self.batch6_improvement = improvement

            if batch_size == 8:
                self.batch8_num_of_exec = self.num_of_exec
                self.batch8_num_of_merge = self.num_of_merge
                self.batch8_improvement = improvement

            if batch_size == 10:
                self.batch10_num_of_exec = self.num_of_exec
                self.batch10_num_of_merge = self.num_of_merge
                self.batch10_improvement = improvement

        return self.batch_size_list, self.improvement_list

    def print_report(self):
        print('Max Improvement: {:.2f}'.format(self.max_improvement))
        print('{:.2f}%'.format(self.max_improvement))

        print('Lowest Num of Exec: ', self.lowest_num_of_exec)
        print(self.lowest_num_of_exec)

        print('Optimum Num of Merge', self.optimum_num_of_merge)
        print(self.optimum_num_of_merge)

        print('Best Batch Size: ', self.best_batch_size)
        print(self.best_batch_size)

        print('TestAll Num of Exec: ', len(self.y_test))
        print(len(self.y_test))

        print('Batch2 Num of Exec:', self.batch2_num_of_exec)
        print(self.batch2_num_of_exec)

        print('Batch2 Num of Merge:', self.batch2_num_of_merge)
        print(self.batch2_num_of_merge)

        print('Batch2 Improvement:', self.batch2_improvement)
        print(self.batch2_improvement)

        print('Batch4 Num of Exec:', self.batch4_num_of_exec)
        print(self.batch4_num_of_exec)

        print('Batch4 Num of Merge:', self.batch4_num_of_merge)
        print(self.batch4_num_of_merge)

        print('Batch4 Improvement:', self.batch4_improvement)
        print(self.batch4_improvement)

        print('Batch6 Num of Exec:', self.batch6_num_of_exec)
        print(self.batch6_num_of_exec)

        print('Batch6 Num of Merge:', self.batch6_num_of_merge)
        print(self.batch6_num_of_merge)

        print('Batch6 Improvement:', self.batch6_improvement)
        print(self.batch6_improvement)

        print('Batch8 Num of Exec:', self.batch8_num_of_exec)
        print(self.batch8_num_of_exec)

        print('Batch8 Num of Merge:', self.batch8_num_of_merge)
        print(self.batch8_num_of_merge)

        print('Batch8 Improvement:', self.batch8_improvement)
        print(self.batch8_improvement)

        print('Batch10 Num of Exec:', self.batch10_num_of_exec)
        print(self.batch10_num_of_exec)

        print('Batch10 Num of Merge:', self.batch10_num_of_merge)
        print(self.batch10_num_of_merge)

        print('Batch10 Improvement:', self.batch10_improvement)
        print(self.batch10_improvement)

    def plot_batch_size_histogram(self):
        plt.hist(self.batch_size_list)
        plt.gca().set(title='Batch Size Histogram', ylabel='Frequency')
        plt.show()


class RiskTopN:
    top_n = 1
    batch_size_max = 20

    batch_size_list = []
    num_of_exec_list = []
    num_of_merge_list = []
    improvement_list = []

    y_proba = []
    y_test = []

    num_of_exec = 0
    num_of_merge = 0
    max_improvement = -1000
    best_batch_size = 1
    lowest_num_of_exec = 0
    optimum_num_of_merge = 0

    def __init__(self, y_proba, y_test, top_n, batch_size_max=20):
        self.y_test = y_test
        self.y_proba = y_proba
        self.top_n = top_n
        self.batch_size_max = batch_size_max

        self.num_of_exec = 0
        self.num_of_merge = 0

        self.batch_size_list = []
        self.num_of_exec_list = []
        self.num_of_merge_list = []
        self.improvement_list = []

    def get_num_of_exec_per_batch_size(self):
        for batch_size in range(1, self.batch_size_max + 1):
            self.num_of_exec = 0
            self.num_of_merge = 0

            for i in range(0, len(self.y_test), batch_size):
                if i + batch_size > len(self.y_test):
                    selected_y_test = self.y_test[i:]
                    selected_y_proba = self.y_proba[i:]
                else:
                    selected_y_test = self.y_test[i:i + batch_size]
                    selected_y_proba = self.y_proba[i:i + batch_size]

                self.test_batch(selected_y_proba, selected_y_test)

            improvement = (1 - self.num_of_exec / len(self.y_test)) * 100

            if improvement > self.max_improvement:
                self.max_improvement = improvement
                self.best_batch_size = batch_size
                self.lowest_num_of_exec = self.num_of_exec
                self.optimum_num_of_merge = self.num_of_merge

            self.batch_size_list.append(batch_size)
            self.num_of_exec_list.append(self.num_of_exec)
            self.num_of_merge_list.append(self.num_of_merge)
            self.improvement_list.append(improvement)

        return self.batch_size_list, self.improvement_list

    def test_batch(self, partial_y_proba, partial_y_test):

        while True:
            before_length = len(partial_y_test)
            partial_y_proba, partial_y_test = self.isolate_risky_jobs(partial_y_proba, partial_y_test)
            after_length = len(partial_y_test)

            self.num_of_exec += 1

            if len(partial_y_test) == 1:
                return

            if before_length == after_length:
                break

            self.num_of_merge += 1

            no_failure = True

            for item in partial_y_test:
                if item != 'passed':
                    no_failure = False
                    break

            if no_failure:
                return

            if len(partial_y_test) <= 4:
                self.num_of_exec += len(partial_y_test)
                return

    def isolate_risky_jobs(self, partial_y_proba, partial_y_test):
        if len(partial_y_proba) == 1:
            return partial_y_proba, partial_y_test

        for _ in range(self.top_n):
            if not len(partial_y_proba):
                break

            top_n_idx, top_n_value = max(enumerate(partial_y_proba), key=operator.itemgetter(1))

            if top_n_value > 0:
                self.num_of_exec += 1
                partial_y_test = np.delete(partial_y_test, top_n_idx)
                partial_y_proba = np.delete(partial_y_proba, top_n_idx)

        return partial_y_proba, partial_y_test

    def print_report(self):
        print('Max Improvement: {:.2f}'.format(self.max_improvement))
        print('{:.2f}%'.format(self.max_improvement))

        print('Lowest Num of Exec: ', self.lowest_num_of_exec)
        print(self.lowest_num_of_exec)

        print('Optimum Num of Merge', self.optimum_num_of_merge)
        print(self.optimum_num_of_merge)

        print('Best Batch Size: ', self.best_batch_size)
        print(self.best_batch_size)

        print('TestAll Num of Exec: ', len(self.y_test))
        print(len(self.y_test))


class RiskIsolate:
    risk_threshold = int()

    batch_size_list = []
    num_of_exec_list = []
    num_of_merge_list = []
    improvement_list = []

    y_proba = []
    y_test = []

    num_of_exec = 0
    num_of_merge = 0
    max_improvement = -1000
    best_batch_size = 1
    lowest_num_of_exec = 0
    optimum_num_of_merge = 0

    def __init__(self, y_proba, y_test, risk_threshold, **kwargs):
        self.y_test = y_test
        self.y_proba = y_proba
        self.risk_threshold = risk_threshold

        self.num_of_exec = 0
        self.num_of_merge = 0

    def get_num_of_exec_per_batch_size(self):

        for batch_size in range(1, 8 + 1):
            self.num_of_exec = 0
            self.num_of_merge = 0

            for i in range(0, len(self.y_test), batch_size):
                if i + batch_size > len(self.y_test):
                    selected_y_test = self.y_test[i:]
                    selected_y_proba = self.y_proba[i:]
                else:
                    selected_y_test = self.y_test[i:i + batch_size]
                    selected_y_proba = self.y_proba[i:i + batch_size]

                self.test_batch(selected_y_proba, selected_y_test)

            improvement = (1 - self.num_of_exec / len(self.y_test)) * 100

            if improvement > self.max_improvement:
                self.max_improvement = improvement
                self.best_batch_size = batch_size
                self.lowest_num_of_exec = self.num_of_exec
                self.optimum_num_of_merge = self.num_of_merge

            self.batch_size_list.append(batch_size)
            self.num_of_exec_list.append(self.num_of_exec)
            self.num_of_merge_list.append(self.num_of_merge)
            self.improvement_list.append(improvement)

        return self.batch_size_list, self.improvement_list

    def test_batch(self, partial_y_proba, partial_y_test):
        partial_y_proba, partial_y_test = self.isolate_risky_jobs(partial_y_proba, partial_y_test)

        self.num_of_exec += 1

        if len(partial_y_test) == 1:
            return

        self.num_of_merge += 1

        no_failure = True

        for item in partial_y_test:
            if item != 'passed':
                no_failure = False
                break

        if no_failure:
            return

        if len(partial_y_test) <= 4:
            self.num_of_exec += len(partial_y_test)
            return

        self.test_batch(partial_y_proba[:len(partial_y_proba) // 2],
                        partial_y_test[:len(partial_y_test) // 2])

        self.test_batch(partial_y_proba[len(partial_y_proba) // 2:],
                        partial_y_test[len(partial_y_test) // 2:])

    def isolate_risky_jobs(self, partial_y_proba, partial_y_test):
        if len(partial_y_proba) == 1:
            return partial_y_proba, partial_y_test

        for _ in range(len(partial_y_proba)):
            if not len(partial_y_proba):
                break

            top_n_idx, top_n_value = max(enumerate(partial_y_proba), key=operator.itemgetter(1))

            if top_n_value > self.risk_threshold:
                self.num_of_exec += 1
                partial_y_test = np.delete(partial_y_test, top_n_idx)
                partial_y_proba = np.delete(partial_y_proba, top_n_idx)

        return partial_y_proba, partial_y_test


class RiskBatch:
    cumulative_risk_threshold = int()
    batch_size_limit = 20

    threshold_list = []
    optimum_batch_size_list = []
    num_of_exec_list = []
    num_of_merge_list = []
    improvement_list = []

    y_proba = []
    y_test = []

    num_of_exec = 0
    num_of_merge = 0
    max_improvement = -1000
    best_threshold = 1
    lowest_num_of_exec = 0
    optimum_num_of_merge = 0

    def __init__(self, y_proba, y_test, **kwargs):
        self.y_test = y_test
        self.y_proba = y_proba

        self.num_of_exec = 0
        self.num_of_merge = 0
        self.batch_size_list = []

    def test_batch(self, partial_y_test):
        self.num_of_exec += 1

        if len(partial_y_test) == 1:
            return

        self.num_of_merge += 1

        no_failure = True

        for item in partial_y_test:
            if item != 'passed':
                no_failure = False
                break

        if no_failure:
            return

        if len(partial_y_test) <= 4:
            self.num_of_exec += len(partial_y_test)
            return

        self.test_batch(partial_y_test[:len(partial_y_test) // 2])
        self.test_batch(partial_y_test[len(partial_y_test) // 2:])

    def get_num_of_exec_per_threshold(self):
        for risk_threshold in range(1, 20):
            risk_threshold /= 10

            self.num_of_exec = 0
            self.num_of_merge = 0

            i = 0

            batch_size_list = []

            while i < len(self.y_test):
                batch_size = 0
                cumulative_risk = 0

                while cumulative_risk <= risk_threshold and i + batch_size < len(self.y_proba):
                    cumulative_risk += self.y_proba[i + batch_size]
                    batch_size += 1

                    if batch_size >= self.batch_size_limit:
                        break

                if i + batch_size > len(self.y_test):
                    selected_y_test = self.y_test[i:]
                else:
                    selected_y_test = self.y_test[i:i + batch_size]

                batch_size_list.append(len(selected_y_test))
                self.test_batch(selected_y_test)

                i += batch_size

            improvement = (1 - self.num_of_exec / len(self.y_test)) * 100

            if improvement > self.max_improvement:
                self.max_improvement = improvement
                self.best_threshold = risk_threshold
                self.lowest_num_of_exec = self.num_of_exec
                self.optimum_num_of_merge = self.num_of_merge
                self.optimum_batch_size_list = batch_size_list

            self.threshold_list.append(risk_threshold)
            self.num_of_exec_list.append(self.num_of_exec)
            self.num_of_merge_list.append(self.num_of_merge)
            self.improvement_list.append(improvement)

        return self.threshold_list, self.improvement_list
