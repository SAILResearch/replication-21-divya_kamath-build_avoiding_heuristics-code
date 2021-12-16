from batching import BatchBisect

y_test = ['passed', 'passed', 'passed', 'failed', 'passed', 'passed', 'passed', 'passed',
          'passed', 'failed', 'passed', 'failed', 'passed', 'passed', 'passed', 'passed']

num_of_exec_list = [16, 14, 12, 12, 18, 13, 18, 14, 13, 19, 18, 14, 15, 19, 20, 15, 15, 15, 15, 15]
num_of_merge_list = [0, 8, 5, 4, 9, 7, 7, 6, 6, 10, 10, 8, 8, 8, 7, 7, 7, 7, 7, 7]

b = BatchBisect(y_test)
b.get_num_of_exec_per_batch_size()

assert b.num_of_exec_list == num_of_exec_list
assert b.num_of_merge_list == num_of_merge_list
