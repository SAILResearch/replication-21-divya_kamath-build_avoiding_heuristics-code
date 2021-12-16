from batching import BatchBisect

# y_proba = [0.1, 0.2, 0.0, 0.8, 0.3, 0.0, 0.0, 0.2, 0.1, 0.6,
#            0.0, 0.7, 0.0, 0.2, 0.3, 0.0, 0.6, 0.9, 0.1, 0.3,
#            0.0, 0.8, 0.7, 0.2, 0.3, 0.0, 0.9, 0.3, 0.1, 0.2,
#            0.2, 0.1, 0.1, 0.2, 0.3, 0.0, 0.2, 0.8, 0.1, 0.1]

y_test = ['passed', 'passed', 'passed', 'failed', 'passed', 'passed', 'passed', 'passed', 'passed', 'failed',
          'passed', 'failed', 'passed', 'passed', 'passed', 'passed', 'failed', 'failed', 'passed', 'passed',
          'passed', 'failed', 'failed', 'passed', 'passed', 'passed', 'failed', 'passed', 'passed', 'passed',
          'passed', 'passed', 'passed', 'passed', 'passed', 'passed', 'passed', 'failed', 'passed', 'passed']

b = BatchBisect(y_test, 40)
x, y = b.get_num_of_exec_per_batch_size()

assert x == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
assert y == [0.0, 0.09999999999999998, 0.19999999999999996, 0.15000000000000002, -0.125, 0.09999999999999998, -0.10000000000000009, 0.025000000000000022, 0.025000000000000022, -0.2250000000000001, -0.125, 0.025000000000000022, -0.1499999999999999, -0.17500000000000004, -0.2250000000000001, -0.02499999999999991, -0.125, -0.02499999999999991, -0.125, -0.2749999999999999]

