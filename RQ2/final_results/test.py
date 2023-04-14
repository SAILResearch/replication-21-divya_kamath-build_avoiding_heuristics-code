#from trial_data import ci, result
import pandas as pd

def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0)
    return Y_t


def compute_delay(ci, result):
	length = len(ci)
	missed_indexes = []
	built_indexes = []
	total_delay = 0

	for i in range(length):
		if ci[i] == 0:
			built_indexes.append(i)

		if result[i] == 0 and ci[i] != 0:
			missed_indexes.append(i)
	bp = 0
	mp = 0
	temp_delay = 0

	while bp < len(built_indexes):
		while mp < len(missed_indexes) and missed_indexes[mp] < built_indexes[bp]:
			temp_delay = built_indexes[bp] - missed_indexes[mp]
			total_delay += temp_delay
			mp += 1
		bp += 1

	while mp < len(missed_indexes):
		temp_delay = length - missed_indexes[mp]
		total_delay += temp_delay
		mp += 1
	return total_delay

def convert_to_list(x):
	x = x.split(',')
	val = []
	val.append(int(x[0].split('[')[1]))
	for i in range(1, len(x)-1):
		val.append(int(x[i]))
	val.append(int(x[-1].split(']')[0]))
	return val


def get_filename(filename):
	if '9_10' in filename:
		return filename.split('_9_10')[0]
	else:
		return filename.split('_ml_')[0]

files = open('files').readlines()

for f in files:
	aici = pd.read_csv(f[:-1])
	filename = get_filename(f[:-1])
	y_result = pd.read_csv('../../data/test_data/'+filename+ '_test.csv', usecols=['tr_status'])

	for index in range(len(aici)):
		heur = aici.iloc[index]
		ci = convert_to_list(heur[-1])
		result = output_values(y_result['tr_status'].tolist())
		delay = compute_delay(ci, result)
		aici.loc[index, 'total_delay'] = delay
	aici.to_csv(f[:-1], index=False)
