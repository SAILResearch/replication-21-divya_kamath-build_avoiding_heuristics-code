from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import math
import csv

dataset_list = ['gradle.csv', 'vagrant.csv', 'geoserver.csv', 'cloudify.csv', 'cloud_controller_ng.csv']

for dataset in dataset_list:
	path = 'even_data/' + dataset
	temp_data = []
	final_data = []

	csv_file = csv.reader(open(path, 'r'))

	for item in csv_file:
		temp_data.append(item)

	for i in range(len(temp_data[0])):
		temp = []
		for index in range(1, len(temp_data)):
			temp.append(temp_data[index][i])
		final_data.append(temp)

	indices = range(len(final_data[33]))
	
	build_id = []
	src_churn = []
	file_churn = []
	test_churn = []
	team_size = []
	build_result = []
	git_num_all_built_commits = []
	gh_num_commits_on_files_touched = []
	argument = []

	for index in indices:
		build_id.append(float(final_data[0][index]))
		src_churn.append(float(final_data[23][index]))
		file_churn.append(float(final_data[27][index]))
		test_churn.append(float(final_data[24][index]))
		team_size.append(float(final_data[14][index]))

		argument.append([])

	for item in indices:
		if final_data[42][item] == 'passed':
			build_result.append(1)
		else:
			build_result.append(0)

	for index in range(len(src_churn)):
		argument[index].append(build_id[index])
		argument[index].append(src_churn[index])
		argument[index].append(team_size[index])
		argument[index].append(file_churn[index])
		argument[index].append(test_churn[index])

	
	#print(argument)
	#print(build_result)
	
	X_train, X_test, y_train, y_test = train_test_split(argument, build_result, random_state=1, stratify=build_result)

	file_name = path.split('.')[0]
	csv_headers = ['tr_build_id', 'gh_diff_src_churn', 'gh_team_size', 'gh_diff_files_modified', 'gh_diff_test_churn', 'tr_status']
	with open(file_name+'_train.csv', 'w') as train_file:
		writer = csv.writer(train_file)
		writer.writerow(csv_headers)

		for index in range(len(X_train)):
			row = X_train[index]
			row.append(y_train[index])
			writer.writerow(row)

	with open(file_name+'_test.csv', 'w') as test_file:
		writer = csv.writer(test_file)
		writer.writerow(csv_headers)

		for index in range(len(X_test)):
			row = X_test[index]
			row.append(y_test[index])
			writer.writerow(row)
	