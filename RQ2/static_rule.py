import csv
import numpy as np
import xlwt
import math
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

def process_data():

	project_names=['gradle.csv', 'cloud_controller_ng.csv', 'geoserver.csv']


	for project in project_names:
		string = "../even_data/" + project.split('.')[0] + '_train.csv'
		print(string)
		csv_file = csv.reader(open(string, 'r'))

		temp_data = []
		final_data = []

		for item in csv_file:
			temp_data.append(item)

		for i in range(len(temp_data[0])):
			temp = []
			for index in range(1, len(temp_data)):
				temp.append(temp_data[index][i])
			final_data.append(temp)

		indices = range(len(final_data[3]))

		#capture the metrics of source churn, test churn, file churn and team size in a list
		src_churn = []
		file_churn = []
		test_churn = []
		team_size = []
		build_result = []
		git_num_all_built_commits = []
		gh_num_commits_on_files_touched = []
		argument = []

		for index in indices:
			src_churn.append(float(final_data[1][index]))
			file_churn.append(float(final_data[3][index]))
			test_churn.append(float(final_data[4][index]))
			team_size.append(float(final_data[2][index]))
			build_result.append(final_data[5][index])

			argument.append([])

		'''for item in indices:
			if final_data[5][item] == 'passed':
				build_result.append(1)
			else:
				build_result.append(0)'''

		for index in range(len(src_churn)):
			argument[index].append(src_churn[index])
			argument[index].append(team_size[index])
			argument[index].append(file_churn[index])
			argument[index].append(test_churn[index])


		#start the training:
		X_train = np.array(argument)
		Y_train = np.array(build_result)


		string = "../even_data/" + project.split('.')[0] + '_test.csv'
		csv_file = csv.reader(open(string, 'r'))

		temp_data = []
		final_data = []

		for item in csv_file:
			temp_data.append(item)

		for i in range(len(temp_data[0])):
			temp = []
			for index in range(1, len(temp_data)):
				temp.append(temp_data[index][i])
			final_data.append(temp)

		indices = range(len(final_data[3]))

		#capture the metrics of source churn, test churn, file churn and team size in a list
		build_ids = []
		src_churn = []
		file_churn = []
		test_churn = []
		team_size = []
		build_result = []
		git_num_all_built_commits = []
		gh_num_commits_on_files_touched = []
		argument = []

		for index in indices:
			build_ids.append(float(final_data[0][index]))
			src_churn.append(float(final_data[1][index]))
			file_churn.append(float(final_data[3][index]))
			test_churn.append(float(final_data[4][index]))
			team_size.append(float(final_data[2][index]))
			build_result.append(final_data[5][index])

			argument.append([])

		for index in range(len(src_churn)):
			argument[index].append(src_churn[index])
			argument[index].append(team_size[index])
			argument[index].append(file_churn[index])
			argument[index].append(test_churn[index])


		#start the training:
		X_test = np.array(argument)
		Y_test = np.array(build_result)
		
		num_test = 0
		num_feature=4


		X_train = X_train.reshape((int(len(X_train)), num_feature))

		rf = RandomForestClassifier()
		predictor = rf.fit(X_train, Y_train)
		Y_result = []

		fp = 0
		tp = 0
		tn = 0
		fn = 0

		for b in range(len(X_test)):
			build = X_test[b]
			build = build.reshape((1, num_feature))
			value = predictor.predict(build)
			

			Y_result.append(value[0])
			if Y_test[b] == '0':
				if value[0] == '1':
					fp += 1
				elif value[0] == '0':
					tn += 1
			elif Y_test[b] == '1':
				if value[0] == '1':
					tp += 1
				elif value[0] == '0':
					fn += 1

		grouped_batch = []
		actual_results = []
		max_batch_size = 4
		num_builds = 0
		
		for b in range(len(X_test)):
			#computing the time required
			if Y_result[b] == '0':
				if len(grouped_batch) < max_batch_size:
					grouped_batch.append(build_ids[b])
					actual_results.append(Y_test[b])

				if len(grouped_batch) == max_batch_size:
					num_builds += 1
					if '0' in actual_results:
						num_builds += max_batch_size
				
					grouped_batch.clear()
					actual_results.clear()

		if len(grouped_batch) != 0:
			num_builds += 1
			if '0' in actual_results:
				num_builds += len(grouped_batch)


		print(num_builds)
		print(len(X_test))

		accuracy = (tp + tn) / (tp + tn + fp + fn)
		print('The efficiency of '+ project + ' is ' + str(accuracy))
		print(fp, fn, tp, tn)

		#print(split_index)
		#print(Y_result)
		#print(range(len(X_test)))

		flag = 0
		count = 0
		delay = []
		print(len(Y_result))
		print(len(Y_test))

		for b in range(len(Y_result)):
			if flag == 1:
				if Y_result[b] == '1':
					count += 1
				if Y_result[b] == '0':
					delay.append(count)
					count = 0
					flag = 0
			if Y_test[b] == '0':
				if Y_result[b] == '1':
					flag = 1

		delay.append(count)
		print(delay)
		print(sum(delay))

process_data()
project_names=['gradle.csv']

'''for project in project_names:

	#file to write results of SBS algorithm
	proj_result = 'static_rule_results/' + project.split('.')[0] + '_result.csv'
	result_file = open(proj_result, 'w')
	res_headers = ['index', 'duration', 'total_builds']
	res_writer = csv.writer(result_file)
	res_writer.writerow(res_headers)


	file_name = project.split('.')[0] + '_metrics.csv'
	csv_file = csv.reader(open(file_name, 'r'))
	built_commits = []
	build_time = 0
	total_builds = 0
	commit_num = 1
	flag = 0
	batches = []
	num = 0
	for build in csv_file:
		if num == 0:
			num = 1
			continue


		#if a build is predicted to fail, they will build it
		if build[-1] == '0':
			#add the build time
			build_time += int(build[1])
			total_builds += 1
			flag = 1

		#if a build passes,
		if build[-1] == '1':
			#check if this is the first build pass after failure
			if (flag == 1):
				flag = 0
				build_time += int(build[1])
				total_builds += 1

		if commit_num == 4:
			batches.append([int(build[0]), build_time, total_builds])
			res_writer.writerow([int(build[0]), build_time, total_builds])
			commit_num = 0
			built_commits.append(build_time)
			build_time = 0
			total_builds = 0

		commit_num += 1
	print(batches)
	print('Total time taken for builds:')
	#print(built_commits)'''










