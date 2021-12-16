import csv
import numpy as np
import xlwt
import math
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

def process_data():
	project_names=['chef.csv', 'jruby.csv', 'opal.csv', 'orbeon-forms.csv']

	for project in project_names:
		string = "../data/" + project.split('.')[0] + '_train.csv'
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

		indices = range(len(final_data[33]))

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
			argument[index].append(src_churn[index])
			argument[index].append(team_size[index])
			argument[index].append(file_churn[index])
			argument[index].append(test_churn[index])


		#start the training:
		X_train = np.array(argument)
		Y_train = np.array(build_result)

		string = "../data/" + project.split('.')[0] + '_test.csv'
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

		indices = range(len(final_data[33]))

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
			argument[index].append(src_churn[index])
			argument[index].append(team_size[index])
			argument[index].append(file_churn[index])
			argument[index].append(test_churn[index])


		#start the training:
		X_test = np.array(argument)
		Y_test = np.array(build_result)
		#print(X)
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
			if Y_test[b] == 0:
				if value == 1:
					fp += 1
				elif value == 0:
					tn += 1
			elif Y_test[b] == 1:
				if value == 1:
					tp += 1
				elif value == 0:
					fn += 1

		accuracy = (tp + tn) / (tp + tn + fp + fn)
		print('The efficiency of '+ project + ' is ' + str(accuracy))
		print(fp, fn, tp, tn)

		#print(split_index)
		#print(Y_result)
		#print(range(len(X_test)))

		commit_values = []
		build_ids = []
		#print(X_test[0])
		for commit in range(len(X_test)):
			commit_values.append(final_data[43][commit])
			build_ids.append(final_data[0][commit])

		#print(commit_values)
		headers = ['Build_ID', 'Index','Duration', 'Build_Result', 'Actual_Result']

		file_name = project.split('.')[0] + '_real_metrics.csv'
		with open(file_name, 'w+') as file:
			writer = csv.DictWriter(file, headers)
			writer = csv.writer(file)
			writer.writerow(headers)
			for c_index in range(len(Y_result)):
				commit = Y_result[c_index]
				writer.writerow([build_ids[c_index], c_index+1, commit_values[c_index], Y_result[c_index], Y_test[c_index]])


process_data()
project_names=['chef.csv', 'jruby.csv', 'opal.csv', 'orbeon-forms.csv']

for project in project_names:

	#file to write results of SBS algorithm
	proj_result = 'results/' + project.split('.')[0] + '_real_result.csv'
	result_file = open(proj_result, 'w')
	res_headers = ['index', 'duration', 'total_builds']
	res_writer = csv.writer(result_file)
	res_writer.writerow(res_headers)


	file_name = project.split('.')[0] + '_real_metrics.csv'
	csv_file = csv.reader(open(file_name, 'r'))
	built_commits = []
	build_time = 0
	total_builds = 0
	actual_builds = 0
	commit_num = 1
	flag = 0
	batches = []
	num = 0
	b_id = 0
	for build in csv_file:
		if num == 0:
			num = 1
			continue


		#if a build is predicted to fail, they will build it
		if build[-2] == '0':
			#add the build time
			build_time += int(build[2])
			actual_builds += 1
			total_builds += 1
			b_id = build[0]
			flag = 1

		#if prev build has failed, build until you see a true build pass
		if flag == 1:
			if build[-1] == '0':
				if b_id != build[0]:
					build_time += int(build[2])
					actual_builds += 1
					total_builds += 1				
			if build[-1] == '1':
				#this is the first build pass after failure
				#go back to predicting
				if b_id != build[0]:
					build_time += int(build[2])
					actual_builds += 1
					total_builds += 1
				flag = 0


		'''#if a build passes,
		if build[-2] == '1':
			#check if this is the first build pass after failure
			if (flag == 1):
				flag = 0
				build_time += int(build[2])
				total_builds += 1'''

		if commit_num == 4:
			batches.append([int(build[1]), build_time, total_builds])
			res_writer.writerow([int(build[1]), build_time, total_builds])
			commit_num = 0
			built_commits.append(build_time)
			build_time = 0
			total_builds = 0

		commit_num += 1
	#print(batches)
	#print(total_builds)
	print(actual_builds)
	#print(len(csv_file))
	#print('Total time taken for builds:')
	#print(built_commits)










