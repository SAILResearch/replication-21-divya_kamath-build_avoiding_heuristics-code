import pandas as pd
import numpy as np
from numpy import argmax
from numpy import sqrt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.utils import resample
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold
from matplotlib import pyplot
from statistics import median
import pickle
import csv
import warnings
import datetime
import multiprocess
warnings.filterwarnings("ignore")

def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:  
        median = (data[size // 2] + data[size // 2 - 1]) / 2
        data[0] = median
    if size % 2 == 1:  
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]

def get_first_failures(df):
    
    results = df['tr_status'].tolist()
    length = len(results)
    verdict = ['keep']
    prev = results[0]
    
    for i in range(1, length):
        if results[i] == 0:
            if prev == 0:
                verdict.append('discard')
                #print(i+1)
            else:
                verdict.append('keep')
        else:
            verdict.append('keep')
        prev = results[i]
    
    df['verdict'] = verdict
    df = df[ df['verdict'] == 'keep' ]
    df.drop('verdict', inplace=True, axis=1)
    return df

def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0) 
    return Y_t

def get_data(project_path):
	columns = ['tr_build_id', 'git_num_all_built_commits', 'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_modified', 'tr_status']
	df = pd.read_csv(project_path, usecols = columns)
	'''src_churn = df['git_diff_src_churn'].tolist()
	file_churn = df['gh_diff_files_modified'].tolist()
	test_churn = df['git_diff_test_churn'].tolist()
	num_commits = df['git_num_all_built_commits'].tolist()
	build_result = output_values(df['tr_status'])

	argument = []
	for index in range(len(src_churn)):
		argument.append([src_churn[index], file_churn[index], test_churn[index], num_commits[index]])

	X = np.array(argument)
	y = np.array(build_result)'''
	df['tr_status'] = output_values(df['tr_status'])

	return df

def get_duration_data(project_path):
	columns = ['tr_build_id', 'tr_duration']
	df = pd.read_csv(project_path, usecols = columns)
	return df



def with_cv_val(p_name):
		
	string = "../data/train_data/" + p_name.split('.')[0] + "_train.csv"
	train_data = get_data(string)

	n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
	max_depth = [int(x) for x in np.linspace(10, 110, num=5)]

	param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
	forest = RandomForestClassifier()
	grid_search = GridSearchCV(estimator = forest, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 0)

	#pkl_file = '../data/data_pickles/'+ p_name.split('.')[0] + '_indexes.pkl'
	#with open(pkl_file, 'rb') as load_file:
	#	train_build_ids = pickle.load(load_file)
	#	test_build_ids = pickle.load(load_file)

	#train_data = project [ project['tr_build_id'].isin(train_build_ids)]
	#test_data = project [ project['tr_build_id'].isin(test_build_ids)]
	#test_result = test_data['tr_status'].tolist()

	train_result = train_data['tr_status'].tolist()


	best_n_estimators = []
	best_max_depth = []

	best_f1 = 0
	best_f1_sample = 0
	best_f1_sample_result = 0
	best_f1_estimator = 0
	best_thresholds = []



	for i in range(100):
		print('Bootstrapping {} for {}'.format(i, p_name))

		while True:
			print('Here for {} {}'.format(i, p_name))
			sample_train = resample(train_data, replace=True, n_samples=len(train_data))
			sample_train_result = sample_train['tr_status']

			build_ids = sample_train['tr_build_id'].tolist()
			sample_test = train_data [~train_data['tr_build_id'].isin(build_ids)] 
			sample_test_result = sample_test['tr_status']
			
			if len(sample_test_result) != 0:
				break

		sample_train.drop('tr_status', inplace=True, axis=1)
		sample_train.drop('tr_build_id', inplace=True, axis=1)
		sample_test.drop('tr_status', inplace=True, axis=1)
		sample_test.drop('tr_build_id', inplace=True, axis=1)
		
		print('Training {} for {}'.format(i, p_name))
		grid_search.fit(sample_train, sample_train_result)
		sample_pred_vals = grid_search.predict_proba(sample_test)

		pred_vals = sample_pred_vals[:, 1]
		fpr, tpr, t = roc_curve(sample_test_result, pred_vals)
		gmeans = sqrt(tpr * (1-fpr))
		ix = argmax(gmeans)
		bt = t[ix]
		best_thresholds.append(bt)

		final_pred_result = []
		for j in range(len(pred_vals)):
			if pred_vals[j] > bt:
				final_pred_result.append(1)
			else:
				final_pred_result.append(0)

		try:
			f1 = f1_score(sample_test_result, final_pred_result)
		except:
			print('')

		if f1 > best_f1:
			best_f1 = f1
			best_f1_sample = sample_train
			best_f1_sample_result = sample_train_result
			best_f1_estimator = grid_search.best_estimator_

		best_n_estimators.append(grid_search.best_params_['n_estimators'])
		best_max_depth.append(grid_search.best_params_['max_depth'])
	
	#completed bootstrapping
	threshold = median(best_thresholds)
	n_estimator = median(best_n_estimators)
	max_depth = median(best_max_depth)

	forest = RandomForestClassifier(n_estimators=int(n_estimator), max_depth=int(max_depth))
	forest.fit(best_f1_sample, best_f1_sample_result)

	filename = '../RQ2/dump_data/rq2_' + p_name.split('.')[0] + '_best_model.pkl'
	model_file = open(filename, 'rb')
	forest = pickle.load(model_file)

	#test_builds = test_data['tr_build_id'].tolist()
	#test_data.drop('tr_build_id', inplace=True, axis=1)
	#test_data.drop('tr_status', inplace=True, axis=1)


	string = "../data/test_data/" + p_name.split('.')[0] + "_test.csv"
	test_data = get_data(string)
	y_val = test_data['tr_status'].tolist()
	test_data.drop('tr_status', inplace=True, axis=1)
	test_data.drop('tr_build_id', inplace=True, axis=1)


	y_pred = forest.predict(test_data)
	print(y_pred)
	print(y_val)

	print(precision_score(y_val, y_pred))
	print(recall_score(y_val, y_pred))

	#Since we have already divided train and test data, we don't need to collect build ids again
	result_df = get_duration_data(string)
	result_df['Build_Result'] = y_pred
	result_df['Actual_Result'] = y_val
	result_df['Index'] = list(range(1, len(y_val)+1))

	#print(commit_values)
	headers = ['tr_build_ids', 'tr_duration','Duration', 'Build_Result', 'Actual_Result']

	file_name = './' + p_name.split('.')[0] + '_200_metrics.csv'
	result_df.to_csv(file_name)


def without_cv_val():

	project_names=['rails.csv', 'jruby.csv', 'metasploit-framework.csv', 'heroku.csv', 'vagrant.csv', 'opal.csv', 'cloudify.csv', 'cloud_controller_ng.csv', 'rubinius.csv', 'open-build-service.csv', 'gradle.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'puppet.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv']
	
	for project in project_names[:]:
		
		string = "../data/datasets/all_datasets/" + project
		X, y = get_data(string)

		KF = KFold(n_splits=10)


		less = 0
		more = 0
		yes = 0

		#print(X)
		num_test = 0
		num_feature=4

		precision = []
		recall = []
		build_save = []
		fitted_model = []

		for train_index, test_index in KF.split(X):
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = y[train_index], y[test_index]

			num_test = num_test + len(Y_test)
			X_train = X_train.reshape((int(len(X_train)), num_feature))

			try:
				rf = RandomForestClassifier(class_weight={0:0.05,1:1})
				predictor = rf.fit(X_train, Y_train)
			except:
				rf = RandomForestClassifier()
				predictor = rf.fit(X_train, Y_train)

			X_test = X_test.reshape((int(len(X_test)), num_feature))
			Y_result=(predictor.predict(X_test))


			for index in range(len(Y_result)):
				if Y_result[index]==0 and Y_test[index]==0 and Y_test[index-1]==1 and Y_result[index-1]==1:
					yes=yes+1
				if Y_result[index] == 0 and Y_result[index-1]==1:
					more=more+1
				if Y_test[index] == 0 and Y_test[index-1]==1:
					less=less+1

			if less != 0:
				recall0 = yes/less
				if more == 0:
					precision0=1
				else:
					precision0 = yes / more

			precision0 = precision_score(Y_test, Y_result)
			recall0 = recall_score(Y_test, Y_result)

			precision.append(precision0)
			recall.append(recall0)
			build_save.append(1-more/num_test)
			fitted_model.append(rf)


		best_precision = max(precision)
		max_index = precision.index(best_precision)

		print(precision)
		print(best_precision)

		best_fit_model = fitted_model[max_index]

		string = "../data/even_data/test/" + project.split('.')[0] + "_test.csv"
		X_val, y_val = get_data(string)

		y_pred = best_fit_model.predict(X_val)
		print(y_pred)
		print(y_val)

		print(precision_score(y_val, y_pred))
		print(recall_score(y_val, y_pred))

		#Since we have already divided train and test data, we don't need to collect build ids again
		result_df = get_duration_data(string)
		result_df['Build_Result'] = y_pred
		result_df['Actual_Result'] = y_val
		result_df['Index'] = list(range(1, len(y_val)+1))

		#print(commit_values)
		headers = ['tr_build_ids', 'tr_duration','Duration', 'Build_Result', 'Actual_Result']

		file_name = './' + project.split('.')[0] + '_abcd_metrics.csv'
		result_df.to_csv(file_name)


project_names=['rails.csv', 'jruby.csv', 'metasploit-framework.csv', 'vagrant.csv', 'opal.csv', 'cloudify.csv', 'cloud_controller_ng.csv', 'rubinius.csv', 'open-build-service.csv', 'gradle.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'puppet.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv']

new_projects = ['rubinius.csv', 'gradle.csv', 'loomio.csv', 'fog.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv', 'puppet.csv', 'cloud_controller_ng.csv', 'rails.csv']

# jobs = []
# for p_name in project_names:
# 	q = multiprocess.Process(target=with_cv_val, args=(p_name,))
# 	jobs.append(q)
# 	q.start()

# for j in jobs:
#     j.join()





















def validation():
	global project_names
	
	batchsize = [1,2,4,8,16,32]
	batch_result = 'results/final_result.csv'
	final_file = open(batch_result, 'w')
	final_headers = ['project', 'method', 'batch', 'reqd_builds', 'delay']
	final_writer = csv.writer(final_file)
	final_writer.writerow(final_headers)

	for project in project_names:

		for b in batchsize:
		

			#file to write results of SBS algorithm
			proj_result = 'results/batch_' + str(b) + '_' + project.split('.')[0] + '_result.csv'
			result_file = open(proj_result, 'w')
			res_headers = ['index', 'duration', 'total_builds']
			res_writer = csv.writer(result_file)
			res_writer.writerow(res_headers)

			file_name = './final_sbs_results/' + project.split('.')[0] + '_200_metrics.csv'

			csv_file = pd.read_csv(file_name)

			actual_results = csv_file['Actual_Result'].tolist()
			pred_results = csv_file['Build_Result'].tolist()

			delay_indexes = []
			built_indexes = []
			first_failure = 0
			ci = []
			
			total_builds = len(actual_results)
			sbs_builds = 0

			for i in range(len(actual_results)):

				#If first failure is already found, continue building until actual build pass is seen
				if first_failure == 1:
					ci.append(0)
					sbs_builds += 1

					if actual_results[i] == 1:
						#actual build pass is seen, switch to prediction
						first_failure = 0
					else:
						first_failure = 1
				else:
					#we're in prediction state, if predicted to skip, we skip
					if pred_results[i] == 1:
						ci.append(1)
					else:
						#if predicted to fail, we switch to determine state and set first_failure to True
						ci.append(0)
						sbs_builds += 1
						first_failure = 1-actual_results[i]


			total_builds = len(ci)
			actual_builds = ci.count(0)

			saved_builds = 100*ci.count(1)/total_builds
			reqd_builds = 100*ci.count(0)/total_builds
			
			for i in range(len(ci)):
				if ci[i] == 0:
					built_indexes.append(i)
				else:
					if actual_results[i] == 0:
						delay_indexes.append(i)

			
			'''from_value = 0
			delay = []
			for k in range(len(built_indexes)):
				for j in range(len(delay_indexes)):
					if delay_indexes[j] > from_value and delay_indexes[j] < built_indexes[k]:
						delay.append(built_indexes[k] - delay_indexes[j])
				from_value = built_indexes[k]

			final_index = len(ci)

			for j in range(len(delay_indexes)):
				if delay_indexes[j] > from_value and delay_indexes[j] < final_index:
					delay.append(final_index - delay_indexes[j])'''

			bp = 0
			mp = 0
			temp_delay = 0
			total_delay = 0

			while bp < len(built_indexes):
				while mp < len(delay_indexes) and delay_indexes[mp] < built_indexes[bp]:
					temp_delay = built_indexes[bp] - delay_indexes[mp]
					print("Difference: {}, Built_index = {} , Missed_index = {}".format(temp_delay, built_indexes[bp], delay_indexes[mp]))
					total_delay += temp_delay
					mp += 1
				bp += 1

			while mp < len(delay_indexes):
				temp_delay = total_builds - delay_indexes[mp]
				print("Difference: {}, Built_index = {} , Missed_index = {}".format(temp_delay, total_builds, delay_indexes[mp]))
				total_delay += temp_delay
				mp += 1


			delay = [total_delay]


			print('saved_builds for {} is {}'.format(project, saved_builds))
			print('delay for {} is {}\n\n'.format(project, sum(delay)))
			final_writer.writerow([project, 'sbs', b, reqd_builds, sum(delay)])


			durations = csv_file['tr_duration'].tolist()
			batch_size = b
			batch_builds = 0
			commit_num = 1
			build_time = 0

			for i in range(len(ci)):

				if commit_num == batch_size:
					res_writer.writerow([i+1, build_time, batch_builds])
					commit_num = 1
					build_time = 0
					batch_builds = 0
					continue

				if ci[i] == 0:
					batch_builds += 1
					build_time += durations[i]

				commit_num += 1

		 
			# file_name = 'metrics/' + project.split('.')[0] + '_real_metrics.csv'

			# csv_file = csv.reader(open(file_name, 'r'))

			# built_commits = []
			# build_time = 0
			# total_builds = 0
			# actual_builds = 0
			# commit_num = 1
			# flag = 0
			# batches = []
			# num = 0
			# b_id = 0
			




			# 	# if a build is predicted to fail, they will build it
			# 	if build[-2] == '0':
			# 		#add the build time
			# 		build_time += int(build[2])
			# 		actual_builds += 1
			# 		total_builds += 1
			# 		b_id = build[0]
			# 		flag = 1

			# 	#if prev build has failed, build until you see a true build pass
			# 	if flag == 1:
			# 		if build[-1] == '0':
			# 			if b_id != build[0]:
			# 				build_time += int(build[2])
			# 				actual_builds += 1
			# 				total_builds += 1				
			# 		if build[-1] == '1':
			# 			#this is the first build pass after failure
			# 			#go back to predicting
			# 			if b_id != build[0]:
			# 				build_time += int(build[2])
			# 				actual_builds += 1
			# 				total_builds += 1
			# 			flag = 0


			# 	'''#if a build passes,
			# 	if build[-2] == '1':
			# 		#check if this is the first build pass after failure
			# 		if (flag == 1):
			# 			flag = 0
			# 			build_time += int(build[2])
			# 			total_builds += 1'''

			# 	if commit_num == 4:
			# 		batches.append([int(build[1]), build_time, total_builds])
			# 		res_writer.writerow([int(build[1]), build_time, total_builds])
			# 		commit_num = 0
			# 		built_commits.append(build_time)
			# 		build_time = 0
			# 		total_builds = 0

			# 	commit_num += 1
			# #print(batches)
			# #print(total_builds)
			# print(actual_builds)
			# #print(len(csv_file))
			# #print('Total time taken for builds:')
			# #print(built_commits)

validation()
