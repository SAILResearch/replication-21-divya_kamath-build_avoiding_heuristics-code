import csv
from csv import reader
import pandas as pd

datasets = []
with open('dataset_list') as file:
	content = file.read()
	datasets = content.split('\n')

for file in datasets:
	path = 'data/' + file
	data_file = pd.read_csv(path)
	length = len(data_file['tr_build_id'])

	proj_file = open(path, 'r')
	split_index = length*0.70
	
	headers = ['tr_build_id', 'gh_project_name', 'gh_is_pr', 'gh_pr_created_at', 'gh_pull_req_num',
				'gh_lang', 'git_merged_with', 'git_branch', 'gh_num_commits_in_push',
				'gh_commits_in_push',	'git_prev_commit_resolution_status', 'git_prev_built_commit',
				'tr_prev_build', 'gh_first_commit_created_at', 'gh_team_size', 'git_all_built_commits',
				'git_num_all_built_commits', 'git_trigger_commit', 'tr_virtual_merged_into',
				'tr_original_commit',	'gh_num_issue_comments', 'gh_num_commit_comments',
				'gh_num_pr_comments',	'git_diff_src_churn',	'git_diff_test_churn', 'gh_diff_files_added',
				'gh_diff_files_deleted', 'gh_diff_files_modified', 'gh_diff_tests_added',	
				'gh_diff_tests_deleted', 'gh_diff_src_files', 'gh_diff_doc_files', 
				'gh_diff_other_files', 'gh_num_commits_on_files_touched', 'gh_sloc', 
				'gh_test_lines_per_kloc',	'gh_test_cases_per_kloc',	'gh_asserts_cases_per_kloc',
				'gh_by_core_team_member',	'gh_description_complexity', 'gh_pushed_at',
				'gh_build_started_at', 'tr_status', 'tr_duration',	'tr_jobs', 'tr_build_number', 
				'tr_job_id', 'tr_log_lan', 'tr_log_status', 'tr_log_setup_time', 
				'tr_log_analyzer', 'tr_log_frameworks',	'tr_log_bool_tests_ran',
				'tr_log_bool_tests_failed', 'tr_log_num_tests_ok', 'tr_log_num_tests_failed',
				'tr_log_num_tests_run', 'tr_log_num_tests_skipped',	'tr_log_tests_failed', 
				'tr_log_testduration', 'tr_log_buildduration', 'build_successful']
	
	test_data_name = file.split('.')[0] + '_test.csv'
	train_data_name = file.split('.')[0] + '_train.csv'

	testfile = open(test_data_name, 'w')
	testwriter = csv.writer(testfile)
	testwriter.writerow(headers)

	trainfile = open(train_data_name, 'w')
	trainwriter = csv.writer(trainfile)
	trainwriter.writerow(headers)


	index = 0
	csv_reader = reader(proj_file)
	for line in csv_reader:
		#print(line)
		if index == 0:
			index = 1
			print('continued')
			continue
		elif index < split_index:
			print('here')
			trainwriter.writerow(line)
		else:
			print('yahan')
			testwriter.writerow(line)
		index += 1
		#print(line)
