from learning import IncrementalLearningModel

line_styles = ['-', '-.', ':', '--']
color_list = ['#33a02c', '#6a3d9a', '#e41a1c']

cols = [
    'gh_is_pr',
    # 'gh_lang',
    'gh_num_commits_in_push',
    'git_prev_commit_resolution_status',
    # 'gh_team_size',
    'git_num_all_built_commits',
    # 'gh_num_issue_comments',
    'gh_num_commit_comments',
    # 'gh_num_pr_comments',
    'git_diff_src_churn',
    'git_diff_test_churn',
    'gh_diff_files_added',
    'gh_diff_files_deleted',
    'gh_diff_files_modified',
    'gh_diff_tests_added',
    'gh_diff_tests_deleted',
    'gh_diff_src_files',
    'gh_diff_doc_files',
    'gh_diff_other_files',
    'gh_num_commits_on_files_touched',
    'gh_sloc',
    'gh_asserts_cases_per_kloc',
    'gh_by_core_team_member',
    # 'gh_description_complexity',
    'tr_status'
]

cols_with_date = cols[:]
cols_with_date.append('gh_build_started_at')

project_list = [
    {
        'name': 'heroku',
        'params': {'n_estimators': 10, 'max_depth': 10}
    }
]
''',
    {
        'name': 'opal',
        'params': {'n_estimators': 10, 'max_depth': 10}
    },
    {
        'name': 'vagrant',
        'params': {'n_estimators': 10, 'max_depth': 10}
    },
    {
        'name': 'metasploit-framework',
        'params': {'n_estimators': 10, 'max_depth': 10}
    },
    {
        'name': 'gradle',
        'params': {'n_estimators': 10, 'max_depth': 10}
    },
    {
        'name': 'jruby',
        'params': {'n_estimators': 10, 'max_depth': 10}
    },
    {
        'name': 'orbeon-forms',
        'params': {'n_estimators': 10, 'max_depth': 10}
    }
    # {
    #     'name': 'FenixEdu--fenixedu-academic',
    #     'params': {'n_estimators': 10, 'criterion': 'entropy'},
    #     'batch4': 943,
    #     'batch_bisect': 650
    # },
]'''

# er_project_list = [
#     'er-prj-1',
#     'er-prj-2',
#     'er-prj-3'
# ]

er_project_list = [
    'bb',
    'ebb',
    'racoam'
]

num_of_learning_days_list = [30, 60, 90, 180, 360, 720]


def get_y_test(prj):
    learning_model = IncrementalLearningModel(prj['name'], 'RF', 30, 1)
    y_proba, y_test = learning_model.get_predicted_data()
    y_test = [True if item == 'passed' else False for item in y_test]
    return y_test


def get_y_test_y_proba(prj):
    learning_model = IncrementalLearningModel(prj['name'], 'RF', 30, 1)
    y_proba, y_test = learning_model.get_predicted_data()
    y_test = [True if item == 'passed' else False for item in y_test]
    return y_test, y_proba
