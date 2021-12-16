import pandas as pd

from utils import project_list

cols = [
    'job_status',
    'build_status',
    'tr_log_status'
]


def get_failure_rate(gh_project_name):
    print(gh_project_name)

    dataset = pd.read_csv('../../data/{}.csv'.format(gh_project_name), usecols=cols)

    # unknown_jobs = dataset[dataset['tr_log_status'] == 'unknown']
    # print('Number of unkown jobs: ', len(unknown_jobs),
    #       '({:.2f}%)'.format(len(unknown_jobs) / len(dataset) * 100))

    dataset = dataset[dataset['job_status'] != 'canceled']
    print('Total number of jobs: {:,}'.format(len(dataset)))

    for job_status in ['passed', 'failed', 'errored']:
        get_job_rate(dataset, job_status)

        for build_status in ['passed', 'failed', 'errored']:
            get_job_rate_filtered_by_build_rate(dataset, job_status, build_status)

        print()

    print()


def get_job_rate_filtered_by_build_rate(dataset, job_status, build_status):
    job_list = dataset[dataset['job_status'] == job_status]
    build_list = job_list[job_list['build_status'] == build_status]

    print('Number of {} jobs with {} build: {:,}'.format(job_status, build_status, len(build_list)),
          '({:.2f}%)'.format(len(build_list) / len(dataset) * 100))


def get_job_rate(dataset, job_status):
    job_list = dataset[dataset['job_status'] == job_status]
    print('Number of {} jobs: {:,}'.format(job_status, len(job_list)),
          '({:.2f}%)'.format(len(job_list) / len(dataset) * 100))


for project in project_list:
    get_failure_rate(project)
