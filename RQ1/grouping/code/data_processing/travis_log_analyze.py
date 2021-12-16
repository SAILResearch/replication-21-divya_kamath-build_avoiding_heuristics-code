import os
import re
from json import JSONDecodeError
from threading import Thread

import requests
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")

application = get_wsgi_application()

from old.main_app.models import Build


def get_failed_tests(job):
    for i in range(10):
        try:
            response = requests.get('https://api.travis-ci.org/v3/job/{}'.format(job.tr_job_id))
            break
        except Exception as e:
            print(e)
            break

    try:
        job_status = response.json()['state']
        if job_status == 'passed':
            return job_status, None, None
    except JSONDecodeError:
        job_status = 'invalid json'

    print(job_status)

    api_response = response.content.decode()
    with open('../data/raw_data/{}/{}.json'.format(job.gh_project_name.replace('/', '--'), job.tr_job_id), 'w') as f:
        f.write(api_response)

    response = requests.get('https://api.travis-ci.org/v3/job/{}/log.txt'.format(job.tr_job_id))

    if response.status_code == 404:
        return job_status, 'No log', []

    logs = response.content.decode()
    with open('../data/raw_data/{}/{}.txt'.format(job.gh_project_name.replace('/', '--'), job.tr_job_id), 'w') as f:
        f.write(logs)

    timeout = re.findall(
        r'No output has been received in the last 10 minutes |Execution of ci\/travis\.rb timed out and was terminated',
        logs)
    if timeout:
        return job_status, 'Timeout', []

    git_error = re.findall(
        r'remote: aborting due to possible repository corruption on the remote side\.|fatal: The remote end hung up unexpectedly',
        logs)
    if git_error:
        return job_status, 'Git Error', []

    # activerecord_error = re.findall('rake aborted!\r\nErrors in activerecord', logs)
    # if activerecord_error:
    #     return 'Activerecord error'

    failures = re.findall(r'\S*#test[^:\s]*', logs)
    if failures:
        print('#test finder')
        return job_status, None, failures

    error_list = re.findall(r'\d+\) Error:\r\ntest_.*:', logs)
    error_list = [item.split('\n')[1][:-1] for item in error_list]
    if error_list:
        failures += error_list
        print('Errors: ')

    failure_list = re.findall(r'\d+\) Failure:\r\ntest_.*\[', logs)
    failure_list = [item.split('\n')[1][:-1] for item in failure_list]

    if failure_list:
        failures += failure_list
        print('Failure: ')

    return job_status, None, failures


def crawl(gh_project_name):
    failed_builds = Build.objects.filter(
        gh_project_name=gh_project_name,
        # fail_type__isnull=True,
        # failed_tests__isnull=True
    ).exclude(
        tr_log_status='ok'
    ).exclude(
        job_status='passed'
    ).order_by(
        'tr_job_id'
    )

    for job in failed_builds:

        print(job.tr_job_id)
        print(job.tr_log_status)

        job_status, error_type, failure_tests, = get_failed_tests(job)
        job.job_status = job_status

        if job_status == 'passed':
            print(job_status)
            job.save(update_fields=['job_status'])

        elif error_type:
            print(error_type)
            job.fail_type = error_type
            job.save(update_fields=['job_status', 'fail_type'])

        elif failure_tests:
            failure_tests = set(failure_tests)
            print(failure_tests)
            job.failed_tests = ';'.join(failure_tests)
            job.save(update_fields=['job_status', 'failed_tests'])

        print()


for p in [
    # 'rails/rails',
    'jruby/jruby',
    'opf/openproject',
    'jruby/activerecord-jdbc-adapter',
    'rollbar/rollbar-gem',
    'rspec/rspec-rails',
    'slim-template/slim',
    'ruby/ruby',
    'diaspora/diaspora',
    'SonarSource/sonarqube'
]:
    t = Thread(target=crawl, args=(p,))
    t.start()
