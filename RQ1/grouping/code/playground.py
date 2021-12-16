import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time


from utils import project_list, cols


def create_project_list():
    df = pd.read_csv('../data/travistorrent_8_2_2017.csv')
    project_list = df['gh_project_name'].unique()

    col_names = ['project_name', 'failure_rate', 'number_of_builds']
    projects_df = pd.DataFrame(columns=col_names)

    for prj in project_list:
        df1 = df[df['gh_project_name'] == prj]
        df1 = df1[df1['tr_status'] != 'canceled']
        df1 = df1.drop_duplicates(subset=['tr_build_id'])

        failure_rate = float(
            '{0:.2f}'.format(
                df1[df1['tr_status'] != 'passed']['tr_build_id'].count() / df1['tr_build_id'].count() * 100))

        number_of_builds = df1['tr_build_id'].count()

        row = [prj, failure_rate, number_of_builds]
        print(row)
        projects_df.loc[len(projects_df)] = row

    projects_df.to_csv('../data/projects.csv')


def extract_project(prj):
    df = pd.read_csv('../data/travistorrent_8_2_2017.csv')
    df1 = df[df['gh_project_name'] == prj]
    df1 = df1.drop_duplicates(subset=['tr_build_id'])
    df1.to_csv('../data/{}.csv'.format(prj.replace('/', '--')))


def run_extract():
    extract_project('opal/opal')
    extract_project('rspec/rspec-core')
    extract_project('FenixEdu/fenixedu-academic')


def get_pr_ratio():
    for prj in project_list:
        df = pd.read_csv('../data/{}.csv'.format(prj), usecols=['gh_is_pr'])
        print(prj, '{0:.2f}%'.format(df[df['gh_is_pr'] == True]['gh_is_pr'].count() / df['gh_is_pr'].count() * 100))


def get_projects_stats():
    for prj in project_list:
        df = pd.read_csv('../data/{}.csv'.format(prj['name']), usecols=cols)
        print(prj['name'],
              '{0:.2f}%'.format(df[df['tr_status'] == 'canceled']['tr_status'].count() * 100 / df['tr_status'].count()))


def get_projects_years():
    for prj in project_list:
        df = pd.read_csv('../data/{}.csv'.format(prj['name']), usecols=['gh_build_started_at'])
        print(prj['name'])
        print(df['gh_build_started_at'].min())
        print(df['gh_build_started_at'].max())
        print()


def plot_build_date_histogram():
    for prj in project_list:
        df = pd.read_csv('../data/{}.csv'.format(prj['name']), usecols=['gh_build_started_at'])
        df['gh_build_started_at'] = pd.to_datetime(df['gh_build_started_at'])
        df['gh_build_started_at'] = df['gh_build_started_at'].dt.date

        # print(len(set(df['gh_build_started_at'])))
        # print(df['gh_build_started_at'].min())
        # print(df['gh_build_started_at'].max())

        fig, ax = plt.subplots(1, 1)
        ax.hist(df['gh_build_started_at'], bins=len(df['gh_build_started_at']), histtype='bar')

        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))

        plt.xlabel('Date')
        plt.ylabel('Build frequency')
        plt.title('{} Build Date Histogram'.format(prj['name']))
        plt.show()


def plot_build_duration_histogram():
    for prj in project_list:
        # print(prj['name'], 'Build Duration')
        df = pd.read_csv('../data/{}.csv'.format(prj['name']), usecols=['tr_duration', 'tr_status'])

        df = df[df['tr_status'] != 'canceled']

        # print('Min: ', int(df['tr_duration'].min() / 60), 'm')
        # print('Max: ', int(df['tr_duration'].max() / 60), 'm')
        # print('Mean: ', int(df['tr_duration'].mean() / 60), 'm')
        # print('Median: ', int(df['tr_duration'].median() / 60), 'm')
        # print('STD: ', int(df['tr_duration'].std() / 60), 'm')

        # print('Max: ', int(df['tr_duration'].max()), 's')
        # print('Mean: ', int(df['tr_duration'].mean()), 's')
        # print('Median: ', int(df['tr_duration'].median()), 's')
        # print('STD: ', int(df['tr_duration'].std()), 's')

        print(time.strftime('%H:%M:%S', time.gmtime(int(df['tr_duration'].mean()))))

        # print()

        # fig, ax = plt.subplots(1, 1)
        # ax.hist(df['tr_duration'], bins=len(df['tr_duration']), histtype='bar')
        #
        # locator = mdates.AutoDateLocator()
        # ax.xaxis.set_major_locator(locator)
        # # ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        # ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
        #
        # plt.xlabel('Date')
        # plt.ylabel('Build frequency')
        # plt.title('{} Build Date Histogram'.format(project))
        # plt.show()

        # break


def get_projects_num_of_builds():
    for prj in project_list:
        df = pd.read_csv('../data/{}.csv'.format(prj['name']), usecols=['tr_status', 'tr_build_id'])
        df = df[df['tr_status'] != 'canceled']
        # print(prj['name'])
        print(df['tr_build_id'].count())
        # print()


def get_projects_num_of_failures():
    for prj in project_list:
        df = pd.read_csv('../data/{}.csv'.format(prj['name']), usecols=['tr_status', 'tr_build_id'])
        df = df[df['tr_status'] != 'canceled']
        df = df[df['tr_status'] != 'passed']
        # print(prj['name'])
        print(df['tr_build_id'].count())
        # print()


def get_failure_distribution_avg():
    d = dict()

    lav = list()

    for prj in project_list:
        df = pd.read_csv('../data/{}.csv'.format(prj['name']), usecols=['tr_status', 'tr_build_id'])
        df = df[df['tr_status'] != 'canceled']

        av = [prj['name']]

        i = 0
        for item in list(df['tr_status']):
            if item != 'passed':
                av.append(i)
                i = 0
            else:
                i += 1

        lav.append(av)
        # d[prj['name']] = ','.join(str(item) for item in av)


        # print(statistics.mean(av))
        # print(df['tr_build_id'].count())
        # break

    # for item in lav:
    #     print(item)

    s = ''

    for i in range(20000):
        l = ''
        c = True
        for item in lav:
            if len(item) > i:
                c = False
                l += str(item[i])

            l += ','

        if c:
            break

        l = l[:-1] + '\n'
        s += l

    with open('num_of_builds_until_failure.csv', 'w') as f:
        f.write(s)

    # print(s)

    # p = pd.DataFrame(d.items(), columns=['Project', 'Num of builds until failre'])
    # p.to_csv('num_of_builds_until_failure.csv')

    # def tgrp(df):
    #     df = df.drop('Project', axis=1)
    #     return df.reset_index(drop=True).T
    #
    # df2 = p.groupby('Project').apply(tgrp).unstack()

    # df2 = p.set_index(['Project', 'Num of builds until failre']).unstack()
    # print(df2)

    # print(p)


def get_last_failure_distance():
    lav = list()

    for prj in project_list:
        df = pd.read_csv('../data/{}.csv'.format(prj['name']), usecols=['tr_status', 'tr_build_id'])
        df = df[df['tr_status'] != 'canceled']

        av = [prj['name']]

        result = []
        d = 0

        for item in df['tr_status']:
            result.append(d)

            if item != 'passed':
                d = 0
            else:
                d += 1

        av.extend(result)

        # lav.
