#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from numpy import argmax
from numpy import sqrt
import math
from sklearn.model_selection import RepeatedKFold
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
from statistics import median
import pickle
import csv
import multiprocess
import warnings
warnings.filterwarnings("ignore")

# In[17]:


MAX_BATCH = [2, 4, 8, 16]
algorithm = ['BATCH4','BATCHBISECT', 'BATCHSTOP4']


# In[18]:


projects = ['rails.csv', 'jruby.csv', 'metasploit-framework.csv', 'cloudify.csv', 'vagrant.csv', 'rubinius.csv', 'open-build-service.csv', 'gradle.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'opal.csv', 'cloud_controller_ng.csv', 'puppet.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv', 'heroku.csv']
data_path = '../data/'
#confidence = list(range(2,21,1))
confidence = [10, 20]

# In[19]:


result_file = open('trial_ssr_debug.csv', 'w')
result_headers = ['project', 'algorithm', 'batch_size', 'confidence', 'project_reqd_builds', 'project_missed_builds', 'project_build_duration', 'project_saved_builds', 'project_delays', 'testall_size', 'batch_delays']
writer = csv.writer(result_file)
writer.writerow(result_headers)


# In[20]:


def get_train_test_data(filename):
    
    csv_file = csv.reader(open(filename, 'r'))
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
        src_churn.append(float(final_data[23][index]))
        file_churn.append(float(final_data[27][index]))
        test_churn.append(float(final_data[24][index]))
        team_size.append(float(final_data[14][index]))
        
        if final_data[42][index] == 'passed':
            build_result.append(1)
        else:
            build_result.append(0)

        argument.append([])

    for index in range(len(src_churn)):
        argument[index].append(src_churn[index])
        argument[index].append(team_size[index])
        argument[index].append(file_churn[index])
        argument[index].append(test_churn[index])
    
    return np.array(argument), np.array(build_result)


# In[21]:


def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0) 
    return Y_t


# In[22]:


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


# In[23]:


def pd_get_train_test_data(file_path):
    columns = ['gh_team_size', 'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_modified', 'tr_status']
    X = pd.read_csv(file_path, usecols = columns)
    X['tr_status'] = output_values(X['tr_status'])
    Y = X['tr_status']
    X.drop('tr_status', inplace=True, axis=1)
    #X = get_first_failures(X)
    #X.drop('tr_status', inplace=True, axis=1)
    #return X, Y
    return X.to_numpy(), np.array(Y)


# In[52]:


def sbs(project_name):
    global writer    
    #dataset already has first failures
    train_file = "../data/train_data/" + project_name + '_train.csv'
    num_feature = 4
    num_test = 0

    precision = []
    recall = []
    f1 = []
    build_save = []
    fitted_model = []
    

    X, y =  pd_get_train_test_data(train_file)
    pkl_file = '../data/data_pickles/' + project_name + '_indexes.pkl'
    with open(pkl_file, 'rb') as load_file:
        train_build_ids = pickle.load(load_file)
        test_build_ids = pickle.load(load_file)

    print("We are training from scratch and this is repeated K fold") 
    KF = RepeatedKFold(n_splits=10, n_repeats=10)
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
    
        precision0 = precision_score(Y_test, Y_result)
        recall0 = recall_score(Y_test, Y_result)
        f10 = f1_score(Y_test, Y_result)

        precision.append(precision0)
        recall.append(recall0)
        f1.append(f10)
        fitted_model.append(rf)

    best_f1 = max(f1)
    max_index = f1.index(best_f1)
    print('Best F1: {}, Precision:{}, Recall:{}'.format(best_f1, precision[max_index], recall[max_index]))
    best_fit_model = fitted_model[max_index]
    return best_fit_model



# In[53]:


def get_durations(project):
    csv_file = pd.read_csv(project)
    durations = csv_file['tr_duration'].tolist()
    return durations


# In[54]:


batch_total = 0
batch_durations = 0


# In[55]:


def batch_bisect(grouped_batch, actual_group_results, durations):
    global batch_total
    global batch_durations
 
    batch_total += 1
    batch_durations += max(durations)
    
    if len(actual_group_results) == 1:
        return
    
    if 0 in actual_group_results:
        half_batch = len(actual_group_results)//2
        batch_bisect(grouped_batch[:half_batch], actual_group_results[:half_batch], durations[:half_batch])
        batch_bisect(grouped_batch[half_batch:], actual_group_results[half_batch:], durations[half_batch:])


# In[56]:


def batch_stop_4(grouped_batch, actual_group_results, durations):
    global batch_total
    global batch_durations
    
    batch_total += 1
    batch_durations += max(durations)
    
    if len(actual_group_results) <= 4:
        if 0 in actual_group_results:
            batch_total += 4
            batch_durations += sum(durations)
        return
    
    if 0 in actual_group_results:
        half_batch = len(actual_group_results)//2
        batch_stop_4(grouped_batch[:half_batch], actual_group_results[:half_batch], durations[:half_batch])
        batch_stop_4(grouped_batch[half_batch:], actual_group_results[half_batch:], durations[half_batch:])


# In[57]:


def static_rule(p):
    global writer
    global batch_total
    global batch_durations
    
    p = p.split('.')[0]

    result_file_name = p + '_rkm_ssr.csv'

    #predictor, threshold = sbs(p)
    predictor = sbs(p)
    result_file = open(result_file_name, 'w')
    result_headers = ['project', 'algorithm', 'batch_size', 'confidence', 'project_reqd_builds', 'project_missed_builds', 'project_build_duration', 'project_saved_builds', 'project_delays', 'testall_size', 'batch_delays']
    writer = csv.writer(result_file)
    writer.writerow(result_headers)
    #model_file_name = 'dump_data/rq2_' + p + '_best_model.pkl'
    #model_file = open(model_file_name, 'rb')
    #predictor = pickle.load(model_file)
    #threshold = pickle.load(model_file)
    
    #get the test data
    
    test_file = "../data/test_data/" + p + '_test.csv'
    #X_test, Y_test = pd_get_train_test_data(test_file)
    Y_duration = get_durations(test_file)

    X_test, y_test =  pd_get_train_test_data(test_file)
        
    
    for alg in algorithm:
        for max_batch_size in MAX_BATCH:
                        
            if alg == 'BATCH4':
                if max_batch_size != 4:
                    continue
            
            if alg == 'BATCHSTOP4':
                if max_batch_size < 4:
                    continue
                    
            #print('Processing {} at batch size {} for {}'.format(alg, max_batch_size, p))


            Y_result = []
            grouped_batch = []
            actual_group_results = []
            group_duration = []
            num_feature = 4 
            length_of_test = len(y_test)

            project_reqd_builds = []
            project_missed_builds = []
            project_build_duration = []
            project_saved_builds = []
            project_delays = []
            project_bad_builds = []
            project_batch_delays = []

            #print('Processing {}'.format(p))
            for c in confidence:
                
                batch_delays = 0

                pass_streak = y_test[0]
                #total_builds = Y_test[0]
                total_builds = 0
                missed_builds = 0
                miss_indexes = []
                build_indexes = []
                delay_durations = []

                if pass_streak == 0:
                    total_duration = Y_duration[0]
                    saved_builds = 0
                else:
                    total_duration = 0
                    saved_builds = 1

                index = 1
                while index < len(X_test):
                    commit = X_test[index]
                    value = predictor.predict([commit])
                                        
                    if pass_streak < c :
                        
                        if value == 0:
                            #print(' c < threshold ; predicted to fail')
                            pass_streak = 1
                            while True:

                                grouped_batch = list(X_test[index : index+max_batch_size])
                                actual_group_results = list(y_test[index : index+max_batch_size])
                                group_duration = Y_duration[index : index+max_batch_size]

                                #print(grouped_batch)
                                #print(group_duration)
                                #print('Miss indexes: {}'.format(miss_indexes))

                                if alg == 'BATCH4':
                                    if len(actual_group_results) == max_batch_size:
                                        if len(miss_indexes) > 0:
                                            if miss_indexes[-1] < index:
                                                for l in range(len(miss_indexes)):
                                                    e = miss_indexes.pop()
                                                    delay_durations.append(index - e + 1)

                                        batch_delays += max_batch_size*(max_batch_size-1)/2
                                        total_builds += 1
                                        total_duration += max(group_duration)

                                        if 0 in actual_group_results:
                                            total_builds += max_batch_size
                                            total_duration += sum(group_duration)

                                        grouped_batch.clear()
                                        actual_group_results.clear()
                                        group_duration.clear()

                                elif alg == 'BATCHBISECT':
                                    if len(actual_group_results) == max_batch_size:
                                        if len(miss_indexes) > 0:
                                            if miss_indexes[-1] < index:
                                                for l in range(len(miss_indexes)):
                                                    e = miss_indexes.pop()
                                                    delay_durations.append(index - e + 1)

                                        batch_total = 0
                                        batch_durations = 0

                                        batch_bisect(grouped_batch, actual_group_results, group_duration)
                                        batch_delays += max_batch_size*(max_batch_size-1)/2
                                        total_builds += batch_total
                                        total_duration += batch_durations
                                        #print(actual_group_results)
                                        #print('Total builds = {}, batch_total = {}, batch_delays={}'.format(total_builds, batch_total, batch_delays))

                                        grouped_batch.clear()
                                        actual_group_results.clear()
                                elif alg == 'BATCHSTOP4':
                                    if len(actual_group_results) == max_batch_size:
                                        if len(miss_indexes) > 0:
                                            if miss_indexes[-1] < index:
                                                for l in range(len(miss_indexes)):
                                                    e = miss_indexes.pop()
                                                    delay_durations.append(index - e + 1)

                                        batch_total = 0
                                        batch_durations = 0

                                        batch_stop_4(grouped_batch, actual_group_results, group_duration)

                                        batch_delays += max_batch_size*(max_batch_size-1)/2
                                        total_builds += batch_total
                                        total_duration += batch_durations

                                        #print(actual_group_results)
                                        #print('Total builds = {}, batch_total = {}, batch_delays={}'.format(total_builds, batch_total, batch_delays))

                                        grouped_batch.clear()
                                        actual_group_results.clear()
                                        group_duration.clear()
                                elif alg == 'NOBATCH':
                                    if len(grouped_batch) == max_batch_size:
                                        if len(miss_indexes) > 0:
                                            if miss_indexes[-1] < index:
                                                for l in range(len(miss_indexes)):
                                                    e = miss_indexes.pop()
                                                    delay_durations.append(index - e + 1)

                                        batch_delays += max_batch_size*(max_batch_size-1)/2
                                        total_builds += 1
                                        total_duration += max(group_duration)

                                        grouped_batch.clear()
                                        actual_group_results.clear()
                                        group_duration.clear()


                                if 0 in actual_group_results:
                                    index += max_batch_size
                                else:
                                    break
                            pass_streak = 1
                            index += max_batch_size
                                
                        else:
                            #print(' c < threshold ; predicted to pass')
                            pass_streak += 1
                            saved_builds += 1
                            if y_test[index] == 0:
                                missed_builds += 1
                                miss_indexes.append(index)

                            #seeing only one build
                            index += 1

                    else:
                        #print('c > threshold')
                        while True:

                            grouped_batch = list(X_test[index : index+max_batch_size])
                            actual_group_results = list(y_test[index : index+max_batch_size])
                            group_duration = Y_duration[index : index+max_batch_size]


                            if alg == 'BATCH4':
                                if len(actual_group_results) == max_batch_size:
                                    if len(miss_indexes) > 0:
                                        if miss_indexes[-1] < index:
                                            for l in range(len(miss_indexes)):
                                                e = miss_indexes.pop()
                                                delay_durations.append(index - e + 1)
                                    
                                    batch_delays += max_batch_size*(max_batch_size-1)/2
                                    total_builds += 1
                                    total_duration += max(group_duration)

                                    if 0 in actual_group_results:
                                        total_builds += max_batch_size
                                        total_duration += sum(group_duration)

                                    grouped_batch.clear()
                                    actual_group_results.clear()
                                    group_duration.clear()

                            elif alg == 'BATCHBISECT':
                                if len(actual_group_results) == max_batch_size:
                                    if len(miss_indexes) > 0:
                                        if miss_indexes[-1] < index:
                                            for l in range(len(miss_indexes)):
                                                e = miss_indexes.pop()
                                                delay_durations.append(index - e + 1)

                                    batch_total = 0
                                    batch_durations = 0

                                    batch_bisect(grouped_batch, actual_group_results, group_duration)

                                    batch_delays += max_batch_size*(max_batch_size-1)/2
                                    total_builds += batch_total
                                    total_duration += batch_durations

                                    #print(actual_group_results)
                                    #print('Total builds = {}, batch_total = {}, batch_delays={}'.format(total_builds, batch_total, batch_delays))
                                    #print('Total builds = {}, batch_delays={}'.format(total_builds, batch_delays))

                                    grouped_batch.clear()
                                    actual_group_results.clear()
                                    group_duration.clear()
                            elif alg == 'BATCHSTOP4':
                                if len(actual_group_results) == max_batch_size:
                                    if len(miss_indexes) > 0:
                                        if miss_indexes[-1] < index:
                                            for l in range(len(miss_indexes)):
                                                e = miss_indexes.pop()
                                                delay_durations.append(index - e + 1)

                                    batch_total = 0
                                    batch_durations = 0

                                    batch_stop_4(grouped_batch, actual_group_results, group_duration)

                                    batch_delays += max_batch_size*(max_batch_size-1)/2
                                    total_builds += batch_total
                                    total_duration += batch_durations

                                    #print(actual_group_results)
                                    #print('Total builds = {}, batch_total = {}, batch_delays={}'.format(total_builds, batch_total, batch_delays))

                                    grouped_batch.clear()
                                    actual_group_results.clear()
                                    group_duration.clear()
                            elif alg == 'NOBATCH':
                                if len(grouped_batch) == max_batch_size:
                                    if len(miss_indexes) > 0:
                                        if miss_indexes[-1] < index:
                                            for l in range(len(miss_indexes)):
                                                e = miss_indexes.pop()
                                                delay_durations.append(index - e + 1)
                                    
                                    batch_delays += max_batch_size*(max_batch_size-1)/2
                                    total_builds += 1
                                    total_duration += max(group_duration)

                                    grouped_batch.clear()
                                    actual_group_results.clear()
                                    group_duration.clear()

                            pass_streak = 1 
                            if 0 in actual_group_results:
                                index += max_batch_size
                            else:
                                break
                        index += max_batch_size
                    

                #print('\tFor confidence {}:'.format(c))
                #print('\t\tTotal builds needed : {}'.format(total_builds))
                #print('\t\tTotal number of missed builds : {}'.format(missed_builds))
                #print('\t\tTotal number of saved builds : {}'.format(saved_builds))
                #print('\t\tTotal duration of builds : {}'.format(total_duration))
                #print('\t\tTotal delays: {}'.format(delay_durations))

                project_reqd_builds.append(total_builds)
                project_missed_builds.append(missed_builds)
                project_build_duration.append(total_duration)
                project_saved_builds.append(saved_builds)
                project_delays.append(delay_durations)
                project_batch_delays.append(batch_delays)

            #print(p)
            #print(project_reqd_builds)
            #print(project_missed_builds)
            #print(project_build_duration)
            #print(project_saved_builds)
            #print(project_delays)
            #print(project_batch_delays)
            
            for i in range(len(confidence)):
                print([p, alg, max_batch_size, confidence[i], 100*project_reqd_builds[i]/length_of_test, 100*project_missed_builds[i]/length_of_test, project_build_duration[i], 100*project_saved_builds[i]/length_of_test, project_delays[i], length_of_test, project_batch_delays[i]])
                writer.writerow([p, alg, max_batch_size, confidence[i], 100*project_reqd_builds[i]/length_of_test, 100*project_missed_builds[i]/length_of_test, project_build_duration[i], 100*project_saved_builds[i]/length_of_test, project_delays[i], length_of_test, project_batch_delays[i]])


# In[ ]:


if __name__ == '__main__':
    with multiprocess.Pool(5) as p:
        p.map(static_rule, projects[9:])




