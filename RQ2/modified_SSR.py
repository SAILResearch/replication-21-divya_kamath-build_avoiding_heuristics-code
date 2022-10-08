#!/usr/bin/env python
# coding: utf-8

# In[16]:


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
from statistics import median
import pickle
import csv
import multiprocess


# In[17]:


MAX_BATCH = [2, 4, 8, 16]
algorithm = ['BATCH4', 'BATCHBISECT', 'BATCHSTOP4']


# In[18]:


projects = ['rails.csv', 'jruby.csv', 'metasploit-framework.csv', 'cloudify.csv', 'vagrant.csv', 'rubinius.csv', 'open-build-service.csv', 'gradle.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'opal.csv', 'cloud_controller_ng.csv', 'puppet.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv', 'heroku.csv']
data_path = '../data/'
confidence = list(range(2,21,1))


# In[19]:


result_file = open('ssr_algorithms.csv', 'w')
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
    columns = ['tr_build_id', 'gh_team_size', 'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_modified', 'tr_status']
    X = pd.read_csv(file_path, usecols = columns)
    X['tr_status'] = output_values(X['tr_status'])
    Y = X['tr_status']
    #X = get_first_failures(X)
    #X.drop('tr_status', inplace=True, axis=1)
    
    return X, Y


# In[52]:


def sbs(project):
    
    #dataset already has first failures
    train_file = "../data/train_data/" + project + '_train.csv'
    num_feature = 4
    
    X_train, Y_train = pd_get_train_test_data(train_file)
    
    #X_train = X_train.reshape((int(len(X_train)), num_feature+1))
    print(X_train)
    print(Y_train)
    
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    
    rf = RandomForestClassifier()
    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 0)
    
    best_n_estimators = []
    best_max_depth = []
    
    best_f1 = 0
    best_f1_sample = 0
    best_f1_sample_result = 0
    best_f1_estimator = 0
    best_thresholds = []
    
    for i in range(1):
        print('Bootstrapping {} for {}'.format(i, project))
        
        while True:
            print('Here for {} {}'.format(i, project))
            sample_train = resample(X_train, replace=True, n_samples=len(X_train))
            sample_train_result = sample_train['tr_status']

            build_ids = sample_train['tr_build_id'].tolist()
            sample_test = X_train [~X_train['tr_build_id'].isin(build_ids)] 
            sample_test_result = sample_test['tr_status']

            if len(sample_test_result) != 0:
                break
        
        sample_train.drop('tr_status', inplace=True, axis=1)
        sample_train.drop('tr_build_id', inplace=True, axis=1)
        sample_test.drop('tr_status', inplace=True, axis=1)
        sample_test.drop('tr_build_id', inplace=True, axis=1)
        
        print('Training {} for {}'.format(i, project))
        grid_search.fit(sample_train, sample_train_result)
        sample_pred_vals = grid_search.predict_proba(sample_test)

        pred_vals = sample_pred_vals[:, 1]
        fpr, tpr, t = roc_curve(sample_test_result, pred_vals)
        gmeans = sqrt(tpr * (1-fpr))
        ix = argmax(gmeans)
        bt = t[ix]
        best_thresholds.append(bt)
        
        final_pred_result = []
        #threshold setting
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
            print(best_f1_sample)
            
        best_n_estimators.append(grid_search.best_params_['n_estimators'])
        best_max_depth.append(grid_search.best_params_['max_depth'])
        
    #completed with bootstrapping 
    threshold = median(best_thresholds)
    n_estimator = median(best_n_estimators)
    max_depth = median(best_max_depth)

    #retrain to get the best model
    forest = RandomForestClassifier(n_estimators=int(n_estimator), max_depth=int(max_depth))
    forest.fit(best_f1_sample, best_f1_sample_result)

    file_name = 'dump_data/rq2_' + p_name + '_best_model.pkl'
    dump_file = open(file_name, 'wb')
    pickle.dump(forest, dump_file)
    pickle.dump(threshold, dump_file)
    pickle.dump(n_estimator, dump_file)
    pickle.dump(max_depth, dump_file)
                
    #predictor = rf.fit(X_train, Y_train)
    return forest, threshold


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
    
    if len(grouped_batch) == 1:
        return
    
    if 0 in actual_group_results:
        half_batch = len(grouped_batch)//2
        batch_bisect(grouped_batch[:half_batch], actual_group_results[:half_batch], durations[:half_batch])
        batch_bisect(grouped_batch[half_batch:], actual_group_results[half_batch:], durations[half_batch:])


# In[56]:


def batch_stop_4(grouped_batch, actual_group_results, durations):
    global batch_total
    global batch_durations
    
    batch_total += 1
    batch_durations += max(durations)
    
    if len(grouped_batch) <= 4:
        batch_total += 4
        batch_durations += sum(durations)
        return
    
    if 0 in actual_group_results:
        half_batch = len(grouped_batch)//2
        batch_stop_4(grouped_batch[:half_batch], actual_group_results[:half_batch], durations[:half_batch])
        batch_stop_4(grouped_batch[half_batch:], actual_group_results[half_batch:], durations[half_batch:])


# In[57]:


def static_rule(p):
    
    global batch_total
    global batch_durations
    
    p = p.split('.')[0]

    predictor, threshold = sbs(p)

    #get the test data
    
    test_file = "../data/test_data/" + p + '_test.csv'
    X_test, Y_test = pd_get_train_test_data(test_file)
    Y_duration = get_durations(test_file)
    
    X_test.drop('tr_build_id', inplace=True, axis=1)
    X_test.drop('tr_status', inplace=True, axis=1)
    
    
    for alg in algorithm:
        for max_batch_size in MAX_BATCH:
                        
            if alg == 'BATCH4':
                if max_batch_size != 4:
                    continue
            
            if alg == 'BATCHSTOP4':
                if max_batch_size < 4:
                    continue
                    
            


            Y_result = []
            grouped_batch = []
            actual_group_results = []
            group_duration = []
            num_feature = 4 
            length_of_test = len(Y_test)

            project_reqd_builds = []
            project_missed_builds = []
            project_build_duration = []
            project_saved_builds = []
            project_delays = []
            project_bad_builds = []
            project_batch_delays = []

            print('Processing {}'.format(p))
            for c in confidence:
                
                batch_delays = 0

                pass_streak = Y_test[0]
                total_builds = Y_test[0]
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
                    commit = X_test.iloc[index]
                    print(commit)
                    predict = predictor.predict_proba([commit])
                    #we're setting a confidence of 'c' builds on SBS, if more than 'c' passes have been suggested in a row, we don't want to trust sbs
                    
                    if predict[0][1] > threshold:
                        value = 1
                    else:
                        value = 0
                    
                    
                    if pass_streak < c :
                        
                        if value == 0:
                            
                            while True:

                                grouped_batch = list(X_test[index : index+max_batch_size])
                                actual_group_results = list(Y_test[index : index+max_batch_size])
                                group_duration = Y_duration[index : index+max_batch_size]

                #                 if len(grouped_batch) < max_batch_size:
                #                     grouped_batch.append(index)
                #                     actual_group_results.append(Y_test[index])
                #                     group_duration.append(Y_duration[index])
                                if alg == 'BATCH4':
                                    if len(grouped_batch) == max_batch_size:
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
                                    if len(grouped_batch) == max_batch_size:
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

                                        grouped_batch.clear()
                                        actual_group_results.clear()
                                elif alg == 'BATCHSTOP4':
                                    if len(grouped_batch) == max_batch_size:
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

                                index += max_batch_size
                                pass_streak = 1                            

                                if 0 in actual_group_results:
                                    index += max_batch_size
                                else:
                                    break
                                
#                             total_builds += 1
#                             total_duration += Y_duration[index]
#                             if len(miss_indexes) > 0:
#                                 if miss_indexes[-1] < index:
#                                     for l in range(len(miss_indexes)):
#                                         e = miss_indexes.pop()
#                                         delay_durations.append(index - e + 1)
                        else:
                            pass_streak += 1
                            saved_builds += 1
                            if Y_test[index] == 0:
                                missed_builds += 1
                                miss_indexes.append(index)

                            #seeing only one build
                            index += 1

                    else:
                        
                        while True:

                            grouped_batch = list(X_test[index : index+max_batch_size])
                            actual_group_results = list(Y_test[index : index+max_batch_size])
                            group_duration = Y_duration[index : index+max_batch_size]

            #                 if len(grouped_batch) < max_batch_size:
            #                     grouped_batch.append(index)
            #                     actual_group_results.append(Y_test[index])
            #                     group_duration.append(Y_duration[index])

                            if alg == 'BATCH4':
                                if len(grouped_batch) == max_batch_size:
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
                                if len(grouped_batch) == max_batch_size:
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

                                    grouped_batch.clear()
                                    actual_group_results.clear()
                                    group_duration.clear()
                            elif alg == 'BATCHSTOP4':
                                if len(grouped_batch) == max_batch_size:
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

                            index += max_batch_size
                            pass_streak = 1
                            
                            if 0 in actual_group_results:
                                index += max_batch_size
                            else:
                                break
                    

        #         print('\tFor confidence {}:'.format(c))
        #         print('\t\tTotal builds needed : {}'.format(total_builds))
        #         print('\t\tTotal number of missed builds : {}'.format(missed_builds))
        #         print('\t\tTotal number of saved builds : {}'.format(saved_builds))
        #         print('\t\tTotal duration of builds : {}'.format(total_duration))
        #         print('\t\tTotal delays: {}'.format(delay_durations))

                project_reqd_builds.append(total_builds)
                project_missed_builds.append(missed_builds)
                project_build_duration.append(total_duration)
                project_saved_builds.append(saved_builds)
                project_delays.append(delay_durations)
                project_batch_delays.append(batch_delays)

            print(p)
            print(project_reqd_builds)
            print(project_missed_builds)
            print(project_build_duration)
            print(project_saved_builds)
            print(project_delays)
            print(project_batch_delays)
            
            for i in range(len(confidence)):
                print([p, alg, max_batch_size, confidence[i], 100*project_reqd_builds[i]/length_of_test, 100*project_missed_builds[i]/length_of_test, project_build_duration[i], 100*project_saved_builds[i]/length_of_test, project_delays[i], length_of_test, project_batch_delays[i]])
                writer.writerow([p, alg, max_batch_size, confidence[i], 100*project_reqd_builds[i]/length_of_test, 100*project_missed_builds[i]/length_of_test, project_build_duration[i], 100*project_saved_builds[i]/length_of_test, project_delays[i], length_of_test, project_batch_delays[i]])


# In[ ]:


if __name__ == '__main__':
    with multiprocess.Pool(5) as p:
        p.map(static_rule, project_list[9:])




