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
import warnings
warnings.filterwarnings("ignore")

# In[17]:


MAX_BATCH = [1, 2, 4, 8, 16]
algorithm = ['BATCHBISECT', 'BATCH4', 'BATCHSTOP4']


# In[18]:


projects = ['vagrant.csv', 'puppet.csv', 'graylog2-server.csv', 'heroku.csv', 'rails.csv', 'gradle.csv', 'jruby.csv', 'metasploit-framework.csv', 'cloudify.csv', 'rubinius.csv', 'open-build-service.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'opal.csv', 'cloud_controller_ng.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv']
data_path = '../data/'
confidence = list(range(2,21,1))
#confidence = list(range(2,20))
#confidence = [10, 20]
# In[19]:


result_file = open('all_rq2_results.csv', 'w')
result_headers = ['version', 'project', 'algorithm', 'batch_size', 'confidence', 'project_reqd_builds', 'project_missed_builds', 'project_saved_builds', 'project_delays', 'testall_size', 'batch_delays', 'batch_median', 'ci']
writer = csv.writer(result_file)
writer.writerow(result_headers)
result_file.close()


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


def pd_get_train_test_data(file_path, first_failures=True):
    columns = ['tr_build_id', 'git_num_all_built_commits', 'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_modified', 'tr_status']
    X = pd.read_csv(file_path, usecols = columns)
    X['tr_status'] = output_values(X['tr_status'])
    if first_failures:
        X = get_first_failures(X)
    #X.drop('tr_status', inplace=True, axis=1)
    
    #return X, Y
    return X


# In[52]:


def sbs(project_name, ver):

    project_file = "../data/full_data/" + project_name + '.csv'
    
    project =  pd_get_train_test_data(project_file)
    pkl_file = '../data/project_data_pickles/' + project_name + '.csv_' + str(ver) + '_indexes.pkl'
    with open(pkl_file, 'rb') as load_file:
        train_build_ids = pickle.load(load_file)
        test_build_ids = pickle.load(load_file)

    #dataset already has first failures
    #train_file = "../data/train_data/" + project_name + '_train.csv'
    
    
    #X_train, Y_train = pd_get_train_test_data(train_file)
    #X_train.drop('tr_status', inplace=True, axis=1)
    #X_train.drop('tr_build_id', inplace=True, axis=1)
    #X_train = X_train.reshape((int(len(X_train)), num_feature+1))
    #print(X_train)
    #print(Y_train)

    

    X_train = project [ project['tr_build_id'].isin(train_build_ids)]
    Y_train = X_train['tr_status'].tolist()
    
    X_test = project [ project['tr_build_id'].isin(test_build_ids)]
    #X_train.drop('tr_status', inplace=True, axis=1)
    #X_train.drop('tr_build_id', inplace=True, axis=1)
    
    num_feature = 4
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
    
    for i in range(100):
        print('Bootstrapping {} for {}'.format(i, project_name))
        
        while True:
            print('Here for {} {}'.format(i, project_name))
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
        
        print('Training {} for {}'.format(i, project_name))
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

    file_name = 'rq2_dump_data/rq2_' + project_name + '_' + str(ver) + '_best_model.pkl'
    dump_file = open(file_name, 'wb')
    pickle.dump(forest, dump_file)
    pickle.dump(threshold, dump_file)
    pickle.dump(n_estimator, dump_file)
    pickle.dump(max_depth, dump_file)
                
    #grid_search.fit(X_train, Y_train)
    #return grid_search
    return forest, threshold, X_test


# In[53]:


def get_durations(project):
    csv_file = pd.read_csv(project)
    durations = csv_file['tr_duration'].tolist()
    return durations


# In[54]:


batch_total = 0
batch_durations = 0


# In[55]:


def batch_bisect(grouped_batch, actual_group_results):
    global batch_total
 
    batch_total += 1
    
    if len(actual_group_results) == 1:
        return
    
    if 0 in actual_group_results:
        half_batch = len(actual_group_results)//2
        batch_bisect(grouped_batch[:half_batch], actual_group_results[:half_batch])
        batch_bisect(grouped_batch[half_batch:], actual_group_results[half_batch:])


# In[56]:


def batch_stop_4(grouped_batch, actual_group_results):
    global batch_total
    
    batch_total += 1

    if len(actual_group_results) <= 4:
        if 0 in actual_group_results:
            batch_total += 4
        return
    
    if 0 in actual_group_results:
        half_batch = len(actual_group_results)//2
        batch_stop_4(grouped_batch[:half_batch], actual_group_results[:half_batch])
        batch_stop_4(grouped_batch[half_batch:], actual_group_results[half_batch:])


# In[57]:


def static_rule(p, ver):
    
    
    global batch_total
    global batch_durations
    
    p = p.split('.')[0]
    
    #result_file_name = p + '_rq2_results.csv'
    result_file = open('all_rq2_results.csv', 'a+')
    writer = csv.writer(result_file)

    
    #predictor, threshold, X_test = sbs(p, ver)
    #if len(X_test) == 0:
    #    return


    model_file_name = '../../RQ2-Models/' + p + '_models/rq2_' + p + '_' + str(ver) + '_best_model.pkl'
    model_file = open(model_file_name, 'rb')
    predictor = pickle.load(model_file)
    threshold = pickle.load(model_file)
    
    #get the test data
    
    test_file = "../data/full_data/" + p + '.csv'

    project = pd_get_train_test_data(test_file, first_failures=False)
    
    pkl_file = '../data/project_data_pickles/' + p + '.csv_' + str(ver) + '_indexes.pkl'
    with open(pkl_file, 'rb') as load_file:
        train_build_ids = pickle.load(load_file)
        test_build_ids = pickle.load(load_file)

    X_test = project [ project['tr_build_id'].isin(test_build_ids)]
    if len(X_test) == 0:
        return
    Y_test = X_test['tr_status'].tolist()

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
                    
            print('Processing {} at batch size {} for {}'.format(alg, max_batch_size, p))


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
            project_batch_medians = []
            project_ci = []

            print('Processing {}'.format(p))
            commit = predictor.predict(X_test)
            for c in confidence:
                ci = [Y_test[0]]
                batch_median = []
                batch_delays = 0

                pass_streak = Y_test[0]
                total_builds = 0
                missed_builds = 0
                miss_indexes = []
                build_indexes = []
                delay_durations = []

                if pass_streak == 0:
                    saved_builds = 0
                else:
                    saved_builds = 1

                index = 1

                while index < len(X_test):
                    value = commit[index]
                    #we're setting a confidence of 'c' builds on SBS, if more than 'c' passes have been suggested in a row, we don't want to trust sbs
                    
                    #if predict[0][1] > threshold:
                    #    value = 1
                    #else:
                    #    value = 0
                    #print('Build {} : predict_proba={}\tprediction={}'.format(index, predict, value))
                    
                    
                    if pass_streak < c :
                        
                        if value == 0:
                            while True:

                                grouped_batch = list(X_test[index : index+max_batch_size])
                                actual_group_results = list(Y_test[index : index+max_batch_size])

                #                 if len(grouped_batch) < max_batch_size:
                #                     grouped_batch.append(index)
                #                     actual_group_results.append(Y_test[index])
                #                     group_duration.append(Y_duration[index])
                                if alg == 'BATCH4':
                                    if len(actual_group_results) != max_batch_size:
                                        fb = 0
                                        while fb < len(actual_group_results):
                                            #miss_indexes.append(index)
                                            batch_delays += len(actual_group_results) - fb
                                            batch_median.append(max_batch_size-fb-1)
                                            ci.append(0)
                                            fb += 1
                                            index += 1
                                            total_builds += 1
                                    else:
                                        if len(miss_indexes) > 0:
                                            if miss_indexes[-1] < index:
                                                for l in range(len(miss_indexes)):
                                                    e = miss_indexes.pop()
                                                    delay_durations.append(index - e + 1)

                                        batch_delays += max_batch_size*(max_batch_size-1)/2
                                        batch_median.extend([max_batch_size-clb-1 for clb in range(max_batch_size)])
                                        ci.extend([0 for clb in range(max_batch_size)])
                                        total_builds += 1
                                        

                                        if 0 in actual_group_results:
                                            total_builds += max_batch_size
                                            
                                        #grouped_batch.clear()
                                        #actual_group_results.clear()
                                        #group_duration.clear()

                                elif alg == 'BATCHBISECT':
                                    if len(actual_group_results) != max_batch_size:
                                        fb = 0
                                        while fb < len(actual_group_results):
                                            total_builds += 1
                                            ci.append(0)
                                            batch_delays += len(actual_group_results) - fb
                                            batch_median.append(max_batch_size-fb-1)
                                            fb += 1
                                            index += 1
                                    else:
                                        if len(miss_indexes) > 0:
                                            if miss_indexes[-1] < index:
                                                for l in range(len(miss_indexes)):
                                                    e = miss_indexes.pop()
                                                    delay_durations.append(index - e + 1)

                                        batch_total = 0
                                        
                                        batch_bisect(grouped_batch, actual_group_results)
                                        batch_delays += max_batch_size*(max_batch_size-1)/2
                                        ci.extend([0 for clb in range(max_batch_size)])
                                        batch_median.extend([max_batch_size-clb-1 for clb in range(max_batch_size)])
                                        total_builds += batch_total

                                        #grouped_batch.clear()
                                        #actual_group_results.clear()
                                elif alg == 'BATCHSTOP4':
                                    if len(actual_group_results) != max_batch_size:
                                        fb = 0
                                        while fb < len(actual_group_results):
                                            total_builds += 1
                                            ci.append(0)
                                            batch_delays += len(actual_group_results) - fb
                                            batch_median.append(max_batch_size-fb-1)
                                            fb += 1
                                            index += 1
                                    else:
                                        if len(miss_indexes) > 0:
                                            if miss_indexes[-1] < index:
                                                for l in range(len(miss_indexes)):
                                                    e = miss_indexes.pop()
                                                    delay_durations.append(index - e + 1)

                                        batch_total = 0
                                        batch_durations = 0

                                        batch_stop_4(grouped_batch, actual_group_results)

                                        batch_delays += max_batch_size*(max_batch_size-1)/2
                                        ci.extend([0 for clb in range(max_batch_size)])
                                        batch_median.extend([max_batch_size-clb-1 for clb in range(max_batch_size)])
                                        total_builds += batch_total


                                        #grouped_batch.clear()
                                        #actual_group_results.clear()
                                        #group_duration.clear()
                                elif alg == 'NOBATCH':
                                    if len(grouped_batch) != max_batch_size:
                                        if len(miss_indexes) > 0:
                                            if miss_indexes[-1] < index:
                                                for l in range(len(miss_indexes)):
                                                    e = miss_indexes.pop()
                                                    delay_durations.append(index - e + 1)

                                        batch_delays += max_batch_size*(max_batch_size-1)/2
                                        total_builds += 1
                                        total_duration += max(group_duration)

                                        #grouped_batch.clear()
                                        #actual_group_results.clear()
                                        #group_duration.clear()


                                if 0 in actual_group_results:
                                    index += max_batch_size
                                    grouped_batch.clear()
                                    actual_group_results.clear()
                                    
                                else:
                                    break
                            index += max_batch_size
                            pass_streak = 1
                            grouped_batch.clear()
                            actual_group_results.clear()
                            
                                
#                             total_builds += 1
#                             total_duration += Y_duration[index]
#                             if len(miss_indexes) > 0:
#                                 if miss_indexes[-1] < index:
#                                     for l in range(len(miss_indexes)):
#                                         e = miss_indexes.pop()
#                                         delay_durations.append(index - e + 1)
                        else:
                            pass_streak += 1
                            ci.append(1)
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
                            

                            #print(grouped_batch)
                            #print(actual_group_results)

            #                 if len(grouped_batch) < max_batch_size:
            #                     grouped_batch.append(index)
            #                     actual_group_results.append(Y_test[index])
            #                     group_duration.append(Y_duration[index])

                            if alg == 'BATCH4':
                                if len(actual_group_results) != max_batch_size:
                                    fb = 0
                                    while fb < len(actual_group_results):
                                        total_builds += 1
                                        ci.append(0)
                                        batch_delays += len(actual_group_results) - fb
                                        batch_median.append(max_batch_size-fb-1)
                                        fb += 1
                                        index += 1
                                else:
                                    if len(miss_indexes) > 0:
                                        if miss_indexes[-1] < index:
                                            for l in range(len(miss_indexes)):
                                                e = miss_indexes.pop()
                                                delay_durations.append(index - e + 1)
                                    
                                    batch_delays += max_batch_size*(max_batch_size-1)/2
                                    ci.extend([0 for clb in range(max_batch_size)])
                                    batch_median.extend([max_batch_size-clb-1 for clb in range(max_batch_size)])
                                    total_builds += 1
                                    

                                    if 0 in actual_group_results:
                                        total_builds += max_batch_size
                                        

                                    #grouped_batch.clear()
                                    #actual_group_results.clear()
                                    #group_duration.clear()

                            elif alg == 'BATCHBISECT':
                                if len(actual_group_results) != max_batch_size:
                                    fb = 0
                                    while fb < len(actual_group_results):
                                        total_builds += 1
                                        ci.append(0)
                                        batch_delays += len(actual_group_results) - fb
                                        batch_median.append(max_batch_size-fb-1)
                                        fb += 1
                                        index += 1
                                else:

                                    if len(miss_indexes) > 0:
                                        if miss_indexes[-1] < index:
                                            for l in range(len(miss_indexes)):
                                                e = miss_indexes.pop()
                                                delay_durations.append(index - e + 1)

                                    batch_total = 0
                                    

                                    batch_bisect(grouped_batch, actual_group_results)

                                    batch_delays += max_batch_size*(max_batch_size-1)/2
                                    batch_median.extend([max_batch_size-clb-1 for clb in range(max_batch_size)])
                                    
                                    ci.extend([0 for clb in range(max_batch_size)])
                                    total_builds += batch_total
                                    

                                    #print(actual_group_results)
                                    #print('Total builds = {}, batch_total = {}, batch_delays={}'.format(total_builds, batch_total, batch_delays))
                                    #print('Total builds = {}, batch_delays={}'.format(total_builds, batch_delays))

                                    #grouped_batch.clear()
                                    #actual_group_results.clear()
                                    #group_duration.clear()
                            elif alg == 'BATCHSTOP4':
                                if len(actual_group_results) != max_batch_size:
                                    fb = 0
                                    while fb < len(actual_group_results):
                                        total_builds += 1
                                        ci.append(0)
                                        batch_delays += len(actual_group_results) - fb
                                        batch_median.append(max_batch_size-fb-1)
                                        fb += 1
                                        index += 1
                                else:

                                    if len(miss_indexes) > 0:
                                        if miss_indexes[-1] < index:
                                            for l in range(len(miss_indexes)):
                                                e = miss_indexes.pop()
                                                delay_durations.append(index - e + 1)

                                    batch_total = 0
                                    

                                    batch_stop_4(grouped_batch, actual_group_results)

                                    batch_delays += max_batch_size*(max_batch_size-1)/2
                                    batch_median.extend([max_batch_size-clb-1 for clb in range(max_batch_size)])
                                    ci.extend([0 for clb in range(max_batch_size)])
                                    total_builds += batch_total
                                    

                                    #print(actual_group_results)
                                    #print('Total builds = {}, batch_total = {}, batch_delays={}'.format(total_builds, batch_total, batch_delays))

                                    #grouped_batch.clear()
                                    #actual_group_results.clear()
                                    #group_duration.clear()
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
                                grouped_batch.clear()
                                actual_group_results.clear()
                                
                            else:
                                break
                        index += max_batch_size
                        pass_streak = 1
                        grouped_batch.clear()
                        actual_group_results.clear()
                        
                mi = 0
                while len(miss_indexes) > 0:
                        m_index = miss_indexes.pop()
                        delay_durations.append(length_of_test - m_index + 1)

                project_reqd_builds.append(total_builds)
                project_missed_builds.append(missed_builds)
                project_saved_builds.append(saved_builds)
                project_delays.append(delay_durations)
                project_batch_delays.append(batch_delays)
                project_batch_medians.append(batch_median)
                project_ci.append(ci)
                
                if len(ci) != len(commit):
                    print(len(ci))
                    print(len(commit))
                    print('PROBLEM!')
                else:
                    print('NO PROBLEM!')

            #print(p)
            #print(project_reqd_builds)
            #print(project_missed_builds)
            #print(project_build_duration)
            #print(project_saved_builds)
            #print(project_delays)
            #print(project_batch_delays)
            
            for i in range(len(confidence)):
                #print([p, alg, max_batch_size, confidence[i], 100*project_reqd_builds[i]/length_of_test, 100*project_missed_builds[i]/length_of_test, project_build_duration[i], 100*project_saved_builds[i]/length_of_test, project_delays[i], length_of_test, project_batch_delays[i]])
                writer.writerow([ver, p, alg, max_batch_size, confidence[i], 100*project_reqd_builds[i]/length_of_test, 100*project_missed_builds[i]/length_of_test, 100*project_saved_builds[i]/length_of_test, project_delays[i], length_of_test, project_batch_delays[i], project_batch_medians[i], project_ci[i]])
    result_file.close()


for pr in projects:
    for i in range(1,11):
        static_rule(pr, i)

