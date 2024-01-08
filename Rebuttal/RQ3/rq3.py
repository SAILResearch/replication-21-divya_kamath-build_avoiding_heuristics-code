#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import multiprocessing
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")



# In[2]:

num_cores = multiprocessing.cpu_count()
cores_used = num_cores/5
pd.options.mode.chained_assignment = None  # default='warn'


project_list = ['gradle.csv', 'rails.csv', 'heroku.csv', 'jruby.csv', 'metasploit-framework.csv', 'cloudify.csv', 'vagrant.csv', 'rubinius.csv', 'open-build-service.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'opal.csv', 'cloud_controller_ng.csv', 'puppet.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv']


confidence = list(range(2, 21, 1))

# In[4]:


batch_total = 0
batch_duration = 0


# In[5]:


def batch_bisect(batch_results, duration_subbatch):
    global batch_total
    global batch_duration
    
    batch_total += 1
    batch_duration += duration_subbatch[-1]
    
    if len(batch_results) == 1:
        return
    
    if 0 in batch_results:
        half_batch = len(batch_results)//2
        batch_bisect(batch_results[:half_batch], duration_subbatch[:half_batch])
        batch_bisect(batch_results[half_batch:], duration_subbatch[half_batch:])


# In[6]:


def batch_stop_4(batch_results, duration_subbatch):
    global batch_total
    global batch_duration
    
    batch_total += 1
    batch_duration += duration_subbatch[-1]
    
    if len(batch_results) <= 4:
        if 0 in batch_results:
            batch_total += 4
        return
    
    if 0 in batch_results:
        half_batch = len(batch_results)//2
        batch_stop_4(batch_results[:half_batch], duration_subbatch[:half_batch])
        batch_stop_4(batch_results[half_batch:], duration_subbatch[half_batch:])


# In[7]:


def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0) 
    return Y_t


# In[8]:


def get_pass_streak(y_project):
    p = y_project[0]
    pass_streak = [y_project[0]]
    for i in range(1, len(y_project)):
        pass_streak.append(p)
        if y_project[i] == 1:
            p += 1
        else:
            p = 0
    return pass_streak


# In[9]:


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


# In[10]:


def get_complete_data(p_name, first_failures=True):
    
    #open the metrics file
    filename = '../../data/full_data/' + p_name
    columns = ['tr_build_id', 'git_num_all_built_commits', 'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_modified', 'tr_duration', 'tr_status']
    project = pd.read_csv(filename, usecols = columns)
    project['tr_status'] = output_values(project['tr_status'])
    if first_failures:
        project = get_first_failures(project)
    return project



# In[11]:

def get_results(results):
    l = []
    
    for r in results:
        if r == 'passed':
            l.append(1)
        else:
            l.append(0)
            
    return l


def hybrid_performance(p_name, test_builds, test_result, batchsize, ci):
    total_builds = len(test_result)

    bad_builds = 0
    flag = 0
    for i in range(len(test_result)):
        if flag == 1:
            if ci[i] == 1:
                bad_builds += 1
            else:
                flag == 0
        else:
            if test_result[i] == 0:
                if ci[i] == 1:
                    flag = 1
                    bad_builds += 1

    

    delay = []
    delay_indexes = []
    built_indexes = []
    for i in range(len(test_result)):
        if ci[i] == 0:
            built_indexes.append(i)
        if test_result[i] == 0:
            if ci[i] != 0:
                delay_indexes.append(i)
    
    num_failed = test_result.count(0)
    if num_failed == 0:
        failures_found = 100
        failures_not_found = 0

    else:
        num_of_failure_unidentified = len(delay_indexes)
        identified_failures = test_result.count(0) - num_of_failure_unidentified
        failures_found = 100*identified_failures/test_result.count(0)
        failures_not_found = 100*num_of_failure_unidentified/test_result.count(0)
    
    from_value = 0
    
    for k in range(len(built_indexes)):
        for j in range(len(delay_indexes)):
            if delay_indexes[j] > from_value and delay_indexes[j] < built_indexes[k]:
                delay.append(built_indexes[k] - delay_indexes[j])
        from_value = built_indexes[k]
    
    if len(delay_indexes) != 0:
        final_index = len(test_result)
        for j in range(len(delay_indexes)):
            delay.append(final_index - delay_indexes[j])
            
    return (sum(delay), failures_found, failures_not_found, bad_builds)


def get_delay_from_ci(ci, y_test, batch_size):
    
    i = 0        
    delay_list = []
    missed = []
    batch_list = []
    sbs_list = []
    delay_list = []
    
    b = batch_size-1

    while i < len(ci):
        if ci[i] == 0:

            while len(missed) > 0:
                ind = missed.pop()
                sbs_list.append(i - ind)

            batch_list.append(b)
            b -= 1
            if b == -1:
                b = batch_size - 1
            
            #print(y_test[i], ci[i], sbs_delay, batch_delay, sbs_delay + batch_delay)

        if ci[i] == 1:
            if y_test[i] == 0:
                missed.append(i)

        i += 1

    while len(missed) > 0:
            sbs_list.append(i - missed.pop())
    
    delay_list.extend(sbs_list)
    delay_list.extend(batch_list)
    
    return delay_list


def bootstrapping(p_name, train_data, count):
    
    
    #grid search hyperparameters
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    
    #setting up grid search
    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
    forest = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = forest, param_grid = param_grid, cv = 3, n_jobs = 40, verbose = 0)
    
    print(len(train_data))
    if len(train_data) <= 1:
        return -1, -1
        
        
    train_result = train_data['tr_status'].tolist()
    train_data['num_of_passes'] = get_pass_streak(train_result)
    
    best_n_estimators = []
    best_max_depth = []
    best_copy = train_data.copy()
    best_copy.drop('tr_status', inplace=True, axis=1)
    best_copy.drop('tr_build_id', inplace=True, axis=1)
    
    
    best_f1 = 0
    best_f1_sample = best_copy
    best_f1_sample_result = train_result
    best_f1_estimator = 0
    best_thresholds = []


    #bootstrap 100 times
    for i in range(100):
        print('Bootstrapping {} for {}'.format(i, p_name))

        #Ensuring we get a non-zero training or testing sample
        while True:
            print('Here for {} {}'.format(i, p_name))
            sample_train = resample(train_data, replace=True, n_samples=len(train_data))
            sample_train_result = sample_train['tr_status']

            build_ids = sample_train['tr_build_id'].tolist()
            sample_test = train_data [~train_data['tr_build_id'].isin(build_ids)] 
            sample_test_result = sample_test['tr_status']

            if (len(sample_test_result) != 0) and (len(sample_train_result) != 0):
                break

        #dropping result column and build ids column
        sample_train.drop('tr_status', inplace=True, axis=1)
        sample_train.drop('tr_build_id', inplace=True, axis=1)
        sample_test.drop('tr_status', inplace=True, axis=1)
        sample_test.drop('tr_build_id', inplace=True, axis=1)

        #training the sample
        print('Training {} for {}'.format(i, p_name))
        grid_search.fit(sample_train, sample_train_result)
        sample_pred_vals = grid_search.predict_proba(sample_test)
        
        empty_np_array = np.empty([len(sample_test), 1])
        if sample_pred_vals.shape == empty_np_array.shape:
            pred_vals = sample_pred_vals[:, 0]
        else:
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

        best_n_estimators.append(grid_search.best_params_['n_estimators'])
        best_max_depth.append(grid_search.best_params_['max_depth'])


    #completed with bootstrapping 
    threshold = median(best_thresholds)
    n_estimator = median(best_n_estimators)
    max_depth = median(best_max_depth)

    #retrain to get the best model
    forest = RandomForestClassifier(n_estimators=int(n_estimator), max_depth=int(max_depth))
    forest.fit(best_f1_sample, best_f1_sample_result)

    file_name = 'dump_data/rq2_' + p_name + '_' + str(count) + '_best_model.pkl'
    dump_file = open(file_name, 'wb')
    pickle.dump(forest, dump_file)
    pickle.dump(threshold, dump_file)
    pickle.dump(n_estimator, dump_file)
    pickle.dump(max_depth, dump_file)
        
        
    return forest, threshold



def timeout_rule_process(p_name):
    
    global batch_total
    global batch_duration
    
    p = p_name.split('.')[0]
    
    path = 'dump_data/' + p + '_models/rq1_' + p_name + '_'
    
    print('Processing {}'.format(p_name))

    result_rows = []
    empty_prediction = np.empty([1, 2])
    
    project = get_complete_data(p_name, first_failures=False)
    
    #sliding window parameters
    window_size = len(project)//5
    end_p = window_size
    start_p = 0
    
    while end_p <= len(project):
        
        window = project[start_p:end_p]
        
        train_start = start_p
        train_end = start_p + int(0.7*len(window))
        test_start = train_end
        test_end = end_p
        
        train_data = project[train_start:train_end]
        test_data = project[test_start:test_end]
        
        #forest, threshold = bootstrapping(p_name, train_data, end_p)
        
        model_file_path = path + str(end_p) + '_best_model.pkl'
        model_file = open(model_file_path, 'rb')
        forest = pickle.load(model_file)
        threshold = pickle.load(model_file)
        
        if type(forest) == type(int):
            print("Ending at {}".format(end_p))
            break
    
        test_result = test_data['tr_status'].tolist()
        test_builds = test_data['tr_build_id'].tolist()
        test_duration = test_data['tr_duration'].tolist()

        if len(test_result) == 0:
            return 

        test_data.drop('tr_build_id', inplace=True, axis=1)
        test_data.drop('tr_status', inplace=True, axis=1)
        test_data.drop('tr_duration', inplace=True, axis=1)
        
        Y_test = test_result
        X_test = test_data

        batchsizelist = [16, 8, 4, 2, 1]
        algorithms = ['BATCH4', 'BATCHSTOP4', 'BATCHBISECT']

        batch_delays = 0
        final_pred_result = []
        
        commit = forest.predict(test_data)
        print(test_result)
        print(commit)
        print(test_start, end_p)
        
        project_actual_duration = 0
        project_batch_duration = 0

        for alg in algorithms:
            for batchsize in batchsizelist:
                
                if alg == 'BATCH4':
                    if batchsize != 4:
                        continue
                
                if alg == 'BATCHSTOP4':
                    if batchsize < 4:
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
                project_actual_duration_list = []
                project_saved_builds = []
                project_delays = []
                project_bad_builds = []
                project_batch_delays = []
                project_batch_medians = []
                project_ci = []
                
                max_batch_size = batchsize
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

                                    grouped_batch = list(commit[index : index+max_batch_size])
                                    actual_group_results = list(Y_test[index : index+max_batch_size])
                                    duration_subbatch = test_duration[index : index+max_batch_size]
                                    project_actual_duration += sum(duration_subbatch)
                                    
                                    print('actual: {}'.format(actual_group_results))
                                    print('grouped: {}'.format(grouped_batch))

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
                                                print(fb)
                                                
                                                project_batch_duration += duration_subbatch[fb-1]
                                                
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
                                            project_batch_duration += duration_subbatch[-1]


                                            if 0 in actual_group_results:
                                                total_builds += max_batch_size
                                                project_batch_duration += sum(duration_subbatch)


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
                                                
                                                project_batch_duration += duration_subbatch[fb-1]
                                        else:
                                            if len(miss_indexes) > 0:
                                                if miss_indexes[-1] < index:
                                                    for l in range(len(miss_indexes)):
                                                        e = miss_indexes.pop()
                                                        delay_durations.append(index - e + 1)

                                            batch_total = 0
                                            batch_duration = 0

                                            batch_bisect(grouped_batch, duration_subbatch)
                                            
                                            
                                            print(batch_total)
                                            batch_delays += max_batch_size*(max_batch_size-1)/2
                                            ci.extend([0 for clb in range(max_batch_size)])
                                            batch_median.extend([max_batch_size-clb-1 for clb in range(max_batch_size)])
                                            
                                            total_builds += batch_total
                                            project_batch_duration += batch_duration

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
                                                
                                                project_batch_duration += duration_subbatch[fb-1]
                                        else:
                                            if len(miss_indexes) > 0:
                                                if miss_indexes[-1] < index:
                                                    for l in range(len(miss_indexes)):
                                                        e = miss_indexes.pop()
                                                        delay_durations.append(index - e + 1)

                                            batch_total = 0
                                            batch_duration = 0

                                            batch_stop_4(grouped_batch, duration_subbatch)

                                            batch_delays += max_batch_size*(max_batch_size-1)/2
                                            ci.extend([0 for clb in range(max_batch_size)])
                                            batch_median.extend([max_batch_size-clb-1 for clb in range(max_batch_size)])
                                            
                                            total_builds += batch_total
                                            project_batch_duration += batch_duration


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

                                grouped_batch = list(commit[index : index+max_batch_size])
                                actual_group_results = list(Y_test[index : index+max_batch_size])
                                duration_subbatch = test_duration[index : index+max_batch_size]
                                project_actual_duration += sum(duration_subbatch)
                                
                                print('actual: {}'.format(actual_group_results))
                                print('grouped: {}'.format(grouped_batch))
                                


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
                                            print(fb)
                                            
                                            project_batch_duration += duration_subbatch[fb-1]
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
                                        project_batch_duration += duration_subbatch[-1]


                                        if 0 in actual_group_results:
                                            total_builds += max_batch_size
                                            project_batch_duration += sum(duration_subbatch)


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
                                            
                                            project_batch_duration += duration_subbatch[fb-1]
                                    else:

                                        if len(miss_indexes) > 0:
                                            if miss_indexes[-1] < index:
                                                for l in range(len(miss_indexes)):
                                                    e = miss_indexes.pop()
                                                    delay_durations.append(index - e + 1)

                                        batch_total = 0
                                        batch_duration = 0

                                        batch_bisect(grouped_batch, duration_subbatch)

                                        batch_delays += max_batch_size*(max_batch_size-1)/2
                                        batch_median.extend([max_batch_size-clb-1 for clb in range(max_batch_size)])

                                        ci.extend([0 for clb in range(max_batch_size)])
                                        
                                        total_builds += batch_total
                                        project_batch_duration += batch_duration


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
                                            
                                            project_batch_duration += duration_subbatch[fb-1]
                                    else:

                                        if len(miss_indexes) > 0:
                                            if miss_indexes[-1] < index:
                                                for l in range(len(miss_indexes)):
                                                    e = miss_indexes.pop()
                                                    delay_durations.append(index - e + 1)

                                        batch_total = 0
                                        batch_duration = 0


                                        batch_stop_4(grouped_batch, duration_subbatch)

                                        batch_delays += max_batch_size*(max_batch_size-1)/2
                                        batch_median.extend([max_batch_size-clb-1 for clb in range(max_batch_size)])
                                        ci.extend([0 for clb in range(max_batch_size)])
                                        
                                        total_builds += batch_total
                                        project_batch_duration += batch_duration

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
                    project_build_duration.append(project_batch_duration)
                    project_actual_duration_list.append(project_actual_duration)
                    project_ci.append(ci)
                    
                    print(ci)
                    print(Y_test)
                    if len(ci) != len(commit):
                        print(len(ci))
                        print(len(commit))
                        print('PROBLEM!')
                    else:
                        print('NO PROBLEM!')
                
                    
                for i in range(len(confidence)):
                    delay_list = get_delay_from_ci(project_ci[i], Y_test, batchsize)
                    
                    result_rows.append([p, start_p, end_p, alg, batchsize, confidence[i], 100*project_reqd_builds[i]/length_of_test, delay_list, median(delay_list), project_build_duration[i], project_actual_duration_list[i], length_of_test, project_ci[i]])
        
        start_p = start_p + int((0.5)*window_size)
        end_p = start_p + window_size
        
    
    print('converting to csv')
    df = pd.DataFrame(result_rows, columns=['project', 'start_p', 'end_p', 'algorithm', 'batch_size', 'confidence', 'builds_reqd', 'delay_list', 'median_delay', 'build_duration', 'actual_duration', 'testall_size', 'ci'])
    n = p_name + '_sw.csv'
    df.to_csv(n)
    


# In[19]:

# print(len(project_list))
# output = Parallel(n_jobs=15)(delayed(mlci_process)(p_name) for p_name in project_list[:8])

for p_name in project_list:
    timeout_rule_process(p_name)


# In[ ]:




