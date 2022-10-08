#!/usr/bin/env python
# coding: utf-8

# In[93]:


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
from matplotlib import pyplot
from statistics import median
import pickle
import csv
import warnings
import datetime
import multiprocessing
warnings.filterwarnings("ignore")


# In[94]:


project_list = ['cloud_controller_ng', 'jruby', 'chef', 'orbeon-forms']


# In[95]:


def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0) 
    return Y_t


# In[96]:


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


# In[97]:


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


# In[98]:


def get_complete_data(p_name):
    
    #open the metrics file
    filename = 'metrics_data/' + p_name + '_metrics.csv'
    project = pd.read_csv(filename)
    
    #clean the data & remove correlated columns
    project = project [ project['developer_experience'] >= 0]
    project.drop('num_commits', inplace=True, axis=1)
    project.drop('reviewer_experience', inplace=True, axis=1)
    project.drop('num_of_reviewers', inplace=True, axis=1)
    
    build_ids = project['tr_build_id'].tolist()
    #get results data
    res_file = '../data_distributions/data/' + p_name + '.csv'
    res_project = pd.read_csv(res_file, usecols = ['tr_build_id', 'gh_build_started_at', 'tr_status'])
    res_project['gh_build_started_at'] =  pd.to_datetime(res_project['gh_build_started_at'], format='%Y-%m-%d %H:%M:%S')
    y_project = res_project[res_project['tr_build_id'].isin(build_ids)]['tr_status'].tolist()
    y_project = output_values(y_project)
    
    #append date of build
    project_dates = res_project[res_project['tr_build_id'].isin(build_ids)]['gh_build_started_at'].tolist()
    project['gh_build_started_at'] = project_dates
    
    #add results column to the dataframe
    project['tr_status'] = y_project
    
    return project


# In[99]:


def get_start_end_date(project):
    dates = project['gh_build_started_at'].tolist()
    
    start_date = dates[0] - datetime.timedelta(days = 1)
    end_date = dates[-1] - datetime.timedelta(days = 1)
    
    return start_date, end_date


# In[100]:


def get_required_data(p_name, build_ids):
    
    res_file = '../data_distributions/data/' + p_name + '.csv'
    res_project = pd.read_csv(res_file, usecols = ['tr_build_id', 'tr_duration'])
    durations = res_project[res_project['tr_build_id'].isin(build_ids)]['tr_duration'].tolist()
    return durations


# In[101]:


def compute_performance(p_name, test_builds, test_result, pred_result, verbosity):
    
    
    
    durations = get_required_data(p_name, test_builds)
    actual_duration = sum(durations)
    actual_failures = test_result.count(0)
    
    total_builds = len(test_builds)
    num_of_builds = 0
    total_duration = 0
    cbf = 0
    saved_builds = 0
    
    batch = []
    batch_duration = []
    actual_results = []
    max_batch_size = 4
    
    for i in range(len(pred_result)):
        if pred_result[i] == 0:
            
            if test_result[i] == 0:
                cbf += 1
                
            if len(batch) < max_batch_size:
                batch.append(pred_result[i])
                batch_duration.append(durations[i])
                actual_results.append(test_result[i])
            
            if len(batch) == max_batch_size:
                num_of_builds += 1
                total_duration += max(batch_duration)
                
                if 0 in actual_results:
                    num_of_builds += 4
                    total_duration += sum(batch_duration)
        else:
            saved_builds += 1
            
    if len(batch) > 0:
        num_of_builds += 1
        total_duration += max(batch_duration)
        
        if 0 in actual_results:
            num_of_builds += len(batch)
            total_duration += sum(batch_duration)
                    
    #Delay computation
    flag = 0
    count = 0
    delay = []
    for i in range(len(pred_result)):
        if flag == 1:
            if pred_result[i] == 1:
                count += 1
            
            if pred_result[i] == 0:
                delay.append(count)
                count = 0
                flag = 0
                
        if test_result[i] != 1:
            if pred_result[i] == 1:
                flag = 1
    delay.append(count)

    
    try:
        
        time_saved = 100*total_duration/actual_duration
        builds_saved = 100*saved_builds/total_builds
        reqd_builds = 100*num_of_builds/total_builds
        failed = 100*cbf/actual_failures
        median_delays = median(delay)
        total_delays = sum(delay)
        
        if verbosity:
    
            print("===========================================")
            print('The performance of the model is as follows:')
            print('\t Time Reqd : {}'.format(total_duration))
            print('\t % Time Reqd : {}%'.format(time_saved))
            print('\t Num. Builds saved : {}%'.format(saved_builds))
            print('\t % Builds saved : {}%'.format(builds_saved))
            print('\t Num. Builds required : {}'.format(num_of_builds))
            print('\t % Builds required : {}%'.format(reqd_builds))
            print('\t Num. Failed Builds Identified : {}'.format(cbf))
            print('\t % Failed Builds Identified : {}%'.format(failed))
            print('\t Median Delay Induced : {} builds'.format(median_delays))
            print('\t Total Delay Induced: {} builds'.format(total_delays))
            print('\t Total number of builds: {}'.format(total_builds))
            print('\t Total number of failed builds: {}'.format(actual_failures))
            print('\t Total Duration: {}'.format(actual_duration))
            print("===========================================")
        
    except:
        
        print('exception')
        return (0, 0, 0, 0, 0, 0)
    
    return (time_saved, builds_saved, reqd_builds, failed, median_delays, total_delays)


# In[102]:


def hybrid_performance(p_name, test_builds, test_result, ci):
    total_builds = len(test_result)
    actual_builds_made = ci.count(0)//4
    
    
    durations = get_required_data(p_name, test_builds)
    build_time = []
    
    i = 0
    while i < len(test_result):
        if ci[i] == 0:
            batch = ci[i:i+4]
            build_time.append(max(durations[i:i+4]))
            i += 4
        else:
            i+=1
    
    time_reqd = 100*sum(build_time)/sum(durations)
    builds_reqd = 100*actual_builds_made/total_builds
    
    if len(ci) == len(test_result):
        if len(ci) == len(durations):
            print('Going right....')
            
    delay = []
    delay_indexes = []
    built_indexes = []
    for i in range(len(test_result)):
        if ci[i] == 0:
            built_indexes.append(i)
        if test_result[i] == 0:
            if ci[i] != 0:
                delay_indexes.append(i)
                
    num_of_failure_unidentified = len(delay_indexes)
    identified_failures = test_result.count(0) - num_of_failure_unidentified
    total_failures = test_result.count(0)
    if total_failures != 0:
        failures_found = 100*identified_failures/total_failures
        failures_not_found = 100*num_of_failures_unidentified/total_failures
    else:
        failures_found = 0
        failures_not_found = 0
    
#     print(delay_indexes)
#     print(built_indexes)
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
    
    print('Computed the performance')
    return (time_reqd, builds_reqd, sum(delay), failures_found, failures_not_found)


# In[108]:


def bootstrapping(p_name):
    
    print('Processing {}'.format(p_name))
    
    performances = {'time_reqd':[], 'builds_reqd':[], 'total_delay':[], 'failures_found':[], 'failures_not_found':[]}
    
    #This will return the entire dataset with X and Y values
    project = get_complete_data(p_name)
#     print(project)
#     print(list(project['tr_status']))
    start_date, end_date = get_start_end_date(project)
    
    #grid search hyperparameters
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    
    #setting up grid search
    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
    forest = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = forest, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 0)
    
    phase = 1

    while start_date < end_date:
        
        #setting the training and testing period in terms of num of days
        train_period = 400
        test_period = 30
        
        '''
        Getting a good number of training and testing data
        Setting min length of training set to be 100
        Setting min length of testing set to be 10
        
        if training set length < min or testing set length < min,
        increase the corresponding training and testing period
        '''
        
        while True:
            print('Calculating best training size for {}'.format(p_name))
            train_end = start_date + datetime.timedelta(days = train_period + 1)
            test_start = start_date + datetime.timedelta(days = train_period)
            test_end = test_start + datetime.timedelta(days = test_period)

            #getting data of train & test phase wise
            train_data = project[ (project['gh_build_started_at'] > start_date) & (project['gh_build_started_at'] < train_end)]
            test_data = project[ (project['gh_build_started_at'] > test_start) & (project['gh_build_started_at'] < test_end)]

            #getting 'y' data
            train_result = train_data['tr_status'].tolist()
            test_result = test_data['tr_status'].tolist()
            
            if len(train_result) > 300 and len(test_result) > 30 :
                break
            
            if test_end > end_date:
                if len(test_result) == 0:
                    train_end = end_date - datetime.timedelta(days = test_period)
                    test_start = train_end + datetime.timedelta(days = 1)
                    test_end = end_date
                    
                    train_data = project[ (project['gh_build_started_at'] > start_date) & (project['gh_build_started_at'] < train_end)]
                    test_data = project[ (project['gh_build_started_at'] > test_start) & (project['gh_build_started_at'] < test_end)]
                    
                    train_result = train_data['tr_status'].tolist()
                    test_result = test_data['tr_status'].tolist()
                break
                
            if len(train_result) <= 300:
                train_period += 20
            
            if len(test_result) <= 30:
                test_period += 20
                
        #Now we have gotten atleast minimum number of training and testing data    
        print('The training period = {} to {}'.format(start_date, train_end))
        print('The testing period = {} to {}'.format(test_start, test_end))
        
        print('Training size = {}'.format(len(train_result)))
        print('Testing size = {}'.format(len(test_result)))
        
        #dropping build start time column
        train_data.drop('gh_build_started_at', inplace=True, axis=1)
        test_data.drop('gh_build_started_at', inplace=True, axis=1)
        
        #add pass_streak to training data:
        train_data['num_of_passes'] = get_pass_streak(train_result)
        
        best_n_estimators = []
        best_max_depth = []
        
        best_f1 = 0
        best_f1_sample = 0
        best_f1_sample_result = 0
        best_f1_estimator = 0
        best_thresholds = []

        
        #bootstrap 10 times
        for i in range(1):
            print('Bootstrapping {} for {}'.format(i, p_name))
            
            
            #Ensuring we get a non-zero training or testing sample
            while True:
                sample_train = resample(train_data, replace=True, n_samples=len(train_data))
                sample_train_result = sample_train['tr_status']

                build_ids = sample_train['tr_build_id'].tolist()
                sample_test = train_data [~train_data['tr_build_id'].isin(build_ids)] 
                sample_test_result = sample_test['tr_status']
                
                if len(sample_test_result) != 0:
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
                accuracy = accuracy_score(sample_test_result, final_pred_result)
                precision = precision_score(sample_test_result, final_pred_result)
                recall = recall_score(sample_test_result, final_pred_result)
                confusion = confusion_matrix(sample_test_result, final_pred_result)
                auc_score = roc_auc_score(sample_test_result, final_pred_result)
                f1 = f1_score(sample_test_result, final_pred_result)
            except:
                print('')
    
            if f1 > best_f1:
                best_f1 = f1
                best_f1_sample = sample_train
                best_f1_sample_result = sample_train_result
                best_f1_estimator = grid_search.best_estimator_

#             print(precision, recall, accuracy, f1, auc_score)
            best_n_estimators.append(grid_search.best_params_['n_estimators'])
            best_max_depth.append(grid_search.best_params_['max_depth'])
            print('Completed a round for {}'.format(p_name))
        
        #completed with bootstrapping 
        threshold = median(best_thresholds)
#         print(threshold)
        n_estimator = median(best_n_estimators)
        max_depth = median(best_max_depth)
        
        #retrain to get the best model
        forest = RandomForestClassifier(n_estimators=int(n_estimator), max_depth=int(max_depth))
        forest.fit(best_f1_sample, best_f1_sample_result)
        
        test_builds = test_data['tr_build_id'].tolist()
        test_data.drop('tr_build_id', inplace=True, axis=1)
        test_data.drop('tr_status', inplace=True, axis=1)
        
        batch = []
        batch_durations = []
        actual_batch_results = []
        max_batch_size = 4
        final_pred_result = []
        pass_streak = 0
        i = 0
        total = len(test_data)
        
        num_of_builds = 0
        
        #The variable 'ci' will hold the actual execution process of the current phase
        #If ci[i] == 0, it means that build was made
        #If ci[i] == 1, it means that build was saved
        ci = []
        
        while i < total :
            data = test_data.iloc[i]
            data['num_of_passes'] = pass_streak
            predict = forest.predict_proba([data])
#             print(predict)
            #predicted that build has passed
            if predict[0][1] > predict[0][0]:
                final_pred_result.append(1)
                ci.append(1)
                pass_streak += 1
                i+=1
            else:
                #We found first failure
                
                #Until an entire batch passes, we are going to continue group builds ie., subsequent failures are grouped
                while True:
                    if (total - i) > 4:
                        ci.extend([0,0,0,0])
                    else:
                        ci.extend([0 for e in range(total-i)])
                    
                    num_of_builds += 1
                    actual_batch_results = test_result[i:i+4]
                    
                    #if any build has failed in the batch, then whole batch will fail
                    if 0 in actual_batch_results:
                        i = i+4
                    else:
                        break
                #Now that we have found a passing build, we can update pass_streak to 1
                pass_streak = 1
                i += 4
                
                
#         print('Individual testing for {}....'.format(p_name))
#         try:
#             accuracy = accuracy_score(test_result, final_pred_result)
#             precision = precision_score(test_result, final_pred_result)
#             recall = recall_score(test_result, final_pred_result)
#             confusion = confusion_matrix(test_result, final_pred_result)
#             auc_score = roc_auc_score(test_result, final_pred_result)
#             f1 = f1_score(test_result, final_pred_result)

# #             print(precision, recall, accuracy, f1, auc_score)
# #             print(confusion)
            
#         except:
#             print('')
#         print(ci)
#         print(test_result)
#         print(len(ci))

        batch_performance = hybrid_performance(p_name, test_builds, test_result, ci)
        performances['time_reqd'].append(batch_performance[0])
        performances['builds_reqd'].append(batch_performance[1])
        performances['total_delay'].append(batch_performance[2])
        performances['failures_found'].append(batch_performance[3])
        performances['failures_not_found'].append(batch_performance[4])
        
        start_date = test_end
        phase += 1
    
    with open(res_file, 'w') as fres:
        fres.write("Average Time Reqd in {} = {}".format(p_name, sum(performances['time_reqd'])/len(performances['time_reqd'])))
        fres.write("Average Builds Reqd in {} = {}".format(p_name, sum(performances['builds_reqd'])/len(performances['builds_reqd'])))
        fres.write("Average Total Delay in {} = {}".format(p_name, sum(performances['total_delay'])/len(performances['total_delay'])))
        fres.write("Average Failed Identified in {} = {}".format(p_name, sum(performances['failures_found'])/len(performances['failures_found'])))
        fres.write("Average Failed Unidentified in {} = {}".format(p_name, sum(performances['failures_not_found'])/len(performances['failures_not_found'])))
    
    print("Average Time Reqd in {} = {}".format(p_name, sum(performances['time_reqd'])/len(performances['time_reqd'])))
    print("Average Builds Reqd in {} = {}".format(p_name, sum(performances['builds_reqd'])/len(performances['builds_reqd'])))
    print("Average Total Delay in {} = {}".format(p_name, sum(performances['total_delay'])/len(performances['total_delay'])))
    print("Average Failed Identified in {} = {}".format(p_name, sum(performances['failures_found'])/len(performances['failures_found'])))
    print("Average Failed Unidentified in {} = {}".format(p_name, sum(performances['failures_not_found'])/len(performances['failures_not_found'])))

    print('\n\n\n\n\n')


# In[111]:

# In[113]:
# In[ ]:

'''jobs = []
for p_name in project_list:
    
    q = multiprocessing.Process(target=bootstrapping, args=(p_name,))
    jobs.append(q)
    q.start()

for j in jobs:
    j.terminate()
    j.join()'''

print('Starting pooling')
from multiprocessing import Pool

with Pool(9) as pool:
    pool.map(bootstrapping, project_list)


# In[ ]:




