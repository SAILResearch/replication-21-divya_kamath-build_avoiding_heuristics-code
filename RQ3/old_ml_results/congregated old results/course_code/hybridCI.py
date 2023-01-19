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
from matplotlib import pyplot
from statistics import median
import pickle
import csv
import warnings
import datetime
import multiprocess
warnings.filterwarnings("ignore")


# In[58]:


project_list = open('../data/datasets/list').read().split('\n')[:100]


# In[33]:


def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0) 
    return Y_t


# In[34]:


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


# In[35]:


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


# In[36]:


def get_complete_data(p_name):
    
    #open the metrics file
    filename = 'project_metrics/' + p_name.split('.')[0] + '_metrics.csv'
    project = pd.read_csv(filename)
    project['tr_status'] = output_values(project['tr_status'])
    return project


# In[45]:


def get_required_data(p_name, build_ids):
    
    res_file = '../data/datasets/all_datasets/' + p_name
    res_project = pd.read_csv(res_file, usecols = ['tr_build_id', 'tr_duration'])
    durations = res_project[res_project['tr_build_id'].isin(build_ids)]['tr_duration'].tolist()
    return durations


# In[46]:


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
    failures_found = 100*identified_failures/test_result.count(0)
    
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
    
#     print("===========================================")
#     print('Total Number of builds for {} = {}'.format(p_name, total_builds))
#     print('Total % of builds required for {} = {}'.format(p_name, builds_reqd))
#     print('Total % of time required for {} = {}'.format(p_name, time_reqd))
#     print('Total delays made for {} = {}'.format(p_name, sum(delay)))
#     print('Total % of failures identified for {} = {}'.format(p_name, failures_found))
#     print('Total % of failures unidentified for {} = {}'.format(p_name, 100*num_of_failure_unidentified/test_result.count(0)))
#     print("===========================================")
    
    return (time_reqd, builds_reqd, sum(delay), failures_found, 100*num_of_failure_unidentified/test_result.count(0))


# In[51]:


def bootstrapping(p_name):
    
    print('Processing {}'.format(p_name))
    
    performances = {'time_reqd':[], 'builds_reqd':[], 'total_delay':[], 'failures_found':[], 'failures_not_found':[]}
    
    #This will return the entire dataset with X and Y values
    project = get_complete_data(p_name)
    
    #grid search hyperparameters
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    
    #setting up grid search
    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
    forest = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = forest, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 0)
    
    pkl_file = '../data/even_data/data_pickles/' + p_name.split('.')[0] + '_indexes.pkl'
    with open(pkl_file, 'rb') as load_file:
        train_build_ids = pickle.load(load_file)
        test_build_ids = pickle.load(load_file)
    
    train_data = project [ project['tr_build_id'].isin(train_build_ids)]
    test_data = project [ project['tr_build_id'].isin(test_build_ids)]
    
    train_result = train_data['tr_status'].tolist()
    test_result = test_data['tr_status'].tolist()
    
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
    for i in range(100):
        print('Bootstrapping {} for {}'.format(i, p_name))

        file_name = 'dump_data/rq2_' + p_name + '_model_' + str(i+1) + '_model.pkl'

        #Ensuring we get a non-zero training or testing sample
        while True:
            print('Here for {} {}'.format(i, p_name))
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

    batch_performance = hybrid_performance(p_name, test_builds, test_result, ci)
    performances['time_reqd'].append(batch_performance[0])
    performances['builds_reqd'].append(batch_performance[1])
    performances['total_delay'].append(batch_performance[2])
    performances['failures_found'].append(batch_performance[3])
    performances['failures_not_found'].append(batch_performance[4])
    
    result_file = 'ml_results/' + p_name.split('.')[0] + '_result.txt'
    fres = open(result_file, 'w+')
    fres.write("Average Time Reqd in {} = {} \n".format(p_name, sum(performances['time_reqd'])/len(performances['time_reqd'])))
    fres.write("Average Builds Reqd in {} = {} \n".format(p_name, sum(performances['builds_reqd'])/len(performances['builds_reqd'])))
    fres.write("Average Total Delay in {} = {} \n".format(p_name, sum(performances['total_delay'])/len(performances['total_delay'])))
    fres.write("Average Failed Identified in {} = {} \n".format(p_name, sum(performances['failures_found'])/len(performances['failures_found'])))
    fres.write("Average Failed Unidentified in {} = {} \n".format(p_name, sum(performances['failures_not_found'])/len(performances['failures_not_found'])))

    print('\n\n\n\n\n')


# In[61]:
from multiprocess import Pool

with Pool(10) as pool:
    pool.map(bootstrapping, project_list)

# jobs = []
# for p_name in project_list[:10]:
    
#     q = multiprocess.Process(target=bootstrapping, args=(p_name,))
#     jobs.append(q)
#     q.start()

# for j in jobs:
#     j.join()


# # In[59]:


# print(len(project_list))


# In[ ]:




