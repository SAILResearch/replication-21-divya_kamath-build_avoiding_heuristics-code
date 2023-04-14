#!/usr/bin/env python
# coding: utf-8

# In[38]:


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
import multiprocess
warnings.filterwarnings("ignore")


# In[39]:


project_list = ['heroku.csv', 'rails.csv', 'gradle.csv', 'jruby.csv', 'metasploit-framework.csv', 'cloudify.csv', 'vagrant.csv', 'rubinius.csv', 'open-build-service.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'opal.csv', 'cloud_controller_ng.csv', 'puppet.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv']

# In[40]:


r_file_name = 'all_rq3_results.csv'
result_file = open(r_file_name, 'w')
result_headers = ['version', 'project', 'algorithm', 'batch_size', 'builds_reqd', 'sbs_delays', 'failures_found', 'failures_not_found', 'bad_builds', 'batch_delays', 'testall_size', 'ci']
writer = csv.writer(result_file)
writer.writerow(result_headers)


# In[41]:


def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0) 
    return Y_t


# In[42]:


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


# In[43]:


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


# In[44]:


def get_complete_data(p_name, first_failures=True):
    
    #open the metrics file
    filename = 'project_metrics/' + p_name.split('.')[0] + '_metrics.csv'
    project = pd.read_csv(filename)
    project = project.drop(project.columns[9], axis=1)
    project['tr_status'] = output_values(project['tr_status'])
    if first_failures:
        project = get_first_failures(project)
    return project




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
    
    return (sum(delay), failures_found, failures_not_found, bad_builds)


# In[47]:


batch_total = 0
batch_duration = 0


# In[48]:


def batch_bisect(batch_results):
    global batch_total
    
    batch_total += 1
    
    if len(batch_results) == 1:
        return
    
    if 0 in batch_results:
        half_batch = len(batch_results)//2
        batch_bisect(batch_results[:half_batch])
        batch_bisect(batch_results[half_batch:])


# In[49]:


def batch_stop_4(batch_results):
    global batch_total
    
    batch_total += 1
    
    if len(batch_results) <= 4:
        if 0 in batch_results:
            batch_total += 4
        return
    
    if 0 in batch_results:
        half_batch = len(batch_results)//2
        batch_stop_4(batch_results[:half_batch])
        batch_stop_4(batch_results[half_batch:])


# In[50]:


def bootstrapping(p_name, ver):
    
    #This will return the entire dataset with X and Y values
    project = get_complete_data(p_name)
    
    #grid search hyperparameters
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    
    #setting up grid search
    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
    forest = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = forest, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 0)
    
    pkl_file = '../data/project_data_pickles/' + p_name + '_' + str(ver) + '_indexes.pkl'
    with open(pkl_file, 'rb') as load_file:
        train_build_ids = pickle.load(load_file)
        test_build_ids = pickle.load(load_file)
    
    train_data = project [ project['tr_build_id'].isin(train_build_ids)]
    test_data = project [ project['tr_build_id'].isin(test_build_ids)]
    
    train_result = train_data['tr_status'].tolist()
    test_result = test_data['tr_status'].tolist()
    
    if len(test_result) == 0:
        return 
    #add pass_streak to training data:
    train_data['num_of_passes'] = get_pass_streak(train_result)
    
    best_n_estimators = []
    best_max_depth = []

    best_f1 = 0
    best_f1_sample = 0
    best_f1_sample_result = 0
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

    file_name = 'dump_data/rq3_' + p_name + '_' + str(ver) + '_best_model.pkl'
    dump_file = open(file_name, 'wb')
    pickle.dump(forest, dump_file)
    pickle.dump(threshold, dump_file)
    pickle.dump(n_estimator, dump_file)
    pickle.dump(max_depth, dump_file)
    
    

def mlci_process(p_name, ver):
    
    global batch_total
    global batch_duration
    
    p = p_name.split('.')[0]
    
    print('Processing {}'.format(p_name))

    #result_file_name = p + '_rq2_results.csv'
    result_file = open('all_rq3_results.csv', 'a+')
    # writer = csv.writer(result_file)
    
    project = get_complete_data(p_name, first_failures=False)
    pkl_file = '../data/project_data_pickles/' + p_name + '_' + str(ver) + '_indexes.pkl'
    with open(pkl_file, 'rb') as load_file:
        train_build_ids = pickle.load(load_file)
        test_build_ids = pickle.load(load_file)
    
    test_data = project [ project['tr_build_id'].isin(test_build_ids)]
    test_result = test_data['tr_status'].tolist()
    
    if len(test_result) == 0:
        return 

    test_builds = test_data['tr_build_id'].tolist()
    test_data.drop('tr_build_id', inplace=True, axis=1)
    test_data.drop('tr_status', inplace=True, axis=1)
    
    
    file_name = '../../RQ3-Models/' + p + '_models/' + 'rq3_' + p_name + '_' + str(ver) + '_best_model.pkl'
    load_file = open(file_name, 'rb')
    forest = pickle.load(load_file)
    threshold = pickle.load(load_file)
    

    batchsizelist = [2, 4, 8, 16]
    algorithms = ['BATCH4', 'BATCHSTOP4', 'BATCHBISECT']

    batch_delays = 0
    final_pred_result = []
    
    for alg in algorithms:
        for batchsize in batchsizelist:

            batch_delays = 0
            
            pass_streak = 0
            i = 0
            total = len(test_data)
            num_of_builds = 0
            build_duration = 0

            #The variable 'ci' will hold the actual execution process of the current phase
            #If ci[i] == 0, it means that build was made
            #If ci[i] == 1, it means that build was saved
            ci = []

            if alg == 'NOBATCH':
                while i < total:
                    data = test_data.iloc[i]
                    data['num_of_passes'] = pass_streak
                    predict = grid_search.predict_proba([data])

                    if predict[0][1] > threshold:
                        final_pred_result.append(1)
                        ci.append(1)
                        pass_streak += 1
                        i += 1
                    else:

                        while i < total:
                            if (total - i) > batchsize:
                                ci.extend([0 for i in range(batchsize)])
                            else:
                                ci.extend([0 for e in range(total-i)])

                            batch_delays += (batchsize - 1)*batchsize*0.5

                            batch_build_times = durations[i:i+4]
                            actual_batch_results = test_result[i:i+4]

                            num_of_builds += 1
                            build_duration += max(batch_build_times)

                            if 0 in actual_batch_results:
                                i = i+batchsize
                            else:
                                break
                        i += batchsize
                        pass_streak = 1

                
            if alg == 'BATCH4':
                if batchsize != 4:
                    continue
                else:
                        while i < total :
                            data = test_data.iloc[i]
                            data['num_of_passes'] = pass_streak
                            predict = forest.predict_proba([data])

                            #predicted that build has passed
                            if predict[0][1] > threshold:
                                final_pred_result.append(1)
                                ci.append(1)
                                pass_streak += 1
                                i+=1

                            else:
                                #We found first failure

                                #Until an entire batch passes, we are going to continue group builds ie., subsequent failures are grouped
                                while i < total:
                                    if (total - i) > 4:
                                        ci.extend([0,0,0,0])
                                    else:
                                        ci.extend([0 for e in range(total-i)])

                                    batch_delays += (batchsize - 1)*batchsize*0.5

                                    actual_batch_results = test_result[i:i+4]
                                    
                                    num_of_builds += 1                                    
                                        # print(batch_build_times)
                                        # print(durations)
                                        # print(test_result)

                                    #if any build has failed in the batch, then whole batch will fail
                                    if 0 in actual_batch_results:
                                        i = i+4
                                        num_of_builds += 4
                                    else:
                                        break
                                #Now that we have found a passing build, we can update pass_streak to 1
                                pass_streak = 1
                                i += 4
            
            if alg == 'BATCHSTOP4':
                if batchsize < 4:
                    continue
                else:
                    pass_streak = 0
                    ci = []
                    while i < total :
                        data = test_data.iloc[i]
                        data['num_of_passes'] = pass_streak
                        predict = forest.predict_proba([data])
                        
                        if predict[0][1] > threshold:
                            ci.append(1)
                            pass_streak += 1
                            i += 1
                        else:
                            
                            while i < total:
                                if (total - i) > batchsize:
                                    ci.extend([0 for l in range(batchsize)])
                                else:
                                    ci.extend([0 for e in range(total-i)])


                                batch_delays += (batchsize - 1)*batchsize*0.5
                                
                                grouped_batch_results = test_result[i:i+batchsize]
                                batch_total = 0
                                
                                batch_stop_4(grouped_batch_results)
                                num_of_builds += batch_total
                                
                                if 0 not in grouped_batch_results:
                                    break
                                else:
                                    i += batchsize
                                grouped_batch_results.clear()
                            i += batchsize
                            pass_streak = 1
            
            if alg == 'BATCHBISECT':
                
                pass_streak = 0
                ci = []
                
                while i < total :
                    data = test_data.iloc[i]
                    data['num_of_passes'] = pass_streak
                    predict = forest.predict_proba([data])
                    
                    if predict[0][1] > threshold:
                        ci.append(1)
                        pass_streak += 1
                        i += 1
                    else:
                        
                        #this case is when model has predicted a failure
                        #Add [i, i+batchsize] to a group and perform BatchBisect
                        
                        while i < total:
                            
                            #Next batch is being built, so add to ci
                            if (total - i) >= batchsize:
                                ci.extend([0 for l in range(batchsize)])
                            else:
                                ci.extend([0 for e in range(total-i)])
                            
                            batch_delays += (batchsize - 1)*batchsize*0.5

                            grouped_batch_results = test_result[i:i+batchsize]
                            
                            batch_total = 0
                            
                            batch_bisect(grouped_batch_results)
                            num_of_builds += batch_total
                            
                            if 0 not in grouped_batch_results:
                                break
                            else:
                                i += batchsize
                                
                            grouped_batch_results.clear()
                        i += batchsize
                        pass_streak = 1


            batch_performance = hybrid_performance(p_name, test_builds, test_result, batchsize, ci)
            total_delay = batch_performance[0]
            failures_found = batch_performance[1]
            failures_not_found = batch_performance[2]
            bad_builds = batch_performance[3]

            local_builds_reqd = 100*num_of_builds/total
            
            print([ver, p_name, alg, batchsize, local_builds_reqd, total_delay, failures_found, failures_not_found, bad_builds, batch_delays, total, ci])
            writer.writerow([ver, p_name, alg, batchsize, local_builds_reqd, total_delay, failures_found, failures_not_found, bad_builds, batch_delays, total, ci])
            if total != len(ci):
                print('PROBLEM!!')
            else:
                print('NO PROBLEM!!')



# In[51]:
#bootstrapping('cloudify.csv')

for p in project_list[:]:
    for i in range(1, 11):
        mlci_process(p, i)


# In[53]:

#jobs = []
#for p_name in project_list[9:]:
    
#    q = multiprocess.Process(target=bootstrapping, args=(p_name,))
#    jobs.append(q)
#    q.start()

#for j in jobs:
#    j.join()


# if __name__ == '__main__':
# 	with multiprocess.Pool(5) as p:
# 		p.map(bootstrapping, project_list[:])



# In[12]:


# print(len(project_list))


# In[13]:





# In[14]:





# In[15]:





# In[16]:





# In[17]:





# In[20]:
