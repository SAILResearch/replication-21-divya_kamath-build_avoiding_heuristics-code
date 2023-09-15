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
from batching_algs import *
from mlci_bootstrapping import *
warnings.filterwarnings("ignore")


project_list = ['heroku.csv', 'rails.csv', 'gradle.csv', 'jruby.csv', 'metasploit-framework.csv', 'cloudify.csv', 'vagrant.csv', 'rubinius.csv', 'open-build-service.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'opal.csv', 'cloud_controller_ng.csv', 'puppet.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv']



r_file_name = 'all_rq3_results.csv'
result_file = open(r_file_name, 'w')
result_headers = ['version', 'project', 'algorithm', 'batch_size', 'builds_reqd', 'sbs_delays', 'failures_found', 'failures_not_found', 'bad_builds', 'batch_delays', 'testall_size', 'ci']
writer = csv.writer(result_file)
writer.writerow(result_headers)



batch_total = 0
batch_duration = 0



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



for p in project_list[:]:
    for i in range(1, 11):
        mlci_process(p, i)