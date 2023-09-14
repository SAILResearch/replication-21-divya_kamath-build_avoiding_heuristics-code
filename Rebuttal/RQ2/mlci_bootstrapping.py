#!/usr/bin/env python
# coding: utf-8

def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0) 
    return Y_t



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





def bootstrapping(train_data, count):
    
    
    #grid search hyperparameters
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    
    #setting up grid search
    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
    forest = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = forest, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 0)
    
   
        
        
    train_result = train_data['tr_status'].tolist()
    train_data['num_of_passes'] = get_pass_streak(train_result)

    best_n_estimators = []
    best_max_depth = []

    best_f1 = 0
    best_f1_sample = 0
    best_f1_sample_result = 0
    best_f1_estimator = 0
    best_thresholds = []


    #bootstrap 100 times
    for i in range(1):
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

        file_name = 'dump_data/rq2_' + p_name + '_' + str(count) + '_best_model.pkl'
        dump_file = open(file_name, 'wb')
        pickle.dump(forest, dump_file)
        pickle.dump(threshold, dump_file)
        pickle.dump(n_estimator, dump_file)
        pickle.dump(max_depth, dump_file)
        
        
        return forest
        
    

