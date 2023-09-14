#!/usr/bin/env python
# coding: utf-8

# In[46]:


import csv
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


# In[47]:


X_t = pd.read_csv('jruby_yy_metrics.csv')


# In[48]:


Y_data = pd.read_csv('jruby.csv')
build_status = Y_data['tr_status']
Y_t = []


# In[49]:


for e in X_t['tr_build_id']:
    y_index = list(Y_data['tr_build_id']).index(e)
    if Y_data['tr_status'][y_index] == 'passed':
        Y_t.append(1)
    else:
        Y_t.append(0) 


# In[50]:


KF = KFold(n_splits=8)
num_feature = 11

split_index = math.ceil(len(X_t)*0.7)
X = np.array(X_t[:split_index])
Y = np.array(Y_t[:split_index])

test_input = np.array(X_t[split_index:])
test_output = np.array(Y_t[split_index:])

lower_limit = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
upper_limit = lower_limit[::-1]

best_accuracy = 0
best_train_index = 0
best_test_index = 0
threshold = 0

for t1 in lower_limit:
    print('Threshold ' + str(t1))
    for train_index, test_index in KF.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        X_train = X_train.reshape((int(len(X_train)), num_feature))
        rf = RandomForestClassifier()
        predictor = rf.fit(X_train, Y_train)

        y_res = []
        for index in range(len(X_test)):
            new_build = X_test[index]
            new_build = new_build.reshape((1, num_feature))
            predict_result = predictor.predict_proba(new_build)
            if predict_result[0][1] < t1:
                y_res.append(0)
            else:
                y_res.append(1)

        accuracy = 0
        for i in range(len(y_res)):
            if y_res[i] == Y_test[i]:
                accuracy += 1

        print(accuracy*100/len(y_res))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_train_index = train_index
            best_test_index = test_index
            threshold = t1

#extracting the best model:
X_train, X_test = X[best_train_index], X[best_test_index]
Y_train, Y_test = Y[best_train_index], Y[best_test_index]
X_train = X_train.reshape((int(len(X_train)), num_feature))

rf = RandomForestClassifier()
predictor = rf.fit(X_train, Y_train)
y_res = []
print('The threshold is ' + str(threshold))
for index in range(len(test_input)):
    new_build = test_input[index]
    new_build = new_build.reshape((1, num_feature))
    predict_result = predictor.predict_proba(new_build)
    if predict_result[0][1] > threshold:
        y_res.append('group')
    else:
        y_res.append('skip')
#X = X.reshape((int(len(X_train)), num_feature))
#predictor = rf.fit(X, Y)


# In[ ]:





# In[8]:


print(y_res)


# In[59]:


print('The threshold is ' + str(threshold))
threshold = 0.4
y_res = []
for index in range(len(test_input)):
    new_build = test_input[index]
    new_build = new_build.reshape((1, num_feature))
    predict_result = predictor.predict_proba(new_build)
    if predict_result[0][1] > threshold:
        y_res.append('group')
    else:
        y_res.append('skip')
#X = X.reshape((int(len(X_train)), num_feature))
#predictor = rf.fit(X, Y)


# In[60]:


print(y_res)


# In[ ]:




