from datetime import timedelta, datetime
import pickle
import os

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class IncrementalLearningModel:
    classifier_name = 'RF'
    classifier = None
    verbose = False
    use_cache = True
    hyper_params = {}
    time = None

    project_name = ''

    df = None
    x = None
    y = None

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    y_pred = []
    y_proba = []

    cache_file_path = ''

    accuracy = 0
    recall = 0
    precision = 0
    f_score = 0

    num_of_learn_days = 30
    num_of_predict_days = 1

    temp = 0

    def __init__(self, project_name, classifier_name, num_of_learn_days, num_of_predict_days, hyper_params=None,
                 **kwargs):

        self.classifier_name = classifier_name
        self.project_name = project_name

        self.y_proba = []
        self.y_pred = []
        self.y_test = []

        self.load_travis_dataset()

        if hyper_params:
            self.hyper_params = hyper_params

        self.set_classifier()

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']

        if 'use_cache' in kwargs:
            self.use_cache = False

        self.num_of_learn_days = num_of_learn_days
        self.num_of_predict_days = num_of_predict_days

        cache_dir = os.path.join(BASE_DIR, 'cache')

        if not os.path.exists(cache_dir):
            os.mkdir(os.path.join(cache_dir))

        self.cache_file_path = '{}/{}-{}-{}-{}-{}.pickle'.format(cache_dir,
                                                                 self.project_name,
                                                                 self.__class__.__name__,
                                                                 self.classifier_name,
                                                                 self.num_of_learn_days,
                                                                 self.num_of_predict_days)
        self.load_travis_dataset()

    @staticmethod
    def add_last_failure_distance(status_list):
        result = []
        d = 0

        for item in status_list:
            result.append(d)

            if item != 'passed':
                d = 0
            else:
                d += 1

        return result

    def load_travis_dataset(self):
        from utils import cols_with_date
        path = '../../data/full_data/' + self.project_name
        self.df = pd.read_csv('{}.csv'.format(path),
                              usecols=cols_with_date)

        path = '../../data/full_data/' + self.project_name 
        self.df_test = pd.read_csv('{}.csv'.format(path),
                              usecols=cols_with_date)
        # print(self.df['gh_build_started_at'])
        #self.df = self.df[self.df['tr_status'] != 'canceled']

        # print(IncrementalLearningModel.add_latest_failure_distance(self.df['tr_status']))
        # print(list(self.df['tr_status']))
        # exit(1)

        # print(self.df.loc[1])
        # exit(1)

        # print(list(self.df['git_num_all_built_commits']))
        # print(list(self.df['gh_num_commits_in_push']))
        # exit(1)

        if 'gh_build_started_at' in self.df:
            self.df['gh_build_started_at'] = pd.to_datetime(self.df['gh_build_started_at'])

    def set_classifier(self):
        if self.classifier_name == 'DT':
            self.classifier = DecisionTreeClassifier(random_state=0, **self.hyper_params)

        elif self.classifier_name == 'RF':
            self.classifier = RandomForestClassifier(random_state=0, **self.hyper_params)

        elif self.classifier_name == 'NB':
            self.classifier = GaussianNB(var_smoothing=0.9, **self.hyper_params)

        elif self.classifier_name == 'MLP':
            self.classifier = MLPClassifier(random_state=0, **self.hyper_params)

        elif self.classifier_name == 'LR':
            # min_max_scaler = MinMaxScaler(feature_range=(0, 10000))
            # min_max_scaler.fit(self.x_train)
            # self.x_train = min_max_scaler.transform(self.x_train)

            self.classifier = LogisticRegression(random_state=0, **self.hyper_params)

        elif self.classifier_name == 'SGD':
            self.classifier = SGDClassifier(random_state=0, **self.hyper_params)

        else:
            raise Exception('Invalid Classifier')

    def load_cache(self):
        if self.use_cache and os.path.isfile(self.cache_file_path):
            with open(self.cache_file_path, 'rb') as f:
                self.y_proba, self.y_pred, self.y_test = pickle.load(f)

    def save_cache(self):
        if self.use_cache:
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump((self.y_proba, self.y_pred, self.y_test), f)

    def calc_scores(self):
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        self.recall = metrics.recall_score(self.y_test, self.y_pred, average=None)
        self.precision = metrics.precision_score(self.y_test, self.y_pred, average=None)
        self.f_score = metrics.f1_score(self.y_test, self.y_pred, average=None)

        if self.verbose:
            self.print_scores()

    def get_scores(self):
        print(self.accuracy)
        return '{:.2f} {:.2f} {:.2f} {:.2f}'.format(self.accuracy, self.precision[0], self.recall[0], self.f_score[0])

    def print_scores(self):
        # report = metrics.classification_report(self.y_test, self.y_pred)
        # print('\n'.join(report.split('\n')[:3]))
        # return metrics.classification_report(self.y_test, self.y_pred, output_dict=True)

        # print('Accuracy: {:.2f}%'.format(self.accuracy * 100))
        # print('Recall: {:.2f}%'.format(self.recall[0] * 100))
        # print('Precision: {:.2f}%'.format(self.precision[0] * 100))

        s = '{:28}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:5}\t{}\t{}\n'.format(
            self.project_name,
            self.classifier_name,
            self.precision[0],
            self.recall[0],
            self.f_score[0],
            len(set(self.y_proba)),
            self.time,
            self.hyper_params)

        print(s)

        with open('classifiers.txt', 'a') as f:
            f.write(s)

        # print()
        #
        # print('Recall: {}'.format(self.recall))
        # print('Precision: {}'.format(self.precision))
        # print('F Score: {}'.format(self.f_score))
        # print()

    @staticmethod
    def split_train_test_cols(df):
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        for i in range(len(y)):
            if y[i] != 'passed':
                y[i] = 'failed'
            # else:
            #     y[i] = 1

        return x, y

    @staticmethod
    def handle_missing_value(x):
        label_encoder_x = LabelEncoder()
        x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

        label_encoder_x = LabelEncoder()
        x[:, 2] = label_encoder_x.fit_transform(x[:, 2])

        label_encoder_x = LabelEncoder()
        x[:, 18] = label_encoder_x.fit_transform(x[:, 18])

        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        imputer = imputer.fit(x)
        return imputer.transform(x)

    def get_predicted_data(self):

        #self.load_cache()
        if len(self.y_proba) and len(self.y_test):
            self.calc_scores()
            return self.y_proba, self.y_test

        # start = datetime.now()

        # min_date = self.df['gh_build_started_at'].min()
        # max_date = self.df['gh_build_started_at'].max()

        # date_range = pd.date_range(
        #     min_date + timedelta(days=self.num_of_learn_days),
        #     max_date - timedelta(days=self.num_of_predict_days),
        #     normalize=True,
        #     freq='{}D'.format(self.num_of_predict_days)
        # )
        # print(date_range)
        
        '''for predict_date in date_range:

            partial_x_train, partial_x_test, partial_y_train, partial_y_test = self.split_train_test_dataset(
                predict_date)
            #print(partial_x_test)

            if self.verbose:
                print(predict_date)

            # if not partial_y_train or len(np.unique(partial_y_train)) < 2:
            #     continue

            if partial_y_train is None:
                continue

            if self.verbose:
                print('Train df length', len(partial_x_train))
                print('Test df length', len(partial_x_test))
                print()

            partial_y_pred, partial_y_proba = self.predict(partial_x_train, partial_x_test, partial_y_train)

            if partial_y_pred is None:
                continue

            self.y_test.extend(partial_y_test)
            self.y_pred.extend(partial_y_pred)
            self.y_proba.extend(partial_y_proba)

        if self.project_name == 'opal':
            predict_date = '2015-03-09 19:40:48'
        elif self.project_name == 'vagrant':
            predict_date = '2015-02-24 18:04:23'
        elif self.project_name == 'metasploit-framework':
            predict_date = '2015-04-30 16:30:54'
        elif self.project_name == 'gradle':
            predict_date = '2015-02-08 18:23:26'
        elif self.project_name == 'geoserver':
            predict_date = '2015-10-20 08:51:16'

        train_df = self.df[(self.df['gh_build_started_at'] < predict_date)]
        del train_df['gh_build_started_at']
        test_df = self.df[(self.df['gh_build_started_at'] >= predict_date)]
        del test_df['gh_build_started_at']'''

        train_df = self.df
        del train_df['gh_build_started_at']
        test_df = self.df_test
        del test_df['gh_build_started_at']

        partial_x_train, partial_y_train = self.split_train_test_cols(train_df)
        partial_x_test, partial_y_test = self.split_train_test_cols(test_df)

        partial_x_train, partial_x_test = self.handle_missing_value_multiple_dataset(partial_x_train, partial_x_test)

        partial_y_pred, partial_y_proba = self.predict(partial_x_train, partial_x_test, partial_y_train)

        self.y_test.extend(partial_y_test)
        self.y_pred.extend(partial_y_pred)
        self.y_proba.extend(partial_y_proba)

        # stop = datetime.now()
        # self.time = stop - start
        # self.calc_scores()
        # self.save_cache()
        # #print(self.y_proba)

        return [], test_df['tr_status'].tolist()

        return self.y_proba, self.y_test

    def split_train_test_dataset(self, predict_date):
        train_lower_bound_date = predict_date - timedelta(days=self.num_of_learn_days)
        train_df = self.df[
            (self.df['gh_build_started_at'] < predict_date) & (self.df['gh_build_started_at'] > train_lower_bound_date)]
        del train_df['gh_build_started_at']

        test_upper_bound_date = predict_date + timedelta(days=self.num_of_predict_days)
        test_df = self.df[
            (self.df['gh_build_started_at'] > predict_date) & (self.df['gh_build_started_at'] < test_upper_bound_date)]

        del test_df['gh_build_started_at']
        print(train_lower_bound_date, test_upper_bound_date)

        if not len(train_df) or not len(test_df):
            return None, None, None, None

        partial_x_train, partial_y_train = self.split_train_test_cols(train_df)
        partial_x_test, partial_y_test = self.split_train_test_cols(test_df)

        partial_x_train, partial_x_test = self.handle_missing_value_multiple_dataset(partial_x_train, partial_x_test)

        return partial_x_train, partial_x_test, partial_y_train, partial_y_test

    def predict(self, partial_x_train, partial_x_test, partial_y_train):
        self.set_classifier()

        while True:
            try:
                self.classifier.fit(partial_x_train, partial_y_train)
                break

            except ValueError as e:

                return None, None
                # print(partial_y_train)
                # print(str(e))
                # if len(partial_y_train) == 1:
                #     return ['failed'], [0.0]
                #
                # if partial_y_train[-1] == 'passed':
                #     partial_y_train[-1] = 'failed'
                # else:
                #     partial_y_train[-1] = 'passed'

        partial_y_pred = self.classifier.predict(partial_x_test)

        if self.classifier_name == 'SGD':
            partial_y_proba = [abs(item) if item < 0 else 0 for item in
                               self.classifier.decision_function(partial_x_test)]

        else:
            partial_y_proba = self.classifier.predict_proba(partial_x_test)[:, 0]

        return partial_y_pred, partial_y_proba

        # print(self.internal_y_proba)
        # print(self.internal_y_pred)
        # print(self.internal_y_test)

        # y_test_failure_rate = np.count_nonzero(self.internal_y_test == 'failed') / len(self.internal_y_test)
        # y_pred_failure_rate = np.count_nonzero(self.internal_y_pred == 'failed') / len(self.internal_y_pred)
        #
        # if y_test_failure_rate != y_pred_failure_rate:
        #     print(len(self.internal_y_test))
        #     print(y_test_failure_rate)
        #     print(y_pred_failure_rate)
        #     print()

    def handle_missing_value_multiple_dataset(self, partial_x_train, partial_x_test):
        train_size = len(partial_x_train)
        x = np.concatenate([partial_x_train, partial_x_test])
        x = self.handle_missing_value(x)
        partial_x_train = x[:train_size, :]
        partial_x_test = x[train_size:, :]
        return partial_x_train, partial_x_test

    def get_y_pred(self):
        return self.y_pred
