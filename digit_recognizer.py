#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from pandas import read_csv, DataFrame, Series, concat
from sklearn import cross_validation, svm, grid_search

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score

import pylab as pl
import matplotlib.pyplot as plt

def get_train_data():

    print 'Get train data...'
    data = read_csv('./train.csv')
    return data

def get_test_data():

    print 'Get test data...'
    data = read_csv('./test.csv')

    return data

# final function
def go():
    data = get_train_data()

    model_rfc = RandomForestClassifier(n_estimators = 1024, criterion = 'entropy', n_jobs = -1)

    print 'Go!!!'

    print 'RFC...'
    test = get_test_data()
    target = data.label
    train = data.drop(['label'], axis = 1)

    print "..."
    result = DataFrame()
    model_rfc.fit(train, target)
    result['ImageId'] = range(1, len(test) + 1)
    result.insert(1,'Label', model_rfc.predict(test))
    result.to_csv('./test_rfc_1024.csv', index=False)

def grid_search_test():
    data = get_train_data()
    target = data.label
    train = data.drop(['label'], axis = 1)

    model_rfc = RandomForestClassifier()
    params = {"n_estimators" : [100, 250, 500, 625], "criterion" : ('entropy', 'gini')}

    clf = grid_search.GridSearchCV(model_rfc, params)
    clf.fit(train, target)

    # summarize the results of the grid search
    print(clf.best_score_)
    print(clf.best_estimator_.criterion)
    print(clf.best_estimator_.n_estimators)

go()