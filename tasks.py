# -*- coding: utf-8 -*-
from __future__ import print_function, division
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
import os
from manage import app
from features import create_feature
from readers.kaggle import Corpus
from experiments.base import Experiment
from helpers.file_helpers import stdout_redirector
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def run_experiment(name, train, test, feature_names, classifier):
    training_per = 0.7
    dev_per = 0.2


    parameters = {'estimator__C': [1e-2, 1e-1, 1, 2, 5]}
    corpus = Corpus(training_data_fname=train, test_data_fname=test)
    print("*******************************************************************")

    if classifier == 'LR':
        clf = OneVsRestClassifier(LogisticRegression(class_weight='balanced'), n_jobs=1)
    elif classifier == 'RF':
        clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42, class_weight='balanced')
    elif classifier == "Ada":
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600,learning_rate=1.5)



    feature = create_feature(feature_names)

    feature_output = "-".join(feature_names) if isinstance(feature_names, list) else feature_names

    with open(os.path.join('output/new_outputs', feature_output + '_' + classifier + '.txt'),
              'w') as f:

        with stdout_redirector(f):

            Experiment(name +feature_output +'_'+ classifier, corpus.training_set, corpus.dev_set, corpus.test_set)(classifier=clf,
                                                                                                parameters=parameters,
                                                                                                features=feature,
                                                                                                feature_imp=True,
                                                                                                mistake=os.path.join(
                                                                                                    'output',
                                                                                                    'Mistakes1',
                                                                                                    feature_output + '_' + classifier+'_mistake' + '.csv'))

