# -*- coding: utf-8 -*-
from __future__ import print_function
import logging
from sklearn import metrics
import csv

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scipy.sparse as sp
from analysis import feature_imp
import analysis

# Set up logger
log = logging.getLogger(__name__)


class Experiment(object):
    def __init__(self, experiment_name, training_set=None, dev_set=None, test_set=None):
        """
        initialize the experiment
        initializes the name,  training, dev and test data
        """
        self._name = experiment_name
        self._training_set = training_set
        self._dev_set = dev_set
        self._test_set = test_set


    def get_name(self):
        return self._name

    def __call__(self, **kwargs):
        """
        template method
        train and then test on test instances
        finally prints the evaluation report
        """

        try:

            model, features_obj, le, feature_name= self.train(**kwargs)

            X_test, y_test = self._test_set.instances, self._test_set.labels

            X_test_features = features_obj.transform(X_test)

            y_test_labels = le.transform(y_test)

            print()
            print("[INFO] Test Set  instances %d and labels %d" % (
                X_test.shape[0], y_test.shape[0]))

            y_predicted = model.predict(X_test_features)
            score = model.score(X_test_features, y_test_labels)

            # test_predicted = []
            # for i in range(0, len(X_test)):
            #     if y_predicted[i] == 0:
            #         predl = 'cag'
            #     elif y_predicted[i] == 1:
            #         predl = 'nag'
            #     else:
            #         predl = 'oag'
            #     doc = [X_test[i].id, predl]
            #     test_predicted.append(doc)
            #
            # new_path = "spc_agg_agr_en_fb_test.csv"
            # with open(new_path, mode='w') as csvfile:
            #     writer = csv.writer(csvfile, lineterminator='\n')
            #     writer.writerows(test_predicted)


            self.report(le, y_test_labels, y_predicted)

            fea_imp=kwargs.get('feature_imp',False)
            if fea_imp:
                feature_imp.show_most_informative_features(features_obj,model,n=50)

            mistakes=kwargs.get('mistake',None)
            if mistakes:
                analysis.show_mistakes(mistakes,X_test,y_test_labels,y_predicted)


        except Exception as ex:
            log.error('Error running the experiment. {error}s'.format(
                      error=ex))
            return False
        else:
            self.post_process(model, features_obj, y_predicted)

        return score

    # TODO for cases that don't have dev set, cv for training set depending on the parameter
    def train(self, **kwargs):
        # get the feature object and the classifier
        print ("Experiment: %s"%self.get_name())
        feature_name, feature_obj = kwargs.get('features', (None, None))
        clf = kwargs.get('classifier', None)
        parameters = kwargs.get('parameters', {})
        class_le = LabelEncoder()


        print ("Feature: %s"%feature_name)


        if self._dev_set:
            X_train, X_dev = self._training_set.instances, self._dev_set.instances


            y_train, y_dev = class_le.fit_transform(self._training_set.labels), class_le.transform(self._dev_set.labels)

            # concatenate the train and dev
            X_train_all, y_train_all = np.hstack((X_train, X_dev)), np.hstack((y_train, y_dev))


            print("[INFO] Training instances %d and labels %d" % (X_train.shape[0], y_train.shape[0]))
            print("[INFO] Classes are : %s" % ", ".join(list(class_le.classes_)))
            print("[INFO] Validation set instances %d and labels %d" % (X_dev.shape[0], y_dev.shape[0]))


            # perform grid search over the parameter values and find the best classifier on the dev set
            print(X_train)
            print(y_train)
            X_train_features = feature_obj.fit_transform(X_train, y_train)

            X_dev_features = feature_obj.transform(X_dev)

            print ("Shape of X_train: {}".format(X_train_features.shape))
            print ("Shape of X_dev: {}".format(X_dev_features.shape))

            if sp.issparse(X_train_features):
                X_train_all_features = sp.vstack((X_train_features, X_dev_features))

            else:

                X_train_all_features = np.vstack((X_train_features, X_dev_features))


            print ("Shape of X_train + X_dev: {}".format(X_train_all_features.shape))

            # train with best parameter
            X_train_features = feature_obj.fit_transform(X_train_all, y_train_all)
            best_clf = clf.fit(X_train_features, y_train_all)

            print("[INFO] Training + Validations set instances %d and labels %d" % (
                X_train_features.shape[0], y_train_all.shape[0]))

            print("Training Accuracy =%.3f" % best_clf.score(X_train_features, y_train_all))

            return best_clf, feature_obj, class_le, feature_name

        else:
            X_train, X_test = self._training_set.instances, self._test_set.instances
            y_train, y_test = class_le.fit_transform(self._training_set.labels), class_le.transform(self._test_set.labels)


    def report(self, le, y_test, y_pred):
        """
        prints the precision, recall, f-score
        print confusion matrix
        print accuracy
        """


        print('---------------------------------------------------------')
        print()
        print("Classifation Report")
        print()

        target_names = le.classes_
        class_indices = {cls: idx for idx, cls in enumerate(le.classes_)}

        print(metrics.classification_report(y_test, y_pred, target_names=target_names,
                                            labels=[class_indices[cls] for cls in target_names]))

        print("============================================================")
        print("Confusion matrix")
        print("============================================================")
        print(target_names)
        print()
        print(confusion_matrix(
            y_test,
            y_pred,
            labels=[class_indices[cls] for cls in target_names]))

        print()

        precisions_micro, recalls_micro, fscore_micro, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                           average='micro',
                                                                                           pos_label=None)
        precisions_macro, recalls_macro, fscore_macro, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                           average='macro',
                                                                                           pos_label=None)
        precisions_weighted, recalls_weighted, fscore_weighted, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                                    average='weighted',
                                                                                                    pos_label=None)

        print()
        print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))


        print("Macro Precision Score, %f, Micro Precision Score, %f, Weighted Precision Score, %f" % (
            precisions_macro, precisions_micro, precisions_weighted))

        print("Macro Recall score, %f, Micro Recall Score, %f, Weighted Recall Score, %f" % (
            recalls_macro, recalls_micro, recalls_weighted))

        print("Macro F1-score, %f, Micro F1-Score, %f, Weighted F1-Score, %f" % (
            fscore_macro, fscore_micro, fscore_weighted))

        print('Misclassified samples: %d' % (y_test != y_pred).sum())
        print("============================================================")



    def post_process(self, model, features, y_predicted):
        pass
