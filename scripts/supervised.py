# -*- coding: utf-8 -*-

import os
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import models


def cv_train(clf, X_train, y_train, n_splits=10):
    cv = StratifiedKFold(n_splits=n_splits)
    cv_accuracies = list()
    for train, test in cv.split(X_train, y_train):
        clf.fit(X_train[train], y_train[train])
        preds = clf.predict(X_train[test])
        acc = accuracy_score(y_train[test], preds)
        cv_accuracies.append(acc)
    return cv_accuracies


def eval_test(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    return acc, clf_report


def pipe(clf, X_train, X_test, y_train, y_test):
    print("training classifier with %s fold cross-validation..." % n_splits)
    train_acc = cv_train(clf, X_train, y_train, n_splits)
    
    print("mean training accuracy: {}".format( (sum(train_acc) / n_splits) * 100 ))
    print("evaluating test set...")
    test_acc, test_clf_report = eval_test(clf, X_test, y_test)
    
    print("test accuracy: {}".format(test_acc))
    print("classification report:")
    print(test_clf_report)


def main(X_train, X_test, y_train, y_test):
    model = models.default_model()
    pipe(model)
