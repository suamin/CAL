# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid


def build_models(model, param_grid):
    return [model(**p) for p in ParameterGrid(param_grid)]


def build_sgd_classifiers(random_state=None):
    param_grid = {
        'loss': ['hinge'],
        'penalty': ['l2'],
        'fit_intercept': [False],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'optimal'],
        'n_iter': [100, 500, 1000],
        'eta0': [0.001, 0.01, 0.1]
    }
    return build_models(SGDClassifier, param_grid)


def default_model():
    sgd_svm_opts = dict(
        loss='hinge', penalty='l2', alpha=0.0001,
        fit_intercept=False, tol=1e-3, shuffle=True,
        random_state=2018, learning_rate='optimal',
        class_weight='balanced'
    )
    return SGDClassifier(**sgd_svm_opts)
