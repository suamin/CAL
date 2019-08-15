# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics

# supervised metrics

def precision(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred)


def recall(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred)

sensitivity = recall

def accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)


def specifity(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, pos_label=0)


def g_means(y_true, y_pred):
    """
    g-means is a commonly used metric for measuring performance
    of imbalanced datasets [1].
    
    References
    ----------
    .. [1] Ertekin, Seyda, et al. "Learning on the border: active 
           learning in imbalanced data classification." Proceedings 
           of the sixteenth ACM conference on Conference on 
           information and knowledge management. ACM, 2007.
    """
    return np.sqrt(sensitivity(y_true, y_pred) * specifity(y_true, y_pred))


def auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)

user_machine_agreement = metrics.cohen_kappa_score
