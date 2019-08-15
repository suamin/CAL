# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import learning_curve
from metrics import auc


def plot_document_features(feature_score_contribs, feature_names, pred_class):
    colors = ['red' if v < 0. else 'blue' for v in feature_score_contribs]
    x = np.arange(len(feature_names))
    plt.figure(figsize=(15, 5))
    plt.bar(x, feature_score_contribs, color=colors)
    plt.xticks(x, feature_names, rotation=45, ha='right', color='green')
    plt.title('Features contribution in class prediction as {}'.format(pred_class))
    plt.show()


def plot_clf_top_coefs(top_pos, top_neg, feature_names, topn=10):
    # taken and modified from:
    # https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
    pos_coefs, pos_names = zip(*top_pos)
    neg_coefs, neg_names = zip(*top_neg)
    
    colors = ['red'] * len(neg_coefs) + ['blue'] * len(pos_coefs)
    top_coefs = np.hstack([neg_coefs, pos_coefs])
    top_names = np.hstack([neg_names, pos_names])
    
    # create plot
    plt.figure(figsize=(15, 5))
    plt.bar(np.arange(2 * topn), top_coefs, color=colors)
    plt.xticks(np.arange(2 * topn), top_names, rotation=60, ha='right')
    plt.title("Top positive and negative features of classifier")
    plt.show()


def pr_curve(y_true, scores, stage, show_var=False):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    y_pred = (scores > 0).astype(np.int)
    auc_val = auc(y_true, y_pred)
    plt.figure(figsize=(20, 10))
    plt.plot(recall, precision, color='r')
    if show_var:
        precision_std = np.std(precision)
        precision_upper = precision + precision_std
        precision_lower = precision - precision_std
        plt.fill_between(recall, precision_upper, precision_lower, color='r', alpha=0.1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('%s Precision-Recall curve: AP=%0.2f' % (stage.title(), auc_val))
    plt.grid("on") 
    plt.show()


def plot_cal_metrics(precisions, recalls, accuracies, gmeans, aucs, xs, stage):
    for score, vals in [('precision', precisions), ('recall', recalls), ('accuracy', accuracies), ('g-mean', gmeans), ('ROC-AUC', aucs)]:
        plt.figure(figsize=(15, 10))
        plt.plot(xs, vals, color='red')
        plt.xlabel('number of queries')
        plt.ylabel(score)
        plt.title('{} data metrics'.format(stage.title()))
        plt.show()


def plot_LDDS(ldds_output):
    _, y_analysis = zip(*ldds_output)
    c_success = list()
    r_success = list()
    n = list()
    for value in y_analysis:
        c_success.append(value[1]['cluster'])
        r_success.append(value[1]['random'])
        n.append(value[0])
    c_success = np.divide(c_success, n) * 100
    r_success = np.divide(r_success, n) * 100
    c_mean = np.array([np.mean(c_success)] * len(n))
    r_mean = np.array([np.mean(r_success)] * len(n))
    plt.figure(figsize=(20, 10))
    plt.scatter(n, r_success, c='r', marker='o', label='random')
    plt.scatter(n, c_success, c='b', marker='+', label='cluster')
    plt.plot(n, r_mean, 'r', label='random avg.', linewidth=0.5)
    plt.plot(n, c_mean, 'b', label='cluster avg.', linewidth=0.5)
    plt.legend()
    plt.xlabel('number of documents to review')
    plt.ylabel('success rate (%)')
    plt.title('Low-dimensional Density Projection Samplings')
    plt.show()


def plot_drd(hist_arr):
    plt.figure(figsize=(20, 10))
    plt.hist(hist_arr, rwidth=0.25, stacked=True, label=['1-Coded-R', '0-Coded-R', '1-Predicted-NR', '0-Predicted-NR'])
    plt.legend()
    plt.ylabel('number of documents')
    plt.xlabel('relevance rank score')
    plt.title('Document Rank Distribution')
    plt.show()


def plot_prp(xs, ys):
    plt.figure(figsize=(20, 10))
    plt.plot(xs, ys, '-o')
    plt.xlabel('number of documents reviewed')
    plt.ylabel('relevance rate')
    plt.title('Prioritized Review Progress')
    plt.show()


def plot_review_pr_curve(precision, recall, k):
    plt.figure(figsize=(20, 10))
    plt.plot(recall, precision, color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Review Stage Precision vs. Recall @ {}'.format(k))
    plt.show()


def plot_rs_prevalence(rs_output, successes):
    rs_output = sorted(rs_output, key=lambda x: x[0])
    ys, _, _, xs = list(zip(*rs_output))
    plt.figure(figsize=(20, 10))
    plt.scatter(xs, [100*yi for yi in ys], label='at least {} relevant'.format(successes))
    plt.xlabel('number of random samples')
    plt.ylabel('prevalence (%)')
    plt.title("Random Samping vs. Prevalence Estimates")
    plt.legend()
    plt.show()


# scikit-learn script:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring='accuracy'):
    """
    Generate a simple plot of the test and training learning curve.
    
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    
    title : string
        Title for the chart.
    
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure(figsize=(8, 5))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt