# -*- coding: utf-8 -*-

import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import plots


def predicted_label_transform(y_pred):
    # transform unbalanced binary (0, 1) clustering output, lower count = 1, high count = 0
    c1, c2 = set(y_pred)
    c1_len = sum(1 for i in y_pred if i == c1)
    c2_len = sum(1 for i in y_pred if i == c2)
    if c1_len > c2_len:
        c1_new = 0
        c2_new = 1
    else:
        c1_new = 1
        c2_new = 0
    y_pred = [c1_new if i == c1 else c2_new for i in y_pred]
    return y_pred


def LDDS(X, y=None, two_phase=True, size=1000, n_components=None, n_iter=None, random_state=None, show=True):
    """Low-dimensional unbalanced density sampling"""
    if X.shape[0] <= size:
        return np.arange(X.shape[0])
    
    if not isinstance(y, np.ndarray):
        y = np.array([None] * X.shape[0])
        labels = False
    else:
        labels = True
    
    if n_components:
        try:
            assert n_components < X.shape[1]
        except:
            raise ValueError('`n_components` should be less than %d' % X.shape[1])
    
    # set numpy random seed
    if random_state:
        random_state = np.random.RandomState(random_state)
    else:
        random_state = np.random.RandomState(np.random.randint(1111, 9999))
    
    if n_iter == None:
        n_iter = np.inf
    
    output = list()
    iids = np.arange(X.shape[0]) # input ids
    i = 1 # iterations count
    
    while True:
        sample_from = iids[iids!=-1]
        if len(sample_from) < size or i > n_iter:
            break
        rids = random_state.choice(sample_from, size=size, replace=False, p=None) # randomly sampled ids
        mask = np.array([True if i in rids else False for i in range(X.shape[0])])
        
        # make selection
        Xs = X[mask, :]
        if labels:
            ys = y[mask]
        iids = np.array([-1 if i else iids[idx] for idx, i in enumerate(mask)])
        
        if n_components:
            k = n_components
        else:
            k = random_state.randint(2, int(Xs.shape[1] * 0.25))
        
        # reduce to k-dimensions
        Xs = PCA(n_components=k, svd_solver='randomized', random_state=random_state).fit_transform(Xs)
        
        if two_phase:
            # perform agglomerative clustering
            aggc = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='average').fit(Xs)
            y_pred = predicted_label_transform(aggc.labels_)
            
            mask = np.array([True if i == 1 else False for i in y_pred])
            Xs = Xs[mask, :]
            if labels:
                ys = ys[mask]
            rids = rids[mask]
        
        kmeans = KMeans(n_clusters=2, random_state=random_state).fit(Xs)
        y_pred = predicted_label_transform(kmeans.labels_)
        
        mask = np.array([True if i == 1 else False for i in y_pred])
        Xs = Xs[mask, :]
        y_analysis = None
        if labels:
            ys = ys[mask]
            # analyze true labels, predictions and random sampling
            num_relevant_predicted = sum(ys)
            num_docs_to_review = len(ys)
            y_analysis = (num_docs_to_review, {'cluster':num_relevant_predicted , 'random':sum(random_state.choice(y, num_docs_to_review))})
        
        rids = rids[mask]
        
        output.append((rids, y_analysis))
        i += 1
    
    if output and show:
        plots.plot_LDDS(output)
    
    return output
