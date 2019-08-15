# -*- coding: utf-8 -*-

import numpy as np
import os

from al import Dataset, UncertaintySampling

import plots
import metrics
import utils

from samples import random_uniform_sample, clopper_pearson_ci


class ActiveSimulation:
    
    def __init__(self):
        self.metrics = {'train':[], 'valid':[], 'simulate': []}
        self.est_prevelance = None
        self.est_relevant = None
        self.relevant_found = None
        self.ci = None
    
    def random_seed_collection(self, total, n=10):
        y_seed = list()
        drawn = random_uniform_sample(self.dataset.y_ideal, n)
        for i in range(len(self.dataset.y_ideal)):
            if i in drawn[0]:
                y_seed.append(0)
            elif i in drawn[1]:
                y_seed.append(1)
            else:
                y_seed.append(None)
        
        num_draws = len(drawn[0]) + len(drawn[1])
        self.ci = clopper_pearson_ci(n, num_draws)
        self.est_prevelance = n / num_draws
        self.est_relevant = int(self.est_prevelance * total)
        
        return drawn, y_seed
    
    def initialize(self, X, model, y=None, query='US', mode='simulate', y_ideal=None, num_relevant_seed=10,
                   progressive_validation=True, pr_rate=5, **kwargs):
        """Initializes the state of CAL.
        
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Features matrix.
        model : sklearn object type that implements the "fit" and "predict" methods
            Classification model.
        y : array_like, shape (n_samples,), optional
            Class labels, if partially labeled the corresponding examples will be
            considered as seed documents.
        query : string, optional
            Name of active learning algorithm, could be one `al.queries`
        mode : string, optional
            Mode of active learning, can be one of 'simulate' or 'active'.
            In latter, the learning will be done interactively by requesting labels
            from user.
        y_ideal : array_like, shape (n_samples,), optional
            Noiseless labels corresponding to each sample in ``X``, required when
            ``mode`` is 'simulate'.
        num_relevant_seed : int, optional
            How many relevant documents to sample for initial seeding.
            In a purely 'active' mode this will be performed by user.
            For enhanced chance to sample relevant documents see :meth:`ldds.LDDS`.
        progressive_validation : bool, optional
            Whether to allow collection of validation examples from user labeled
            examples as a measure to reflect classifier's generalization ability.
            Right now, it is set as an optional parameter but ideally it should be
            imposed to get an idea of classifier performance without compromising
            too much on performance or enhacing label complexity. It can be argued
            that PV may not reflect true loss but it is a good estimate since we
            don't have access to all labels.
            It can also be used for hyperparameter tuning but then using it as loss
            estimate will be wrong.
        pr_rate : int, optional
            Frequency at which to collect validation examples.
            For example, a value of 5 means that every 5th label we query from user
            it will be assigned to validation set no matter positive or negative.
        docid2path : dict, optional
            A mapping from example indexes to text file paths, will be only ued in
            'active' mode to show user the text and collect label.
        eval_at : int, optional
            Frequency at which we evaluation happens, too often can decreasing
            performance and efficiency with models of high complexity.
        
        """
        self.dataset = Dataset(X, y, mode, y_ideal)
        if mode == 'simulate':
            drawn, _ = self.random_seed_collection(self.dataset.X.shape[0], num_relevant_seed)
            for label, indexes in drawn.items():
                for index in indexes:
                    self.dataset.index2props[index].y_true = label
                    self.dataset.index2props[index].is_seed = True
                    self.dataset.index2props[index].is_train = True
                    self.dataset.index2props[index].state = 1
        if query == 'US':
            self.qs = UncertaintySampling()
        else:
            raise NotImplementedError('currently only Uncertainty Sampling query is supported')
        self.model = model
        self.progressive_validation = progressive_validation
        self.pr_rate = pr_rate
        self.docid2path = kwargs.get('docid2path', None)
        self.eval_at = kwargs.get('eval_at', 10)
        self.eval_xs = {'train':[], 'valid':[], 'simulate':[]}
    
    def supervised_eval(self, train_or_valid):
        data = self.dataset.get_labeled_data(train_or_valid)
        if data == None:
            raise ValueError('no labeled examples present in dataset')
        X_labeled, y_true, _ = data
        y_pred = self.model.predict(X_labeled)
        p, r, ac, g, auc = metrics.precision(y_true, y_pred),metrics.recall(y_true, y_pred),\
                           metrics.accuracy(y_true, y_pred), metrics.g_means(y_true, y_pred),\
                           metrics.auc(y_true, y_pred)
        self.metrics[train_or_valid].append((p, r, ac, g, auc))
    
    def active_simulation_eval(self):
        data = self.dataset.get_unlabeled_data()
        if data == None:
            UserWarning(
                'all examples have been labeled; this eval mode works '
                'if there is unlabeled pool of data in `simulate` mode'
            )
            return
        X_unlabeled, unlabeled_indexes = data
        # get unlabeled examples labels in simulation with `y_ideal`
        y_true = self.dataset.y_ideal[unlabeled_indexes]
        y_pred = self.model.predict(X_unlabeled)
        p, r, ac, g, auc = metrics.precision(y_true, y_pred),metrics.recall(y_true, y_pred),\
                           metrics.accuracy(y_true, y_pred), metrics.g_means(y_true, y_pred),\
                           metrics.auc(y_true, y_pred)
        self.metrics['simulate'].append((p, r, ac, g, auc))
    
    def run(self, quota, **query_kwargs):
        
        for t in range(quota):
            
            if t % 10 == 0 and t != 0:
                print("Query# {}".format(t))
            
            #train model with current labeled train data
            X_train, y_train, _ = self.dataset.get_labeled_data('train')
            self.model.fit(X_train, y_train)
            
            #make active query
            ask_id = self.qs.make_query(self.model, self.dataset, **query_kwargs)
            
            # check current prediction for this id and add user given
            if self.dataset.mode == 'active':
                # prompt user here to label by showing document text
                if not self.docid2path:
                    raise ValueError('document paths are required for interactive learning')
                doctext = utils.read_text_file(self.docid2path[ask_id])
                print("================================================")
                print("                 Document Text                  ")
                print("================================================")
                print(doctext)
                print()
                true_class_ask_id = int(input('DOCUMENT LABEL (Hint: {}): '.format(self.y_ideal[ask_id])))
                _ = os.system('cls')
            else:
                true_class_ask_id = self.dataset.ask_label(ask_id)
            
            # accumulate user labeled example for validation set (if set)
            if self.progressive_validation and t % self.pr_rate == 0 and t > 1:
                self.dataset.update(ask_id, true_class_ask_id, 'valid')
            else:
                # update model and dataset
                self.dataset.update(ask_id, true_class_ask_id, 'query', t=t)
            
            self.relevant_found = sum([1 for props in self.dataset.index2props.values() if props.y_true == 1])
            
            if t > 1 and t % self.eval_at == 0:
                self.supervised_eval('train')
                self.eval_xs['train'].append(t)
                if self.dataset.mode:
                    self.active_simulation_eval()
                    self.eval_xs['simulate'].append(t)
                if self.progressive_validation:
                    # if we have at least 5 positive classes in validation, start evaluation
                    pos_val_count = sum([1 for props in self.dataset.index2props.values() if props.is_valid and props.y_true == 1])
                    if pos_val_count >= 5:
                        self.supervised_eval('valid')
                        self.eval_xs['valid'].append(t)


class ReviewStageSimulation:
    
    def __init__(self, dataset, model, est_prevelance, est_relevant, relevant_found, **kwargs):
        self.dataset = dataset
        self.model = model
        self.est_prevelance = est_prevelance
        self.est_relevant = est_relevant
        self.relevant_found = relevant_found
        
        # sort according to distance from separating hyperplane
        # so e.g. the document predicted as 1  will be on top 
        # and that too in order (the example farthest from plane
        # comes first in review as its safest and most confident
        # guess of document in unlabeled pool to be 1 and so on)
        #
        # *note* : this attempts to approximate the industry
        # standard of having relevance ranked documents
        data = self.dataset.get_unlabeled_data()
        if data == None:
           raise ValueError('there is no unlabeled document for review phase')
        X_unlabeled, unlabeled_indexes = data
        dvalue = self.model.decision_function(X_unlabeled)
        self.sorted_dists = sorted(zip(dvalue, unlabeled_indexes), reverse=True)
        self.n = kwargs.get('num_docs_to_review', None)
    
    def review(self, k=10):
        if not self.n:
            self.n = self.est_relevant - self.relevant_found
            if self.n <= 0:
                UserWarning('estimated number of relevant documents already discovered')
                return
        
        self.r_dists, self.r_inds = [np.array(item) for item in zip(*self.sorted_dists[:self.n])]
        self.nr_dists, self.nr_inds = [np.array(item) for item in zip(*self.sorted_dists[self.n:])]
        
        preds = self.model.classes_[(self.r_dists > 0).astype(np.int)]
        # since this is simulation, we can get true labels, otherwise
        # it will happen interactively with user
        true = self.dataset.y_ideal[self.r_inds]
        
        self.reviewed = 0
        self.rounds = (int(self.n/k) if self.n % k == 0 else int(self.n/k) + 1) - 1
        self.reviewed_batches = list()
        
        for i in range(self.rounds):
            ith_preds = preds[i*k:(i+1)*k]
            ith_true = true[i*k:(i+1)*k]
            ith_inds = self.r_inds[i*k:(i+1)*k]
            self.reviewed_batches.append((ith_preds, ith_true, ith_inds))
            self.reviewed += len(ith_inds)
        
        self.k = k
    
    def drd(self, scale=False):
        # document rank distribution
        r_dists_pos = np.array([i for i in self.r_dists if i >= 0])
        r_dists_neg = np.array([i for i in self.r_dists if i < 0])
        nr_dists_pos = np.array([i for i in self.nr_dists if i >= 0])
        nr_dists_neg = np.array([i for i in self.nr_dists if i < 0])
        
        tmp = np.concatenate([r_dists_pos, r_dists_neg, nr_dists_pos, nr_dists_neg])
        if scale:
            tmp = (tmp - min(tmp)) / (max(tmp) - min(tmp))
            
            i, j = 0, len(r_dists_pos) 
            r_dists_pos = tmp[i:j]
            
            i, j = j, j + len(r_dists_neg)
            r_dists_neg = tmp[i:j]
            
            i, j = j, j + len(nr_dists_pos)
            nr_dists_pos = tmp[i:j]
            
            i, j = j, j + len(nr_dists_neg)
            nr_dists_neg = tmp[i:j]
        
        hist_arr = list()
        for item in [r_dists_pos, r_dists_neg, nr_dists_pos, nr_dists_neg]:
            hist_arr.append(item)
        
        plots.plot_drd(hist_arr)
    
    def prp(self, interval=200):
        # prioritized review progress
        xs = list()
        ys = list()
        docs_reviewed = 0
        for preds, true, _ in self.reviewed_batches:
            rr = sum([1 for i, j in zip(preds, true) if i == 1 and j == 1]) / self.k
            docs_reviewed += len(true)
            if docs_reviewed % interval == 0:
                xs.append(docs_reviewed)
                ys.append(rr)
        
        plots.plot_prp(xs, ys)
    
    def pr_at_k(self):
        # precision-recall analysis
        precision = list()
        recall = list()
        rel_docs_found = self.relevant_found
        
        for preds, true, _ in self.reviewed_batches:
            rel_docs_in_topk = sum([1 for i, j in zip(preds, true) if i == 1 and j == 1])
            rel_docs_found += rel_docs_in_topk
            if rel_docs_found >= self.est_relevant:
                print("estimated number of relevant documents found: {}".format(self.est_relevant))
                break
            precision.append(rel_docs_in_topk / self.k)
            recall.append(rel_docs_found / self.est_relevant)
        
        plots.plot_review_pr_curve(precision, recall, self.k)
        
        return precision, recall
