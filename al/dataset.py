# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse


class Index:
    
    def __init__(self, i, **kwargs):
        """
        states:
            0:  'unlabeled'
            1:  'labeled'
        """
        self.index = i
        self.y_true = kwargs.get('y_true', None)
        self.y_pred = None
        self.state = 1 if self.y_true != None else 0
        self.is_seed =  kwargs.get('is_seed', False)
        self.is_query = kwargs.get('is_query', False)
        
        is_train = kwargs.get('is_train')
        is_valid = kwargs.get('is_valid')
        
        if is_train and is_valid:
            raise ValueError('an example cant be both in validation and training set')
        if self.y_true != None and is_train:
            self.is_train = True
        elif self.y_true != None and is_valid:
            self.is_valid = True
        else:
            self.is_train = False
            self.is_valid = False
        
        self.history = list()
        self.t = None
        self.is_support = None 
        self.weight = 1.0
        self.tags = kwargs.get('labels', list())
    
    def update(self, y, state, **kwargs):
        
        if self.is_seed:
            raise ValueError('cant update seeded examples')
        
        if state != 'pred':
            if state == 'query' or state == 'train':
                if state == 'query' and 't' not in kwargs:
                    raise ValueError('with query label update, the query no. `t` should be passed')
                else:
                    self.is_query = True
                    self.t = kwargs['t']
                if 'is_support' in kwargs:
                    self.is_support = kwargs['is_support']
                self.is_train = True
            elif state == 'valid':
                self.is_valid = True
            else:
                raise ValueError('invalid state')
            self.y_true = y
            self.state = 1
            if 'weight' in kwargs:
                self.weight = kwargs['weight']

        else:
            if self.y_pred:
                self.history.append(self.y_pred)
            self.y_pred = y


class Dataset:
    
    def __init__(self, X, y=None, mode=None, y_ideal=None):
        
        if isinstance(X, np.ndarray):
            self.X = X
            self.is_sparse = False
        elif sparse.isspmatrix_csr(X):
            self.X = X
            self.is_sparse = True
        elif hasattr(X, '__getitem__') or hasattr(X, '__iter__'):
            self.X = np.array([np.array(x) for x in X])
            self.is_sparse = False
        else:
            raise TypeError('invalid data type for feature matrix `X`: %s' % type(X).__name__)
        
        if mode == 'simulate':
            if isinstance(y_ideal, np.ndarray) or isinstance(y_ideal, list):
                assert len(y_ideal) == X.shape[0], '`y_ideal` must have same no. of examples as `X`'
                self.y_ideal = np.array(y_ideal)
                self.mode = 'simulate'
            else:
                raise ValueError('in `simulate` mode, a fully labeled `y_ideal` is required')
        else:
            self.y_ideal = np.array([None] * self.X.shape[0])
            self.mode = 'active'
        
        if isinstance(y, np.ndarray) or isinstance(y, list):
            pass
        elif y == None:
            y = [None] * self.X.shape[0]
        else:
            raise TypeError('invalid data for class labels vector `y`: %s' % type(y).__name__)
        
        self.index2props = dict()
        
        for index, value in enumerate(y):
            if value != None:
                self.index2props[index] = Index(index, y_true=value, is_seed=True, is_train=True)
            else:
                self.index2props[index] = Index(index)
        
        self.modified = False
    
    def get_labeled_data(self, which='train', sample_weights=False):
        
        indexes = list()
        
        for index, props in self.index2props.items():
            if props.state == 1:
                if which == 'valid' and props.is_valid:
                    indexes.append(index)
                elif which == 'query' and props.is_query:
                    indexes.append(index)
                elif which == 'seed' and props.is_seed:
                    indexes.append(index)
                elif which == 'train' and props.is_train:
                    indexes.append(index)
        
        if not indexes:
            return
        else:
            labels = np.array([self.index2props[index].y_true for index in indexes])
            if sample_weights:
                sample_weights = np.array([self.index2props[i].weight for i in indexes])
                return self.X[indexes, :], labels, indexes, sample_weights
            else:
                return self.X[indexes, :], labels, indexes
    
    def get_unlabeled_data(self, sample_weights=False):
        
        indexes = list()
        
        for index, props in self.index2props.items():
            if props.state == 0:
                indexes.append(index)
        
        if not indexes:
            return
        else:
            if sample_weights:
                sample_weights = np.array([self.index2props[i].weight for i in indexes])
                return self.X[indexes, :], indexes, sample_weights
            else:
                return self.X[indexes, :], indexes
    
    def get_entries(self):
        y = np.array([props.y_true for props in self.index2props.values()])
        return self.X, y
    
    def get_entry(self, index, props=False):
        if props:
            return self.X[index, :], self.index2props[index].props
        else:
            return self.X[index, :]
    
    def ask_label(self, index):
        if not self.mode == 'simulate':
            raise ValueError('label request only support in simulation mode')
        else:
            return self.y_ideal[index]
    
    def append(self, x, y=None, to='train'):
        index = self.X.shape[0]
        if self.is_sparse:
            assert x[-1][0] < self.X.shape[1], 'feature vector length mismatch'
            col, data = zip(*x)
            data, col = np.array(data), np.array(col)
            row = np.array([0] * len(col))
            x_sparse_mat = sparse.coo_matrix((data, (row, col)), shape=(1, self.X.shape[1])).asformat('csr')
            # possibly not the most efficient way to add row to sparse matrix as it doesnt work inplace:
            # https://github.com/scipy/scipy/blob/v0.19.0/scipy/sparse/construct.py#L461-L492
            # best is to NOT update sparse matrices once they have been built
            self.X = sparse.vstack([self.X, x_sparse_mat])
        else:
            assert len(x) == self.X.shape[1], 'feature vector length mismatch'
            self.X = np.append(X, np.array(x).reshape(1, -1), axis=0)
        if y:
            if to == 'valid':
                self.index2props[index] = Index(index, y_true=y, is_valid=True)
            else:
                self.index2props[index] = Index(index, y_true=y, is_train=True)
        else:
            self.index2props[index] = Index(index)
        return index
    
    def update(self, entry_id, label, state, **kwargs):
        self.index2props[entry_id].update(label, state, **kwargs)
    
    def len_labeled(self, exclude_val=True):
        count = 0
        for index, props in self.index2props.items():
            if props.state == 1:
                if exclude_val and not props.is_valid:
                    count += 1
                else:
                    count += 1
        return count
    
    def len_unlabeled(self):
        return sum([not props.state for props in self.index2props.values()])
    
    def get_classes(self):
        return set([props.y_true for props in self.index2props.values() if props.y_true != None])
    
    def get_num_of_labels(self):
        return len(self.get_classes())
