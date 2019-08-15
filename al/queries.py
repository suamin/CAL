"""
This module is a clone of `libact` active learning `query_strategies` sub-module,
only those algorithms are ported that runs on windows machine (unlike HintSVM,
variance_reduction). Multi-label algorithms are also not supported yet, multi-class
is possible but target problems are binary classification.

Major modifications were made to Dataset API and its design and those changes
are reflected in query strategies by making necessary editions.

Checkout libact at:
    https://github.com/ntucllab/libact
    http://libact.readthedocs.io/en/latest
"""

import numpy as np
import copy

import bisect
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


class UncertaintySampling:
    
    """Uncertainty Sampling
    
    This class implements Uncertainty Sampling active learning algorithm [1]_.
    
    Parameters
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` or :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The base model used for training.
    
    method: {'lc', 'sm', 'entropy'}, optional (default='lc')
        least confidence (lc), it queries the instance whose posterior
        probability of being positive is nearest 0.5 (for binary
        classification);
        smallest margin (sm), it queries the instance whose posterior
        probability gap between the most and the second probable labels is
        minimal;
        entropy, requires :py:class:`libact.base.interfaces.ProbabilisticModel`
        to be passed in as model parameter;
    
    
    Attributes
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` or :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The model trained in last query.
    
    
    Examples
    --------
    Here is an example of declaring a UncertaintySampling query_strategy
    object:
    
    .. code-block:: python
       
       from libact.query_strategies import UncertaintySampling
       from libact.models import LogisticRegression
       
       qs = UncertaintySampling(
                dataset, # Dataset object
                model=LogisticRegression(C=0.1)
            )
    
    Note that the model given in the :code:`model` parameter must be a
    :py:class:`ContinuousModel` which supports predict_real method.
    
    
    References
    ----------
    
    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.
    """
    
    def __init__(self):
        pass
    
    def make_query(self, model, dataset, method='sm', incremental=False, use_proba=False, return_score=False):
        
        if method not in ['lc', 'sm', 'entropy']:
            raise TypeError(
                "supported methods are ['lc', 'sm', 'entropy'], the given one "
                "is: " + self.method
            )
        
        if method == 'entropy' and not hasattr(model, 'predict_proba'):
            raise TypeError(
                "`entropy` method only supported for probabilistic models"
            )
        
        # models with support for both such as LogisticRegression
        if hasattr(model, 'decision_function') and hasattr(model, 'predict_proba'):
            if use_proba: # when both available take priority as provided
                score_func = 'p'
            else:
                score_func = 'd'
        elif hasattr(model, 'decision_function'):
            score_func = 'd'
        elif hasattr(model, 'predict_proba'):
            score_func = 'p'
        else:
            raise TypeError(
                "only continuos or probabilistic models with `decision_function` or "
                "`predict_proba` are supported"
            )
        
        X_pool, unlabeled_ids = dataset.get_unlabeled_data()
        if score_func == 'p':
            dvalue = model.predict_proba(X_pool)
        else:
            dvalue = model.decision_function(X_pool)
            # n_classes == 2
            if len(np.shape(dvalue)) == 1:
                dvalue = np.vstack((-dvalue, dvalue)).T
        
        # least confident
        if method == 'lc':
            score = -np.max(dvalue, axis=1)
        # smallest margin
        elif method == 'sm':
            if np.shape(dvalue)[1] > 2:
                # Find 2 largest decision values
                dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])
            score = -np.abs(dvalue[:, 0] - dvalue[:, 1])
        # entropy
        elif method == 'entropy':
            score = np.sum(-dvalue * np.log(dvalue), axis=1)
        
        ask_id = np.argmax(score)
        
        if return_score:
            return unlabeled_ids[ask_id], list(zip(unlabeled_ids, score))
        else:
            return unlabeled_ids[ask_id]


class EER:
    
    """Expected Error Reduction(EER)
    
    This class implements EER active learning algorithm [1]_.
    
    Parameters
    ----------
    model: :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The base model used for training.
    
    loss: {'01', 'log'}, optional (default='log')
        The loss function expected to reduce
    
    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.
    
    Attributes
    ----------
    model: :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The model trained in last query.
    
    Examples
    --------
    Here is an example of declaring a UncertaintySampling query_strategy
    object:
    
    .. code-block:: python
       
       from libact.query_strategies import EER
       from libact.models import LogisticRegression
       
       qs = EER(dataset, model=LogisticRegression(C=0.1))
    
    Note that the model given in the :code:`model` parameter must be a
    :py:class:`ContinuousModel` which supports predict_real method.
    
    
    References
    ----------
    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.
    """
    
    def __init__(self):
        pass
    
    def make_query(self, model, dataset, loss='01'):
        
        if loss not in ['01', 'log']:
            raise TypeError(
                "supported methods are ['01', 'log'], the given one "
                "is: " + loss
            )
        
        if not hasattr(model, 'predict_proba'):
            raise TypeError(
                "only probabilistic models are supported"
            )
        
        X, y, _ = zip(*dataset.get_labeled_data('train'))
        X_pool, unlabeled_ids = zip(*dataset.get_unlabeled_data())
        
        classes = np.unique(y)
        n_classes = len(classes)
        
        proba = model.predict_proba(X_pool)
        
        scores = list()
        for i, x in enumerate(X_pool):
            
            score = list()
            for yi in range(n_classes):
                m = copy.deepcopy(model)
                m.fit(np.vstack((X, [x])), y.tolist() + [yi])
                p = m.predict_proba(X_pool)
                
                # 0/1 loss
                if loss == '01':
                    score.append(proba[i, yi] * np.sum(1-np.max(p, axis=1)))
                # log loss
                elif loss == 'log':
                    score.append(proba[i, yi] * -np.sum(p * np.log(p)))
            
            scores.append(np.sum(score))
        
        choices = np.where(np.array(scores) == np.min(scores))[0]
        ask_idx = np.random.choice(choices)
        
        return unlabeled_ids[ask_idx]


"""Query by committee

This module contains a class that implements Query by committee active learning
algorithm.

**change: __init__() method called teach_students() and requires dataset
so call external and call make_query afterwards
"""


class QueryByCommittee:
    
    r"""Query by committee
    
    Parameters
    ----------
    models : list of :py:mod:`libact.models` instances or str
        This parameter accepts a list of initialized libact Model instances,
        or class names of libact Model classes to determine the models to be
        included in the committee to vote for each unlabeled instance.
    
    disagreement : ['vote', 'kl_divergence'], optional (default='vote')
        Sets the method for measuring disagreement between models.
        'vote' represents vote entropy.
        kl_divergence requires models being ProbabilisticModel
    
    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.
    
    Attributes
    ----------
    students : list, shape = (len(models))
        A list of the model instances used in this algorithm.
    
    random_states\_ : np.random.RandomState instance
        The random number generator using.
    
    Examples
    --------
    Here is an example of declaring a QueryByCommittee query_strategy object:
    
    .. code-block:: python
       
       from libact.query_strategies import QueryByCommittee
       from libact.models import LogisticRegression
       
       qs = QueryByCommittee(
                dataset, # Dataset object
                models=[
                    LogisticRegression(C=1.0),
                    LogisticRegression(C=0.1),
                ],
            )
    
    
    References
    ----------
    .. [1] Seung, H. Sebastian, Manfred Opper, and Haim Sompolinsky. "Query by
           committee." Proceedings of the fifth annual workshop on
           Computational learning theory. ACM, 1992.
    """
    
    def __init__(self, **kwargs):
        
        self.disagreement = kwargs.pop('disagreement', 'vote')
        
        models = kwargs.pop('models', None)
        if models is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
            )
        elif not models:
            raise ValueError("models list is empty")
        
        if self.disagreement == 'kl_divergence':
            for model in models:
                if not hasattr(model, 'predict_proba'):
                    raise TypeError(
                        "Given disagreement set as 'kl_divergence', all models"
                        "should be probabilistic."
                    )
        
        random_state = kwargs.pop('random_state', 2018)
        self.random_state_ = np.random.RandomState(random_state)
        
        self.students = list()
        for model in models:
            self.students.append(model)
        self.n_students = len(self.students)
    
    def _vote_disagreement(self, votes):
        """
        Return the disagreement measurement of the given number of votes.
        It uses the vote vote to measure the disagreement.
        
        Parameters
        ----------
        votes : list of int, shape==(n_samples, n_students)
            The predictions that each student gives to each sample.
        
        Returns
        -------
        disagreement : list of float, shape=(n_samples)
            The vote entropy of the given votes.
        """
        ret = list()
        for candidate in votes:
            ret.append(0.0)
            lab_count = {}
            for lab in candidate:
                lab_count[lab] = lab_count.setdefault(lab, 0) + 1
            
            # Using vote entropy to measure disagreement
            for lab in lab_count.keys():
                ret[-1] -= lab_count[lab] / self.n_students * math.log(float(lab_count[lab]) / self.n_students)
        
        return ret
    
    def _kl_divergence_disagreement(self, proba):
        """
        Calculate the Kullback-Leibler (KL) divergence disaagreement measure.
        
        Parameters
        ----------
        proba : array-like, shape=(n_samples, n_students, n_class)
        
        Returns
        -------
        disagreement : list of float, shape=(n_samples)
            The kl_divergence of the given probability.
        """
        n_students = np.shape(proba)[1]
        consensus = np.mean(proba, axis=1) # shape=(n_samples, n_class)
        # average probability of each class across all students
        consensus = np.tile(consensus, (n_students, 1, 1)).transpose(1, 0, 2)
        kl = np.sum(proba * np.log(proba / consensus), axis=2)
        return np.mean(kl, axis=1)
    
    def _labeled_uniform_sample(self, dataset, sample_size):
        """sample labeled entries uniformly"""
        X, y, labeled_entries = dataset.get_labeled_data('train')
        samples = [labeled_entries[self.random_state_.randint(0, len(labeled_entries))] for _ in range(sample_size)]
        return X[samples, :], y[samples]
    
    def teach_students(self, dataset):
        """
        Train each model (student) with the labeled data using bootstrap
        aggregating (bagging).
        """
        for student in self.students:
            bag = self._labeled_uniform_sample(dataset, int(dataset.len_labeled()))
            while len(set(bag[1])) != dataset.get_num_of_labels():
                bag = self._labeled_uniform_sample(dataset, int(dataset.len_labeled()))
                UserWarning('There is student receiving only one label,'
                            're-sample the bag.')
            student.fit(bag[0], bag[1])
    
    def update(self, dataset):
        # Train each model with newly updated label.
        self.teach_students(dataset)
    
    def make_query(self, dataset):
        X_pool, unlabeled_ids = zip(*dataset.get_unlabeled_data())
        
        if self.disagreement == 'vote':
            # Let the trained students vote for unlabeled data
            votes = np.zeros((X_pool.shape[0], len(self.students)))
            for i, student in enumerate(self.students):
                votes[:, i] = student.predict(X_pool)
            
            vote_entropy = self._vote_disagreement(votes)
            ask_idx = self.random_state_.choice(np.where(np.isclose(vote_entropy, np.max(vote_entropy)))[0])
        
        elif self.disagreement == 'kl_divergence':
            proba = list()
            for student in self.students:
                proba.append(student.predict_proba(X_pool))
            proba = np.array(proba).transpose(1, 0, 2).astype(float)
            
            avg_kl = self._kl_divergence_disagreement(proba)
            ask_idx = self.random_state_.choice(np.where(np.isclose(avg_kl, np.max(avg_kl)))[0])
        
        return unlabeled_ids[ask_idx]


""" Active Learning by QUerying Informative and Representative Examples (QUIRE)

This module contains a class that implements an active learning algorithm
(query strategy): QUIRE

"""

class QUIRE:
    
    """Querying Informative and Representative Examples (QUIRE)
    
    Query the most informative and representative examples where the metrics
    measuring and combining are done using min-max approach.
    
    Parameters
    ----------
    lambda: float, optional (default=1.0)
        A regularization parameter used in the regularization learning
        framework.
    
    kernel : {'linear', 'poly', 'rbf', callable}, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', or a callable.
        If a callable is given it is used to pre-compute the kernel matrix
        from data matrices; that matrix should be an array of shape
        ``(n_samples, n_samples)``.
    
    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    
    gamma : float, optional (default=1.)
        Kernel coefficient for 'rbf', 'poly'.
    
    coef0 : float, optional (default=1.)
        Independent term in kernel function.
        It is only significant in 'poly'.
    
    
    Attributes
    ----------
    
    Examples
    --------
    Here is an example of declaring a QUIRE query_strategy object:
    
    .. code-block:: python
       
       from libact.query_strategies import QUIRE
       
       qs = QUIRE(
                dataset, # Dataset object
            )
    
    References
    ----------
    .. [1] S.-J. Huang, R. Jin, and Z.-H. Zhou. Active learning by querying
           informative and representative examples.
    """
    
    def __init__(self, dataset, **kwargs):
        self.Uindex = [
            idx for _, idx in dataset.get_unlabeled_data()
        ]
        self.Lindex = [
            idx for idx in range(dataset.X.shape[0]) if idx not in self.Uindex
        ]
        self.lmbda = kwargs.pop('lambda', 1.)
        
        X, self.y = dataset.get_entries()
        self.y = list(self.y)
        
        self.kernel = kwargs.pop('kernel', 'rbf')
        
        if self.kernel == 'rbf':
            self.K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma', 1.))
        
        elif self.kernel == 'poly':
            self.K = polynomial_kernel(X=X,
                                       Y=X,
                                       coef0=kwargs.pop('coef0', 1),
                                       degree=kwargs.pop('degree', 3),
                                       gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'linear':
            self.K = linear_kernel(X=X, Y=X)
        elif hasattr(self.kernel, '__call__'):
            self.K = self.kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError
        
        if not isinstance(self.K, np.ndarray):
            raise TypeError('K should be an ndarray')
        
        if self.K.shape != (len(X), len(X)):
            raise ValueError('kernel should have size (%d, %d)' % (len(X), len(X)))
        
        self.L = np.linalg.inv(self.K + self.lmbda * np.eye(len(X)))
    
    def update(self, entry_id, label):
        bisect.insort(a=self.Lindex, x=entry_id)
        self.Uindex.remove(entry_id)
        self.y[entry_id] = label
    
    def make_query(self):
        L = self.L
        Lindex = self.Lindex
        Uindex = self.Uindex
        query_index = -1
        min_eva = np.inf
        y_labeled = np.array([label for label in self.y if label is not None])
        det_Laa = np.linalg.det(L[np.ix_(Uindex, Uindex)])
        # efficient computation of inv(Laa)
        M3 = np.dot(self.K[np.ix_(Uindex, Lindex)], np.linalg.inv(self.lmbda * np.eye(len(Lindex))))
        M2 = np.dot(M3, self.K[np.ix_(Lindex, Uindex)])
        M1 = self.lmbda * np.eye(len(Uindex)) + self.K[np.ix_(Uindex, Uindex)]
        inv_Laa = M1 - M2
        iList = list(range(len(Uindex)))
        if len(iList) == 1:
            return Uindex[0]
        for i, each_index in enumerate(Uindex):
            # go through all unlabeled instances and compute their evaluation
            # values one by one
            Uindex_r = Uindex[:]
            Uindex_r.remove(each_index)
            iList_r = iList[:]
            iList_r.remove(i)
            inv_Luu = inv_Laa[np.ix_(iList_r, iList_r)] - 1 / inv_Laa[i, i] * np.dot(inv_Laa[iList_r, i], inv_Laa[iList_r, i].T)
            tmp = np.dot(L[each_index][Lindex] - np.dot(np.dot(L[each_index][Uindex_r], inv_Luu), L[np.ix_(Uindex_r, Lindex)]), y_labeled,)
            eva = L[each_index][each_index] - det_Laa / L[each_index][each_index] + 2 * np.abs(tmp)
            
            if eva < min_eva:
                query_index = each_index
                min_eva = eva
        return query_index


"""Density Weighted Uncertainty Sampling (DWUS)
"""


class DWUS:
    """Density Weighted Uncertainty Sampling (DWUS)
    
    We use the KMeans algorithm for clustering instead of the Kmediod for now.
    
    Support binary case and LogisticRegression only.
    
    Parameters
    ----------
    n_clusters : int, optional, default: 5
        Number of clusters for kmeans to cluster.
    
    sigma : float, optional, default: .1
        The variance of the multivariate gaussian used to model density.
    
    max_iter : int, optional, default: 100
        The maximum number of iteration used in estimating density through EM
        algorithm.
    
    tol : float, default: 1e-4
        Tolerance with regards to inertia to declare convergence.
    
    C : float, default: 1.
        Regularization term for logistic regression.
    
    kmeans_param : dict, default: {}
        Parameter for sklearn.cluster.KMeans.
        see, http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    
    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.
    
    Attributes
    ----------
    kmeans_ : sklearn.cluster.KMeans object
        The clustering algorithm instance.
    
    p_x : ndarray, shape=(n_labeled + n_unlabeled, )
        The density estimate for each x. Its order is the same as dataset.data.
    
    Examples
    --------
    Here is an example of how to declare a DWUS query_strategy object:
    
    .. code-block:: python
       
       from libact.query_strategies import DWUS
       from libact.models import LogisticRegression
       
       qs = DWUS(dataset)
    
    References
    ----------
    .. [1] Donmez, Pinar, Jaime G. Carbonell, and Paul N. Bennett. "Dual
           strategy active learning." Machine Learning: ECML 2007. Springer
           Berlin Heidelberg, 2007. 116-127.
    .. [2] Nguyen, Hieu T., and Arnold Smeulders. "Active learning using
           pre-clustering." Proceedings of the twenty-first international
           conference on Machine learning. ACM, 2004.
    """
    
    def __init__(self, dataset, **kwargs):
        
        self.n_clusts = kwargs.pop('n_clusters', 5)
        self.sigma = kwargs.pop('sigma', 0.1)
        self.max_iter = kwargs.pop('max_iter', 100)
        self.tol = kwargs.pop('tol', 1e-4)
        self.C = kwargs.pop('C', 1.)
        
        random_state = kwargs.pop('random_state', 2018)
        self.random_state_ = np.random.RandomState(random_state)
        
        kmeans_param = kwargs.pop('kmeans_param', {})
        
        if 'random_state' not in kmeans_param:
            kmeans_param['random_state'] = self.random_state_
        
        self.kmeans_ = KMeans(n_clusters=self.n_clusts, **kmeans_param)
        all_x = dataset.X
        
        # Cluster the data.
        self.kmeans_.fit(all_x)
        d = len(all_x[0])
        
        centers = self.kmeans_.cluster_centers_
        P_k = np.ones(self.n_clusts) / float(self.n_clusts)
        
        dis = np.zeros((all_x.shape[0], self.n_clusts))
        for i in range(self.n_clusts):
            dis[:, i] = np.exp(-np.einsum('ij,ji->i', (all_x - centers[i]), (all_x - centers[i]).T) / 2 / self.sigma)
        
        # EM percedure to estimate the prior
        for _ in range(self.max_iter):
            # E-step P(k|x)
            temp = dis * np.tile(P_k, (len(all_x), 1))
            # P_k_x, shape = (len(all_x), n_clusts)
            P_k_x = temp / np.tile(np.sum(temp, axis=1), (self.n_clusts, 1)).T
            
            # M-step
            P_k = 1./len(all_x) * np.sum(P_k_x, axis=0)
        
        self.P_k_x = P_k_x
        
        p_x_k = np.zeros((len(all_x), self.n_clusts))
        for i in range(self.n_clusts):
            p_x_k[:, i] = multivariate_normal.pdf(all_x, mean=centers[i], cov=np.ones(d)*np.sqrt(self.sigma))
        
        self.p_x = np.dot(p_x_k, P_k).reshape(-1)
    
    def make_query(self, dataset):
        _, unlabeled_ids = zip(*dataset.get_unlabeled_data())
        X, y, labeled_entry_ids = zip(*dataset.get_labeled_data('train'))
        labels = y.reshape(-1, 1)
        centers = self.kmeans_.cluster_centers_
        P_k_x = self.P_k_x
        p_x = self.p_x[unlabeled_ids]
        
        clf = DensityWeightedLogisticRegression(P_k_x[labeled_entry_ids, :], centers, self.C)
        clf.train(labeled_entry_ids, labels)
        P_y_k = clf.predict()
        
        P_y_x = np.zeros(len(unlabeled_ids))
        for k, center in enumerate(centers):
            P_y_x += P_y_k[k] * P_k_x[unlabeled_ids, k]
        
        # binary case
        expected_error = P_y_x
        expected_error[P_y_x >= 0.5] = 1. - P_y_x[P_y_x >= 0.5]
        
        ask_id = np.argmax(expected_error * p_x)
        
        return unlabeled_ids[ask_id]


class DensityWeightedLogisticRegression:
    r"""Density Weighted Logistic Regression
    
    Density Weighted Logistice Regression is used in DWUS to estimate the
    probability of representing which label for each cluster.
    Density Weighted Logistic Regression optimizes the following likelihood
    function.
    
    .. math::
        
        \sum_{i\in I_l} \ln P(y_i|\mathbf{x}_i; w)
    
    Including the regularization term and
    :math:`P(y,k|x) = \sum^K_{k=1}P(y|k)P(k|x)`, it becomes the following
    function:
    
    .. math::
        
        \frac{C}{2} \|w\|2 - \sum_{i\in I_l} \ln \{\sum^K_{k=1} P(k|\mathbf{x}_i) P(y_i|k; w)\}
    
    Where :math:`K` is the number of clusters, :math:`I_l` is the indices for
    labled data, :math:`w` is the logistice regression parameter,
    :math:`\mathbf{x}_i` and :math`y_i` are the feature vector and label for
    indice :math:`i`.
    
    Parameters
    ----------
    density_estimate: array-like, shape=(n_samples, n_clusters)
        The probability of each sample to each cluster.
    
    centers : array-like, shape=(n_clusters, n_features)
        The point of each cluster center.
    
    C : float
        Regularization term for logistic regression.
    
    Attributes
    ----------
    self.w_ : ndarray, shape=(n_features + 1, )
        Logistic regression parameter, the last element is the bias term.
    """
    
    def __init__(self, density_estimate, centers, C):
        self.density = np.asarray(density_estimate)
        self.centers = np.asarray(centers)
        self.C = C
        self.w_ = None
    
    def _likelihood(self, w, X, y):
        w = w.reshape(-1, 1)
        sigmoid = lambda t: 1. / (1. + np.exp(-t))
        # w --> shape = (d+1, 1)
        L = lambda w: (self.C/2. * np.dot(w[:-1].T, w[:-1]) - \
                np.sum(np.log(np.sum(self.density * sigmoid(np.dot(y, (np.dot(self.centers, w[:-1]) + w[-1]).T)), axis=1)), axis=0))[0][0]
        
        return L(w)
    
    def train(self, X, y):
        d = np.shape(self.centers)[1]
        w = np.zeros((d+1, 1))
        # TODO Use more sophistic optimization methods
        result = minimize(lambda _w: self._likelihood(_w, X, y), w.reshape(-1), method='CG')
        w = result.x.reshape(-1, 1)
        
        self.w_ = w
    
    def predict(self):
        """
        Returns
        -------
        proba : ndarray, shape=(n_clusters, )
            The probability of given cluster being label 1.
        
        """
        if self.w_ is not None:
            sigmoid = lambda t: 1. / (1. + np.exp(-t))
            return sigmoid(np.dot(self.centers, self.w_[:-1]) + self.w_[-1])
        else:
            # TODO the model is not trained
            pass


"""Active learning by learning (ALBL)

This module includes two classes. ActiveLearningByLearning is the main
algorithm for ALBL and Exp4P is the multi-armed bandit algorithm which will be
used in ALBL.
"""


class ActiveLearningByLearning:
    
    r"""Active Learning By Learning (ALBL) query strategy.
    
    ALBL is an active learning algorithm that adaptively choose among existing
    query strategies to decide which data to make query. It utilizes Exp4.P, a
    multi-armed bandit algorithm to adaptively make such decision. More details
    of ALBL can refer to the work listed in the reference section.
    
    Parameters
    ----------
    T : integer
        Query budget, the maximal number of queries to be made.
    
    query_strategies : list of :py:mod:`libact.query_strategies`\
    object instance
        The active learning algorithms used in ALBL, which will be both the
        the arms in the multi-armed bandit algorithm Exp4.P.
        Note that these query_strategies should share the same dataset
        instance with ActiveLearningByLearning instance.
    
    delta : float, optional (default=0.1)
        Parameter for Exp4.P.
    
    uniform_sampler : {True, False}, optional (default=True)
        Determining whether to include uniform random sample as one of arms.
    
    pmin : float, 0<pmin< :math:`\frac{1}{len(query\_strategies)}`,\
                  optional (default= :math:`\frac{\sqrt{\log{N}}}{KT}`)
        Parameter for Exp4.P. The minimal probability for random selection of
        the arms (aka the underlying active learning algorithms). N = K =
        number of query_strategies, T is the number of query budgets.
    
    model : :py:mod:`libact.models` object instance
        The learning model used for the task.
    
    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.
    
    Attributes
    ----------
    query_strategies\_ : list of :py:mod:`libact.query_strategies` object instance
        The active learning algorithm instances.
    
    exp4p\_ : instance of Exp4P object
        The multi-armed bandit instance.
    
    queried_hist\_ : list of integer
        A list of entry_id of the dataset which is queried in the past.
    
    random_states\_ : np.random.RandomState instance
        The random number generator using.
    
    Examples
    --------
    Here is an example of how to declare a ActiveLearningByLearning
    query_strategy object:
    
    .. code-block:: python
       
       from libact.query_strategies import ActiveLearningByLearning
       from libact.query_strategies import HintSVM
       from libact.query_strategies import UncertaintySampling
       from libact.models import LogisticRegression
       
       qs = ActiveLearningByLearning(
            dataset, # Dataset object
            T=100, # qs.make_query can be called for at most 100 times
            query_strategies=[
                UncertaintySampling(dataset, model=LogisticRegression(C=1.)),
                UncertaintySampling(dataset, model=LogisticRegression(C=.01)),
                HintSVM(dataset)
                ],
            model=LogisticRegression()
        )
    
    The :code:`query_strategies` parameter is a list of
    :code:`libact.query_strategies` object instances where each of their
    associated dataset must be the same :code:`Dataset` instance. ALBL combines
    the result of these query strategies and generate its own suggestion of
    which sample to query.  ALBL will adaptively *learn* from each of the
    decision it made, using the given supervised learning model in
    :code:`model` parameter to evaluate its IW-ACC.
    
    References
    ----------
    .. [1] Wei-Ning Hsu, and Hsuan-Tien Lin. "Active Learning by Learning."
           Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
    
    """
    
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.query_strategies_ = kwargs.pop('query_strategies', None)
        if self.query_strategies_ is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: "
                "'query_strategies'"
            )
        elif not self.query_strategies_:
            raise ValueError("query_strategies list is empty")
        
        # parameters for Exp4.p
        self.delta = kwargs.pop('delta', 0.1)
        
        # query budget
        self.T = kwargs.pop('T', None)
        if self.T is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'T'"
            )
        
        _, self.unlabeled_entry_ids = zip(*self.dataset.get_unlabeled_data())
        self.unlabeled_invert_id_idx = {}
        for i, entry in enumerate(self.unlabeled_entry_ids):
            self.unlabeled_invert_id_idx[entry] = i
        
        self.uniform_sampler = kwargs.pop('uniform_sampler', True)
        if not isinstance(self.uniform_sampler, bool):
            raise ValueError("'uniform_sampler' should be {True, False}")
        
        self.pmin = kwargs.pop('pmin', None)
        n_algorithms = (len(self.query_strategies_) + self.uniform_sampler)
        
        if self.pmin and (self.pmin > (1. / n_algorithms) or self.pmin < 0):
            raise ValueError("'pmin' should be 0 < pmin < 1/len(n_active_algorithm)")
        
        self.exp4p_ = Exp4P(
            query_strategies=self.query_strategies_,
            T=self.T,
            delta=self.delta,
            pmin=self.pmin,
            unlabeled_invert_id_idx=self.unlabeled_invert_id_idx,
            uniform_sampler=self.uniform_sampler
        )
        self.budget_used = 0
        
        # classifier instance
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        
        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)
        
        self.query_dist = None
        
        self.W = []
        self.queried_hist_ = []
    
    def calc_reward_fn(self):
        """Calculate the reward value"""
        model = copy.copy(self.model)
        X, y, _ = zip(*dataset.get_labeled_data('train'))
        model.fit(X, y)
        
        # reward function: Importance-Weighted-Accuracy (IW-ACC) (tau, f)
        reward = 0.
        for i in range(len(self.queried_hist_)):
            reward += self.W[i] * (
                model.predict(self.dataset.X[self.queried_hist_[i], :].reshape(1, -1))[0] == self.dataset.index2props[self.queried_hist_[i]].y_true
            )
        reward /= (self.dataset.len_labeled() + self.dataset.len_unlabeled())
        reward /= self.T
        return reward
    
    def calc_query(self):
        """Calculate the sampling query distribution"""
        # initial query
        if self.query_dist is None:
            self.query_dist = self.exp4p_.next(-1, None, None)
        else:
            self.query_dist = self.exp4p_.next(
                self.calc_reward_fn(),
                self.queried_hist_[-1],
                self.dataset.index2props[self.queried_hist_[-1]].y_true
            )
        return
    
    def update(self, entry_id, label):
        # Calculate the next query after updating the question asked with an
        # answer.
        ask_idx = self.unlabeled_invert_id_idx[entry_id]
        self.W.append(1. / self.query_dist[ask_idx])
        self.queried_hist_.append(entry_id)
    
    def make_query(self):
        dataset = self.dataset
        unlabeled_data = dataset.get_unlabeled_data()
        if unlabeled_data == None:
            # might be no more unlabeled data left
            return
        _, unlabeled_entry_ids = unlabeled_data
        
        while self.budget_used < self.T:
            self.calc_query()
            ask_idx = self.random_state_.choice(
                np.arange(len(self.unlabeled_invert_id_idx)),
                size=1,
                p=self.query_dist
            )[0]
            ask_id = self.unlabeled_entry_ids[ask_idx]
            
            if ask_id in unlabeled_entry_ids:
                self.budget_used += 1
                return ask_id
            else:
                self.update(ask_id, dataset.index2props[ask_id].y_true)
        
        raise ValueError("Out of query budget")


class Exp4P(object):
    
    r"""A multi-armed bandit algorithm Exp4.P.
    
    For the Exp4.P used in ALBL, the number of arms (actions) and number of
    experts are equal to the number of active learning algorithms wanted to
    use. The arms (actions) are the active learning algorithms, where is
    inputed from parameter 'query_strategies'. There is no need for the input
    of experts, the advice of the kth expert are always equal e_k, where e_k is
    the kth column of the identity matrix.
    
    Parameters
    ----------
    query_strategies : QueryStrategy instances
        The active learning algorithms wanted to use, it is equivalent to
        actions or arms in original Exp4.P.
    
    unlabeled_invert_id_idx : dict
        A look up table for the correspondance of entry_id to the index of the
        unlabeled data.
    
    delta : float, >0, optional (default=0.1)
        A parameter.
    
    pmin : float, 0<pmin<1/len(query_strategies), optional (default= :math:`\frac{\sqrt{log(N)}}{KT}`)
        The minimal probability for random selection of the arms (aka the
        unlabeled data), N = K = number of query_strategies, T is the maximum
        number of rounds.
    
    T : int, optional (default=100)
        The maximum number of rounds.
    
    uniform_sampler : {True, False}, optional (default=Truee)
        Determining whether to include uniform random sampler as one of the
        underlying active learning algorithms.
    
    Attributes
    ----------
    t : int
        The current round this instance is at.
    
    N : int
        The number of arms (actions) in this exp4.p instance.
    
    query_models\_ : list of :py:mod:`libact.query_strategies` object instance
        The underlying active learning algorithm instances.
    
    References
    ----------
    .. [1] Beygelzimer, Alina, et al. "Contextual bandit algorithms with
           supervised learning guarantees." In Proceedings on the International
           Conference on Artificial Intelligence and Statistics (AISTATS),
           2011u.
    
    """
    
    def __init__(self, *args, **kwargs):
        """ """
        # QueryStrategy class object instances
        self.query_strategies_ = kwargs.pop('query_strategies', None)
        if self.query_strategies_ is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: "
                "'query_strategies'"
            )
        elif not self.query_strategies_:
            raise ValueError("query_strategies list is empty")
        
        # whether to include uniform random sampler as one of underlying active
        # learning algorithms
        self.uniform_sampler = kwargs.pop('uniform_sampler', True)
        
        # n_armss
        if self.uniform_sampler:
            self.N = len(self.query_strategies_) + 1
        else:
            self.N = len(self.query_strategies_)
        
        # weight vector to each query_strategies, shape = (N, )
        self.w = np.array([1. for _ in range(self.N)])
        
        # max iters
        self.T = kwargs.pop('T', 100)
        
        # delta > 0
        self.delta = kwargs.pop('delta', 0.1)
        
        # n_arms = n_models (n_query_algorithms) in ALBL
        self.K = self.N
        
        # p_min in [0, 1/n_arms]
        self.pmin = kwargs.pop('pmin', None)
        if self.pmin is None:
            self.pmin = np.sqrt(np.log(self.N) / self.K / self.T)
        
        self.exp4p_gen = self.exp4p()
        
        self.unlabeled_invert_id_idx = kwargs.pop('unlabeled_invert_id_idx')
        if not self.unlabeled_invert_id_idx:
            raise TypeError(
                "__init__() missing required keyword-only argument:"
                "'unlabeled_invert_id_idx'"
            )
    
    def __next__(self, reward, ask_id, lbl):
        """For Python3 compatibility of generator."""
        return self.next(reward, ask_id, lbl)
    
    def next(self, reward, ask_id, lbl):
        """Taking the label and the reward value of last question and returns
        the next question to ask."""
        # first run don't have reward, TODO exception on reward == -1 only once
        if reward == -1:
            return next(self.exp4p_gen)
        else:
            # TODO exception on reward in [0, 1]
            return self.exp4p_gen.send((reward, ask_id, lbl))
    
    def exp4p(self):
        """The generator which implements the main part of Exp4.P.
        
        Parameters
        ----------
        reward: float
            The reward value calculated from ALBL.
        
        ask_id: integer
            The entry_id of the sample point ALBL asked.
        
        lbl: integer
            The answer received from asking the entry_id ask_id.
        
        Yields
        ------
        q: array-like, shape = [K]
            The query vector which tells ALBL what kind of distribution if
            should sample from the unlabeled pool.
        
        """
        while True:
            # TODO probabilistic active learning algorithm
            # len(self.unlabeled_invert_id_idx) is the number of unlabeled data
            query = np.zeros((self.N, len(self.unlabeled_invert_id_idx)))
            if self.uniform_sampler:
                query[-1, :] = 1. / len(self.unlabeled_invert_id_idx)
            for i, model in enumerate(self.query_strategies_):
                query[i][self.unlabeled_invert_id_idx[model.make_query()]] = 1
            
            # choice vector, shape = (self.K, )
            W = np.sum(self.w)
            p = (1 - self.K * self.pmin) * self.w / W + self.pmin
            
            # query vector, shape= = (self.n_unlabeled, )
            query_vector = np.dot(p, query)
            
            reward, ask_id, _ = yield query_vector
            ask_idx = self.unlabeled_invert_id_idx[ask_id]
            
            rhat = reward * query[:, ask_idx] / query_vector[ask_idx]
            
            # The original advice vector in Exp4.P in ALBL is a identity matrix
            yhat = rhat
            vhat = 1 / p
            self.w = self.w * np.exp(self.pmin / 2 * (yhat + vhat * np.sqrt(np.log(self.N / self.delta) / self.K / self.T)))
        
        raise StopIteration
