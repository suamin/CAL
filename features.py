# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models import Word2Vec, Doc2Vec, FastText
from gensim.models import TfidfModel

import time
import numpy as np
from gensim import matutils

from sklearn.exceptions import NotFittedError
import plots


DEFAULT_OPTS_WORD2VEC = dict(sg=1, hs=0, size=300, window=10, alpha=0.025, iter=5,
                             min_count=5, negative=15, workers=4, sample=1e-4,
                             min_alpha=0.0001, batch_words=10000, seed=2018)

DEFAULT_OPTS_FT = dict(sg=1, hs=0, size=100, window=10, alpha=0.025, iter=5,
                       min_count=5, negative=10, workers=4, sample=1e-4,
                       min_alpha=0.0001, batch_words=10000, seed=2018,
                       min_n=5, max_n=10, bucket=5000000)

DEFAULT_OPTS_DOC2VEC = dict(dm=1, vector_size=300, window=10, alpha=0.025,
                            min_alpha=0.0001, seed=2018, min_count=5, sample=1e-4,
                            workers=4, epochs=5, negative=15, dm_mean=1)


def tfidf_model(dictionary, smartirs='ntc'):
    model = TfidfModel(dictionary=dictionary, smartirs=smartirs)
    return model


def w2v_model(corpus, model_type='word2vec', opts=None):
    if model_type == 'fasttext':
        if opts == None: opts = DEFAULT_OPTS_FT
        model = FastText(**opts)
    elif model_type == 'word2vec':
        if opts == None: opts = DEFAULT_OPTS_WORD2VEC
        model = Word2Vec(**opts)
    else:
        raise NotImplementedError('invalid embedding model type')
    model.build_vocab(corpus)
    t = time.time()
    model.train(corpus, total_examples=model.corpus_count,
                epochs=model.iter, start_alpha=model.alpha,
                end_alpha=model.min_alpha)
    t = time.time() - t
    print("training took word embeddings model took: %0.3fs" % t)
    return model


def docvec_from_w2v(w2v_model, document, normalize=True):
    docvec = list()
    for token in document:
        if token in w2v_model:
            wv = w2v_model[token]
        else:
            continue
        docvec.append(wv)
    if docvec:
        docvec = np.array(docvec).mean(axis=0).astype(np.float32)
        if normalize:
            docvec = matutils.unitvec(docvec)
    else:
        docvec = np.zeros(w2v_model.vector_size).astype(np.float32)
    return docvec


def d2v_model(corpus, opts=None):
    if opts == None: opts = DEFAULT_OPTS_DOC2VEC
    model = Doc2Vec(**opts)
    model.build_vocab(corpus)
    t = time.time()
    model.train(corpus, total_examples=model.corpus_count,
                epochs=model.iter, start_alpha=model.alpha,
                end_alpha=model.min_alpha)
    t = time.time() - t
    print("training took doc2vec model took: %0.3fs" % t)
    return model


def clf_topn_coef(clf, feature_names, topn=10, show=False):
    # taken and modified from:
    # https://stackoverflow.com/a/11140887
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names), reverse=True)
    
    top_pos = coefs_with_fns[:topn]
    top_neg = coefs_with_fns[:-(topn + 1):-1]
    
    for label, features in (('+', top_pos), ('-', top_neg)):
        print("Class [{}] top-{} features:".format(label, topn))
        print("\n".join(["%0.4f\t%s" % (fval, fname) for fval, fname in features]))
        print("")
    
    if show:
        plots.plot_clf_top_coefs(top_pos, top_neg, feature_names, topn)
    
    return top_pos, top_neg


def features_predictive_contrib(model, feature_vec, feature_names, show=False):
    
    if not hasattr(model, 'coef_') or model.coef_ is None:
        raise NotFittedError("This %(name)s instance is not fitted "
                             "yet" % {'name': type(model).__name__})
    
    if hasattr(feature_vec, 'toarray'):
        feature_vec = feature_vec.toarray().flatten()
    
    nonzero_feat_inds = np.argwhere(feature_vec != 0).flatten()
    relative_coefs = model.coef_.flatten()[nonzero_feat_inds]
    feat_coef_prod = feature_vec[nonzero_feat_inds] * relative_coefs
    
    if sum(feat_coef_prod) < 0:
        pred_class = 0
    else:
        pred_class = 1
    
    feat_coef_prod = -1. * feat_coef_prod if pred_class == 0 else feat_coef_prod
    feat_coef_prod = sorted(zip(feat_coef_prod, feature_names[nonzero_feat_inds]), reverse=True)
    
    print("Class Prediction: {} | Features Contribution:".format(pred_class))
    for fval, fname in feat_coef_prod:
        print("%0.4f\t%s" % (fval, fname))
    
    if show:
        contribs, names = zip(*feat_coef_prod)
        plots.plot_document_features(contribs, names, pred_class)
    
    return feat_coef_prod, pred_class
