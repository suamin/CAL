# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import sys
import os
import codecs
from vocab import Vocab

from gensim import matutils
from gensim.models.doc2vec import TaggedDocument
from scipy import sparse

import numpy as np
import features


class CorpusIterable:
    
    def __init__(self, fname, is_d2v=None):
        self.fname = fname # preprocessed corpus file
        self.is_d2v = is_d2v
    
    def __iter__(self):
        with codecs.open(self.fname, 'r', 'utf-8', 'ignore') as rf:
            if self.is_d2v:
                for doctag, doctext in enumerate(rf):
                    yield TaggedDocument(doctext.split(), [doctag])
            else:
                for doctext in rf:
                    yield doctext.split()


def w2v_features(model, docs, normalize=True):
    num_docs = sum([1 for _ in docs])
    docvecs = np.empty((num_docs, model.vector_size), dtype=np.float32)
    for idx, doctokens in enumerate(docs):
        docvec = features.docvec_from_w2v(model, doctokens, normalize)
        docvecs[idx, :] = docvec
    return docvecs


def d2v_features(model, normalize=True):
    num_docs, vector_size = model.docvecs.doctag_syn0.shape
    if not normalize:
        return model.docvecs.doctag_syn0
    else:
        docvecs = np.empty((num_docs, vector_size), dtype=np.float32)
        for idx in range(num_docs):
            docvec = matutils.unitvec(model.docvecs.doctag_syn0[idx, :])
            docvecs[idx, :] = docvec
        return docvecs


def tfidf_features(model, vocab, docs):
    bow_corpus = vocab.transform(docs, transform_to='bow', with_unk=False)
    vlen = len(vocab)-1 # remove UNK count
    data, row, col = list(), list(), list()
    
    for idx, bow in enumerate(bow_corpus):
        colvals, datavals = list(), list()
        
        for colval, dataval in model[bow]:
            colvals.append(int(colval))
            datavals.append(float(dataval))
        
        row += [idx] * len(colvals)
        col += colvals
        data += datavals
    
    row, col, data = np.array(row), np.array(col), np.array(data)
    X_sparse_mat = sparse.coo_matrix((data, (row, col)), shape=(idx+1, vlen)).asformat('csr')
    
    return X_sparse_mat

join = os.path.join

def main(preprocessed_docs_fname, preprocessed_sents_fname, out_dir,
        tfidf=True, w2v=True, d2v=False):
    
    if tfidf:
        # prepare tf-idf features, dump results and clear memory
        docs = CorpusIterable(preprocessed_docs_fname)
        
        vocab = Vocab()
        vocab.set(docs)
        vocab.prune(no_above=0.5)
        
        model = features.tfidf_model(vocab.dictionary)
        X = tfidf_features(model, vocab, docs)
        
        # save
        vocab.save(join(out_dir, 'tfidf_vocab'))
        model.save(join(out_dir, 'tfidf_model'))
        sparse.save_npz(join(out_dir, 'tfidf_X.npz'), X)
        # free memory
        del vocab, model, X
    
    if w2v:
        # prepare word2vec features
        docs = CorpusIterable(preprocessed_sents_fname)
        model = features.w2v_model(docs) # default: word2vec algo (see features.py for other options)
        docs = CorpusIterable(preprocessed_docs_fname)
        X = w2v_features(model, docs)
        
        model.save(join(out_dir, 'w2v_model'))
        np.save(join(out_dir, 'w2v_X.npy'), X)
        del model, X
    
    if d2v:
        # prepare doc2vec features
        docs = CorpusIterable(preprocessed_docs_fname, is_d2v=True)
        model = features.d2v_model(docs)
        X = d2v_features(model)
        
        model.save(join(out_dir, 'd2v_model'))
        np.save(join(out_dir, 'd2v_X.npy'), X)
        del model, X


if __name__=='__main__':
    preprocessed_docs_fname = sys.argv[1]
    preprocessed_sents_fname = sys.argv[2]
    out_dir = sys.argv[3]
    main(preprocessed_docs_fname, preprocessed_sents_fname, out_dir)
