# -*- coding: utf-8 -*-

from collections import defaultdict
from gensim.corpora import Dictionary
from six import iteritems, string_types


class Vocab:
    
    def __init__(self):
        self.dictionary = Dictionary()
        self.dictionary.token2id['<UNK>'] = -1 
        self.dictionary.id2token[-1] = '<UNK>'
        self.dictionary.dfs[-1] = 0
    
    def set(self, corpus, prune_at=2000000):
        self.dictionary.add_documents(corpus, prune_at)
    
    def prune(self, **kwargs):
        # it is best if pruning is applied after all the updates
        # otherwise dropped tokens during pruning, seen in update 
        # docs will produce wrong counts
        if self.dictionary.dfs == {}:
            raise ValueError('no vocab to filter; build vocab first')
        no_below = kwargs.get('no_below', 5)
        no_above = kwargs.get('no_above', 0.7)
        keep_n = kwargs.get('keep_n', 100000)
        keep_tokens = kwargs.get('keep_tokens', None)
        if keep_tokens:
            keep_tokens.append('UNK')
        else:
            keep_tokens = ['UNK']
        preprune_count = sum([df for _, df in self.dictionary.dfs.items()])
        self.dictionary.filter_extremes(no_below, no_above, keep_n, keep_tokens)
        postprune_count = sum([df for _, df in self.dictionary.dfs.items()])
        self.dictionary.dfs[-1] = preprune_count - postprune_count
        # add UNK back (gets pruned due to 0 initial val)
        self.dictionary.token2id['<UNK>'] = -1 
        self.dictionary.id2token[-1] = '<UNK>'
    
    def update(self, docs, prune_at=2000000):
        self.add_documents(docs, prune_at)
    
    def transform(self, docs, transform_to='ids', with_unk=True):
        if transform_to == 'ids':
            for doc in docs:
                yield self.dictionary.doc2idx(doc)
        elif transform_to == 'bow':
            for doc in docs:
                if with_unk:
                    yield self.doc2bow(doc)
                else:
                    yield self.dictionary.doc2bow(doc)
        else:
            raise ValueError('unknwon transformation format')
    
    def fit_transform(self, docs, transform_to='ids', prune_at=2000000, filter_vocab=False, **kwargs):
        self.set(docs, prune_at)
        if filter_vocab:
            self.prune(**kwargs)
        yield from self.transform(docs, transform_to)
    
    def merge(self, other):
        self.dictionary.merge_with(other)
    
    def save(self, fname, as_text=False, sort_by_word=False):
        if as_text:
            self.dictionary.save_as_text(fname, sort_by_word)
        else:
            self.dictionary.save(fname)
    
    def load(self, fname, from_text=False):
        if from_text:
            self.dictionary = Dictionary.load_from_text(fname)
        else:
            self.dictionary = Dictionary.load(fname)
    
    def __len__(self):
        return len(self.dictionary)
    
    def __iter__(self):
        return iter(self.dictionary)
    
    def keys(self):
        return list(self.dictionary.token2id.values())
    
    def __str__(self):
        return str(self.dictionary)
    
    def __getitem__(self, tokenid):
        return self.dictionary[tokenid]
    
    def doc2bow(self, document):
        # note: slight variation to BoW format conversion from gensim
        # to allow '<UNK>' tokens
        if isinstance(document, string_types):
            raise TypeError("doc2bow expects an array of unicode tokens on input, not a single string")
        
        # Construct (word, frequency) mapping.
        counter = defaultdict(int)
        for w in document:
            if w in self.dictionary.token2id:
                counter[self.dictionary.token2id[w]] += 1
            else:
                counter[-1] += 1
        
        # return tokenids, in ascending id order
        counter = sorted(iteritems(counter))
        return counter
