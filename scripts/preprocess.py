# -*- coding: utf-8 -*-

import os
import sys
import codecs

from corpus import TextsStreamReader
from preprocess import preprocess_text
from gensim import utils


def main(data_dir, out_dir):
    
    docid2path = dict()
    
    # iterable of (doctext, docpath) tuple
    reader = TextsStreamReader(data_dir, as_lines=False)
    outfile = codecs.open(os.path.join(out_dir, 'processed_enron_docs_as_lines.txt'), 'w', 'utf-8', 'ignore')
    
    docid = 0
    opts = dict(sents=False, lower=True, stem=False,
                min_token_len=3, min_sent_len=4, remove_stops=True,
                filters=['strip_multiple_whitespaces', 'strip_tags',
                'strip_punctuation', 'split_alphanum', 'strip_numeric'])
    
    for doctext, docpath in reader:
        doctext = preprocess_text(doctext, **opts)
        # generator to list
        doctext = list(doctext)
        if doctext:
            # when sents=False, each document is returned as single sentence (first element),
            # where every element is a list of tokens
            doctext = doctext[0]
            if doctext:
                docid2path[docid] = docpath
                outfile.write(" ".join(doctext) + '\n')
                docid += 1
    
    outfile.close()
    utils.pickle(docid2path, os.path.join(out_dir, 'docid2path.pkl'))
    
    # create another file to hold sentences (useful for word2vec)
    outfile = codecs.open(os.path.join(out_dir, 'processed_enron_sents_as_lines.txt'), 'w', 'utf-8', 'ignore')
    opts['sents'] = True
    
    for doctext, _ in reader:
        docsents = preprocess_text(doctext, **opts)
        docsents = list(docsents)
        if docsents:
            for sent in docsents:
                if sent:
                    outfile.write(" ".join(sent) + '\n')
    
    outfile.close()


if __name__=='__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    main(data_dir, out_dir)

# call from base dir of repo e.g.
# >python -m scripts.preprocess data output