# -*- coding: utf-8 -*-

from gensim.parsing.preprocessing import (
    strip_punctuation, strip_tags, strip_short,
    strip_numeric, strip_non_alphanum, strip_multiple_whitespaces,
    split_alphanum, stem_text
)

from gensim.summarization.textcleaner import get_sentences
from stopwords import get_stops

STOPWORDS = set(get_stops(['en']))

funcs = {
    'strip_multiple_whitespaces': strip_multiple_whitespaces,
    'strip_tags': strip_tags,
    'strip_punctuation': strip_punctuation,
    'strip_non_alphanum': strip_non_alphanum,
    'split_alphanum': split_alphanum,
    'strip_numeric': strip_numeric
}


def basic_preprocessing(text, sents=False, lower=False, stem=False, min_token_len=3, min_sent_len=4, remove_stops=False,
                        stops=STOPWORDS, filters=['strip_multiple_whitespaces', 'strip_punctuation']):
    # EDT export specific
    text = text.replace('\x00', '')
    text = text.replace('\r\n', '\n')
    
    # note: filters will be applied in order
    if sents:
        sents = get_sentences(text)
    else:
        sents = [text]
    
    for s in sents:
        s = s.strip()
        if lower:
            s = s.lower()
        if stem:
            s = stem_text(s)
        
        for f in filters:
            s = funcs[f](s)
        
        # naive word tokenization
        s = s.split()
        tmp = list()
        for t in s:
            t = t.strip()
            if t:
                if remove_stops and stops:
                    if t not in stops:
                        tmp.append(t)
                    else:
                        continue
                else:
                    tmp.append(t)
            else:
                continue
        s = tmp
        
        if len(s) < min_sent_len:
            yield list()
        else:
            yield s

preprocess_text = basic_preprocessing
