# -*- coding: utf-8 -*-

import os
import re
import codecs
from fnmatch import fnmatch


def read_text_file(fname, encoding='utf-8'):
    """Read complete file data"""
    with codecs.open(fname, mode='r', encoding=encoding, errors='ignore') as rf:
        data = rf.read()
    return data


def read_file_lines(fname, mode='r', encoding='utf-8'):
    """Yields file contents as lines"""
    with codecs.open(fname, mode=mode, encoding=encoding, errors='ignore') as rf:
        for line in rf:
            yield line


class DocLinesIter:
    def __init__(self, fname, mode='r', encoding='utf-8'):
        self.fname = fname
        self.mode = mode
        self.encoding = encoding

    def __iter__(self):
        lines = read_file_lines(self.fname, self.mode, self.encoding)
        for l in lines:
            l = l.strip()
            if l:
                yield l


read_doc_lines = DocLinesIter


def dir_walk(path, fname):
    for i in os.scandir(path):
        if i.is_dir():
            yield from dir_walk(i.path, fname)
        elif i.name == fname:
            yield i.path


def mod_os_walk(path, exts=None, patterns=None):
    for item in os.scandir(path):
        if item.is_dir():
            yield from mod_os_walk(item.path, exts, patterns)
        else:
            if exts:
                _, ext = os.path.splitext(item.path)
                if ext in exts:
                    yield item.path
            elif patterns:
                for p in patterns:
                    if fnmatch(item.path, p):
                        yield item.path
            else:
                yield item.path


def match_file_ext(filename, exts, ignore_case=True):
    """Returns the matched file extension.
    
    Parameters
    ----------
    filename : `str`
        File name whose extension is to be matched against.
    exts : `str` or `list` of `str`
        Expected extension name or sequence of extenion names.
    ignore_case: `bool`, optional
        If `True` lowers the file extension otherwise preserves
        original extension (default is `True`).
    
    Returns
    -------
        The matched extension of file from provided list.     
    
    Examples
    --------
    >>> match = match_file_ext("file.txt", ['.csv', '.txt'], True)
    .txt
    
    >>> match = match_file_ext("file.IMG", ['.txt', '.img', '.IMG'], False)
    .IMG
    """
    ext = os.path.splitext(filename)[1]
    ext = ext.lower() if ignore_case else ext
    if type(exts) == str:
        exts = [exts]
    
    for ext_ in exts:
        if ext_ == ext:
            return ext_
    return None
