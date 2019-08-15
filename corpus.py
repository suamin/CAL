# -*- coding: utf-8 -*-

import os
import json
import pickle
from fnmatch import fnmatch
from collections import OrderedDict
from utils import mod_os_walk, dir_walk, read_doc_lines, read_text_file


class FilesStream:

    def __init__(self, base_dir, sub_dirs=True, extensions=None, patterns=None):
        self.base_dir = base_dir
        self.sub_dirs = sub_dirs
        self.extensions = [ext.lower() for ext in extensions] if extensions else extensions
        self.patterns = patterns 
      
    def __iter__(self):
        if self.sub_dirs:
            for path in mod_os_walk(self.base_dir, self.extensions, self.patterns):
                yield path
        else:
            for item in os.scandir(self.base_dir):
                if not item.is_dir():
                    if self.extensions:
                        _, ext = os.path.splitext(item.path)
                        if ext.lower() in self.extensions:
                            yield item.path
                    elif self.patterns:
                        for p in self.patterns:
                            if fnmatch(item.path, p):
                                yield item.path
                    else:
                        yield item.path
  
    def search(self, fname):
        return next(dir_walk(self.base_dir, fname))


dir_iter = FilesStream


class TextsStreamReader(FilesStream):

    def __init__(self, base_dir, sub_dirs=True, read=True, as_lines=True, match_ids=None, encoding='utf-8'):  
        self.read = read
        self.as_lines = as_lines
        self.match_ids = match_ids
        self.encoding = encoding
        super().__init__(base_dir, sub_dirs, ['.txt'])
  
    def __iter__(self):
        for path in super().__iter__():
            if self.match_ids:
                doc_id = os.path.split(path)[1][:-4]
                if not doc_id in self.match_ids:
                    continue
            if self.read:
                if self.as_lines:
                    lines_iter = read_doc_lines(path, encoding=self.encoding)
                    yield lines_iter, path
                else:
                    raw_text = read_text_file(path, encoding=self.encoding)
                    yield raw_text, path
            else:
                yield path
    
    def get_doc_text(self, doc_id, encoding='utf-8'):
        path = self.search(doc_id+'.txt')
        if path: return read_text_file(path, encoding=encoding)
