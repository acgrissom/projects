# cython: language_level=3, boundscheck=False
# cython: c_string_type=unicode, c_string_encoding=utf8
# -*- coding: utf-8 -*-

import sys
import os
lib_path = os.path.abspath('./lib/')
sys.path.append(lib_path)
import pyximport
pyximport.install()
import logging
from bleu import bleu, brevity_penalty


cdef class MultiBleu:
    cpdef int max_length
    cpdef float normalizer
    cpdef list references
    cpdef bint _cache
    cpdef dict _stored_bleu
    def __init__(self, max_length, references, cache=False):
        self.references = references
        self.max_length = max_length
        self.normalizer = 1.0 / float(max_length)

        self._cache = cache
        self._stored_bleu = {}

    cpdef float score(self, list sent):
        cdef float val = 0.0
        cdef int ii = 0
        cdef unicode key #should be unicode
        if self._cache:
            key = "|".join(sent)
            if key in self._stored_bleu:
                return self._stored_bleu[key]

        if len(sent):
            for ii in xrange(1, self.max_length + 1):
                #try:
                val += self.normalizer * bleu(ii, self.references,
                                                  sent, brevity=True)
                #except ValueError:
                #sys.stderr.write("ZeroDivsionError in multi_bleu.\n")
                #sys.stderr.write("ii: " + ii + "\n")
                # sys.stderr.write("sent: " + " ".join(sent) + "\n")
                # sys.stderr.write("references:\n")
                # for ref in self.references:
                #     sys.stderr.write(" ".join(ref) + "\n")
                    

        if self._cache:
            self._stored_bleu[key] = val

        return val


if __name__ == "__main__":
    s = "He went to the store"
    mb = MultiBleu(5, [s.split()])
    for ii in xrange(len(s.split())):
        part = s.split()[:ii]
        print(ii, part, mb.score(part))
