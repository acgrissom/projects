# cython: c_string_type=unicode, c_string_encoding=utf8, language_level=3
# -*- coding: utf-8 -*-o

from csv import DictReader
from collections import defaultdict
import sys


class Sentence:
    """
    A parallel sentence pair, split into verb / preverb portions
    """

    def __init__(self, line):
        self.raw = line
        self.preverb = line['preverb'].lower().split()
        self.verb = line['verb'].lower().split()
        self.en = line['en'].lower()
        self.id = int(line['id'])

    @property
    def de(self):
        return self.preverb + self.verb

    @property
    def source_length(self):
        return len(self.preverb) + len(self.verb)

    def observations(self):
        obs_preverb = []
        obs_verb = []

        for ii in self.preverb:
            obs_preverb.append(ii)
            yield (obs_preverb, obs_verb)

        for ii in self.verb:
            obs_verb.append(ii)
            yield (obs_preverb, obs_verb)

#Doesn't work with unicode
class UnicodeDictReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect='excel', encoding="utf-8", **kwds):
        self.encoding = encoding
        self.reader = DictReader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return {k: unicode(v, "utf-8") for k, v in row.iteritems()}

    def __iter__(self):
        return self

import codecs
class Corpus:
    # def UnicodeDictReader(utf8_data, **kwargs):
    #     csv_reader = DictReader(utf8_data, **kwargs)
    #     for row in csv_reader:
    #         yield {unicode(key, 'utf-8'):unicode(value, 'utf-8') for key, value in row.iteritems()}

    """
    Given CSV representation of the data, provide programmatic access
    """

    def __init__(self, csv_file):
        self._sentences = defaultdict(dict)

        #dr = DictReader(codecs.open(csv_file, 'r', encoding='utf-8', errors='ignore'))
        dr = UnicodeDictReader(open(csv_file, 'rb'))
        num = 0
        for ii in dr:
            self._sentences[ii['fold']][ii['id']] = Sentence(ii)
            num += 1

    @property
    def folds(self):
        return self._sentences.keys()

    def get_fold(self, fold):
        for ii in sorted(self._sentences[fold]):
            yield self._sentences[fold][ii]

    @property
    def size(self):
        return sum(len(self._sentences[x]) for x in self._sentences)

if __name__ == "__main__":
    filename = sys.argv[1]

    c = Corpus(filename)
    print("Size %i" % c.size)
    print("Folds %s" % str(c.folds))
    for ii in c.get_fold('test1'):
        print(ii.id, ii.verb)
