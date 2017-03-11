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


class Corpus:
    """
    Given CSV representation of the data, provide programmatic access
    """

    def __init__(self, csv_file):
        self._sentences = defaultdict(dict)

        dr = DictReader(open(csv_file))
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
