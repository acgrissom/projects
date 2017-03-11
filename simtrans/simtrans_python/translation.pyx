# cython: language_level=3, boundscheck=False
# cython: c_string_type=unicode, c_string_encoding=utf8
# -*- coding: utf-8 -*-
import sys
import os
lib_path = os.path.abspath('./lib/')
sys.path.append(lib_path)
sys.path.append('./searn_harness')
import string
from collections import defaultdict
from csv import DictReader
import logging
from math import log
import pickle
from filereaders import UntaggedJapaneseFileReader
import kenlm
from nltk import FreqDist
import pyximport
pyximport.install()

#from lib.sets import dict_sample
import sets

from corpus import Corpus

UNKNOWN_WORD = "UNK"

cdef class DummyLM:
    """
    Utility cpdef class for testing functions that use language models.
    """

    cpdef score(x):
        return - len(x)


cdef class Alignment:
    """
    Cdef Class to store an alignment for a single sentence.
    """

    def __init__(self, _id, target):
        """
        Parses the output of GIZA into a form that can be stored in the object.
        """

        self.id = id
        self.target = target.split()

        # Words stores the source words indexed by their position
        self.words = {}
        self.edges = defaultdict(dict)

        self._monotone = {}

    cpdef add_word(self, position, entry):
        """
        Add the alignment for a single source word
        """

        word, edges = entry.split("({")

        self.words[position] = word.strip()

        edges = map(int, edges.split())

        if edges:
            for ii in edges:
                # Alignment is indexed from 1
                assert len(self.target) > ii - 1, \
                    "Index %i out of bounds for %s" % (ii - 1, str(self.target))
                self.edges[position][ii] = self.target[ii - 1]

    cpdef printable(self):
        val = ""
        for ii in sorted(self.words):
            val += "%s: %s" % (self.words[ii], str(self.edges[ii]))
        return val

    def get_edges(self, position):
        for ii in self.edges[position]:
            yield ii

    cpdef get_word(self, position):
        return self.words[position]

    cdef print_coverage(self, cor, hyp):
        val = "COR: "
        for ii in cor:
            val += "\t%s:%s" % (self.words[ii],
                                ",".join(self.target[x] for x \
                                             in sorted(cor[ii])))
        val += "\nHYP:"
        for ii in hyp:
            val += "\t%s:%s" % (self.words[ii],
                                ",".join(self.target[x] for x \
                                             in sorted(cor[ii])))

        return val

    def build_monotone(self):
        """
        Initialize a dictionary that maps target words to their location in a monotone reordering.
        """

        unmatched_target = self.edges[0]
        target_length = None

        for ii in sorted(x for x in self.edges if x > 0):
            for jj in self.edges[ii]:
                # First if there are unaliged target words before this word, add them in
                preceeding_words = []
                target_index = jj - 1
                while target_index not in self._monotone and target_index in unmatched_target:
                    preceeding_words = [target_index] + preceeding_words
                    target_index -= 1

                for kk in preceeding_words:
                    self._monotone[kk] = len(self._monotone) + 1

                # Now Add the word itself
                self._monotone[jj] = len(self._monotone) + 1

                target_index = jj + 1
                while target_index not in self._monotone and target_index in unmatched_target:
                    self._monotone[target_index] = len(self._monotone) + 1
                    target_index += 1

        self._reverse_monotone = {0: 0}
        for ii in self._monotone:
            self._reverse_monotone[self._monotone[ii]] = ii

    cpdef reverse_monotone(self, position):
        """
        Given a monotone position, give its position under oritingal target order.
        """
        if not self._monotone:
            self.build_monotone()

        return self._reverse_monotone[position]


    cpdef monotone(self, position):
        """
        Given a target position, give its position under a monotone
        reordering.
        """
        if not self._monotone:
            self.build_monotone()
        return self._monotone[position]

cdef class Translation:
    cpdef public dict _words
    cpdef public dict _probs
    """
    Cdef Class to represent a translation.
    """

    cpdef initialize(self):
        self._words = {}
        self._probs = {}

    def __init__(self, dict words=None, dict probabilities=None):
        """

        """

        if words and probabilities:
            self._words = words
            self._probs = probabilities
        else:
            self.initialize()

    cpdef add(self, target_pos, word, prob):
        self._words[target_pos] = word
        self._probs[target_pos] = prob

    cpdef as_list(self):
        val = []
        for ii in sorted(self._words):
            val.append(self._words[ii])
        return val


cdef class SequenceTranslation(Translation):
    """
    Cdef Class to represent a translation composed from earlier translations
    """
    cpdef _prev
    #cpdef _words
    cpdef int _index

    def __init__(self, position, previous=None, current=None):
        self._prev = previous
        self._index = position
        self.initialize()

        if previous:
            assert isinstance(previous, SequenceTranslation), \
                "Got %s" % str(type(previous))
            for ii in previous._words:
                self._words[ii] = previous._words[ii]
                self._probs[ii] = previous._probs[ii]

        if current:
            assert isinstance(current, Translation)
            print (u" ".join(current.as_list()))
            for ii in current._words:
                if not ii in self._words:
                    self._words[ii] = current._words[ii]
                    self._probs[ii] = current._probs[ii]

    def translation_histogram(self):
        if self._prev:
            for ii, tt in self._prev.translation_histogram():
                yield ii, tt

        yield self._index, self.as_list()


cdef class Translator(object):
    """
    Defines interface for translations
    """
    
    def translate(self, _id, words,  verbs, num_cands=1):
        raise NotImplementedError

    def top(self, _id, words, list verbs):
        raise NotImplementedError


class StubTranslator(Translator):

    def __init__(self):
        self.stack = []

    def  add(self, translation):
        self.stack.append(translation)

    def translate(self, _id, words, list verbs, num_cands=1):
        for ii in xrange(num_cands):
            trans = self.stack.pop()
            probs = dict((x, 1.0) for x in trans)
            yield probs, trans

import subprocess, codecs
cdef class JoshuaTranslator(Translator):
    cpdef unicode lang_pack_dir
    cdef unicode joshua_command
    cpdef unicode joshua_root
    cdef unicode old_pwd
    cpdef joshua_process
    def __init__(self,
                  unicode lang_pack_dir,
                  unicode mem=u"8g",
                  int topn=0):
        self.lang_pack_dir = lang_pack_dir
        self.old_pwd = unicode(os.getcwd())
        os.chdir(lang_pack_dir)
        cdef unicode joshua_dir = unicode(os.environ["JOSHUA"])
        self.joshua_command = u"java -Xmx" + mem
        self.joshua_command += u" -cp JOSHUA/class:JOSHUA/thrax/bin/thrax.jar:JOSHUA/lib/berkeleylm.jar:JOSHUA/lib/junit-4.10.jar"
        self.joshua_command += u" -Dfile.encoding=utf8 "
        self.joshua_command += u" -Djava.util.logging.config.file=JOSHUA/logging.properties"
        self.joshua_command += u" -Djava.library.path=JOSHUA/lib"
        self.joshua_command += u" joshua.decoder.JoshuaDecoder"
        self.joshua_command += u" -c " + lang_pack_dir + "/joshua.config"
        self.joshua_command += u" -top-n " + unicode(topn)
        #self.joshua_command += u" -output-format \"%t\""
        self.joshua_command += u" -output-format \"%s\""

        self.joshua_command += u" -v 0"
        self.joshua_command = self.joshua_command.replace("JOSHUA",joshua_dir)
        print(self.joshua_command)
        self.initialize()
        #self.test_translate()

    cpdef initialize(self):
        self.joshua_process = subprocess.Popen(self.joshua_command.split(),
                                           stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE,
                                           #stderr=subprocess.PIPE,
                                           close_fds=True,
                                           universal_newlines=True,
                                           shell=False)
        self.joshua_process.stdin = codecs.getwriter('utf8')(self.joshua_process.stdin)
        print("************Chainging directory to",self.old_pwd)
        os.chdir(self.old_pwd)

    cpdef Translation translate(self, unicode sentence):
        self.joshua_process.stdin.write(sentence)
        self.joshua_process.stdin.write("\n")
        self.joshua_process.stdin.flush()
        self.joshua_process.stdout.flush()
        #self.joshua_process.stderr.flush()
        cdef unicode translation = unicode(self.joshua_process.stdout.readline().strip().decode('utf8'))
        #sys.stdout.write("\033[K")
        #sys.stdout.write(translation + "\r")
        cdef dict words = dict()
        cdef dict probs = dict();
        cdef int i = 0
        for w in translation.split():
            words[i] = w
            probs[i] = 1.0
            i += 1
        t = Translation(words, probs)
        return t

    
    """I'm assuming that this is a list of lists of verbs with index 0 the best one.
    """
    cpdef Translation top(self, _id, list words, list verbs):
        cdef unicode sentence
        if len(verbs) > 0:
            sentence = u" ".join(words + [u" ".join(verbs[0])])
        else:
            sentence = u" ".join(words)
        return self.translate(sentence)


    cpdef test_translate(self):
        print(u"command",self.joshua_command)
        #self.joshua_process.stdin.write(u"Editado por Héroes de Papel, se trata de un libro que homenajea a la saga de videojuegos creada por Yu Suzuki, siendo un obsequio tanto para los fans de la misma como un reconocimiento crítico a uno de los mayores logros en construcción de mundos virtuales.\n".encode("utf8"))
        cdef unicode test_sentence = u"Editado por Héroes de Papel, se trata de un libro que homenajea a la saga de videojuegos creada por Yu Suzuki, siendo un obsequio tanto para los fans de la misma como un reconocimiento crítico a uno de los mayores logros en construcción de mundos virtuales"
        #print(u"Sending sentence",test_sentence)
        self.joshua_process.stdin.write(test_sentence)
        self.joshua_process.stdin.write("\n")
            
        #self.joshua_process.stdin.write("me llamo alvino\n")
        self.joshua_process.stdin.flush()
        self.joshua_process.stdout.flush()
        #self.joshua_process.stderr.flush()
        out = self.joshua_process.stdout.readline().strip()
        #err = self.joshua_process.stderr.readline().strip()
        print(u"****************Test Translation::",out)
        #print u"err:",err

#import cdec
import multiprocessing
import gzip
# cdef class CdecTranslator(Translator):
#     cpdef public decoder
#     cpdef int processes
#     cpdef unicode grammars_dir
#     def __init__(self, int processes=multiprocessing.cpu_count(),
#                  unicode formalism=u'scfg',
#                  dict weights=None,
#                  unicode grammars_dir=None
#                  ):
#         self.decoder = cdec.Decoder(formalism=formalism)
#         self.grammars_dir = grammars_dir
#         if weights is not None:
#             self.decoder.read_weights(weights)
#         else:
#             sys.stderr.write("WARNING: Weights not set for decoder.")
#         sys.stderr.write("Decoder initialized.\n")


#     #TODO(acg) how is num_cands supposed to fit into this?
#     def  translate(self, list words, list verbs, int sentence_num, int num_cands=1):
#         cdef dict probs = {}
#         cdef list translations = list()
#         translations.append(self.translate_with_grammar_file(words, verbs, self.grammars_dir + u"/" + unicode(sentence_num), num_cands=num_cands))
#         for ii in xrange(num_cands):
#             trans = translations.pop()
#             probs = dict((x, 1.0) for x in trans)
#             yield probs, trans

    
#     cpdef translate_with_grammar_file(self, list words, list verbs, unicode grammar_file, int num_cands=1):
#         cpdef unicode sentence = words.join(u" ")
#         if grammar_file.endswith("gz"):
#             with gzip.open(grammar_file) as f:
#                 self.grammar = f.read()
#                 sys.stderr.write(u"Done.\n")
#         else:
#             with open(grammar_file) as f:
#                 self.grammar = f.read()

#         if num_cands == 1:
#             return self.decoder.translate(sentence).viterbi()

#         cpdef list kbest_sentences = list()
#         forest = self.decoder.translate(sentence, grammar=self.grammar)
#         kbest = forest.kbest(num_cands)
#         cdef unicode s
#         for s in kbest:
#             t = Translation(s.split(), None)
#             kbest_sentences.append(t)
#         return kbest_sentences

#     cpdef top(self, _id, list words, list verbs, unicode grammar_file):
#         return self.translate(words, verbs, grammar_file)


# """
# CDEC Translator that does not need a grammar file for every

#     sentence, but generates grammars on the fly.  Slower than the
#     alternative, but can generalize to unseen test sentences.

#     The config file is the output of running (for example): python -m
#     cdec.sa.compile -a corpus.es-en.gdfa -b corpus.es-en --online -o sa >
#     sa.ini

#     The sa.ini file would then be the config file used in the constructor.
# """
# cdef class OnlineCdecTranslator(Translator):
#     cpdef public decoder
#     cpdef extractor
#     cpdef int processes
#     def __init__(self, int processes=multiprocessing.cpu_count(),
#                  unicode formalism=u'scfg',
#                  unicode config_file=u'sa.ini',
#                  feature_function=None,
#                  dict weights=None,
#                  add_pass_through_rules=True,
#                  ):
#         self.extractor = cdec.sa.GrammarExtractor(config_file)
#         self.decoder = cdec.Decoder(formalism=formalism)
#         if weights is not None:
#             self.decoder.read_weights(weights)
#         else:
#             sys.stderr.write("WARNING: Weights not set for decoder.")
#         sys.stderr.write("Decoder initialized.\n")

#     cpdef  translate(self, list words, list verbs=[], int num_cands=1):
#         cpdef unicode sentence = u" ".join(words)
#         cdef unicode s = u""
#         grammar = self.extractor.grammar(sentence)
#         if num_cands == 1:
#             d_result = self.decoder.translate(sentence, grammar=grammar)
#             s = d_result.viterbi() #for debugging
#             t = Translation(s.split(), None)
#             return t
#         cpdef list kbest_sentences = list()
#         forest = self.decoder.translate(sentence, grammar=grammar)
#         kbest = forest.kbest(num_cands),
#         for s in kbest:
#             t = Translation(s.split(), None)
#             kbest_sentences.append(t)
#         return kbest_sentences

#     cpdef top(self, _id, list words, list verbs=[]):
#         cpdef unicode sentence = u" ".join(words)
#         return self.translate(words)

        
class MemoTranslator(Translator):
    """
    Utility cdef class for testing that stores canned translations for specific
    inputs.
    """

    def __init__(self):
        self._lookup = {}

    def add(self, prefix, verb, val):
        key = "%s|%s" % ("~".join(prefix), "~".join(verb))

        assert isinstance(val, list), "Add translations as a list; " + \
            "this function will store them as a dictionary for later lookup"

        trans = Translation()
        for ii, ww in enumerate(val):
            trans.add(ii, ww, 0.0)

        self._lookup[key] = trans

    def top(self, _id, prefix, verb):
        key = "%s|%s" % ("~".join(prefix), "~".join(verb))
        if key not in self._lookup:
            print("MISSING KEY")

        val = self._lookup[key]
        return val

    def translate(self, prefix, verb, num_candidates=1):
        key = "%s|%s" % ("~".join(prefix), "~".join(verb))
        for ii in xrange(num_candidates):
            prob = {}
            yield self._lookup[key]


cdef class OmniscientTranslator(Translator):
    """
    Simple decoder that reads back perfect translations when the input is
    person, using Model 1 lexical models otherwise.
    """

    def __init__(self, source_voc, target_voc, word_trans, \
                 target_lm, pos_file, limit=-1, contiguous=True, lang="de"):

        """

        @contiguous: Only provide contiguous translations
        """
        self._contiguous = contiguous
        self._lexical = defaultdict(FreqDist)
        if word_trans:
            self.read_lexical(source_voc, target_voc, word_trans)

        self._target_lm = DummyLM()
        if target_lm:
            self._target_lm = kenlm.LanguageModel(target_lm)

        self._aligns = {}

        self._cache = {}
        self._cache_id = -1

        self._limit = limit

        self._lang = lang

        # The source positions of verbs
        self._verbs = defaultdict(list)
        if pos_file:
            self.read_verbs(pos_file)

        self.clear_random()

    # Convenience function for easier unit testing
    cpdef add_random_stub(self, val):
        self._random_stub.append(val)

    cpdef clear_random(self):
        self._random_stub = []

    cpdef add_verb(self, _id, pos):
        assert not pos in self._verbs[id], "Sent already has that verb"
        self._verbs[id].append(pos)

    cpdef first_verb(self, id):
        """
        Returns the first verb in the sentence.
        """

        if self._verbs[id]:
            return min(self._verbs[id])
        else:
            return -1

    cpdef read_verbs(self, pos_csv):
        for ii in DictReader(open(pos_csv)):
            # How many words were in the original sentence?
            _id = int(ii["id"])
            # We're using 1-indexing, which is not provided by the length
            for ii in range(len(ii["preverb"].split()) + 1,
                            len(ii[self._lang].split()) + 1):
                self.add_verb(_id, ii)

    cpdef add_alignment(self, sent_id, source_index, target, align_entry):
        if not sent_id in self._aligns:
            self._aligns[sent_id] = Alignment(sent_id, target)

        self._aligns[sent_id].add_word(source_index, align_entry)

    cpdef dump_all_alignments(self, cache_directory):
        for ii in self._aligns:
            filename = "%s/%i.pkl" % (cache_directory, ii)
            logging.debug("Writing pickle to %s", filename)
            o = open(filename, 'w')
            pickle.dump(self._aligns[ii], o, pickle.HIGHEST_PROTOCOL)

    cpdef load_single_alignment(self, cache_directory, align_id):
        """
        We write out all our alignments to a cache directory, so
        """

        # If it's already loaded, skip this sentence
        if not align_id in self._aligns:
            self._aligns[align_id] = \
                pickle.load(open("%s/%i.pkl" % (cache_directory, align_id)))

        return self._aligns[align_id]

    cpdef read_all_alignments(self, alignment_file):
        infile = open(alignment_file)

        line = 1
        metadata = infile.readline()
        while metadata:
            target = infile.readline()
            source = infile.readline()

            if not target.startswith("NULL "):
                target = "NULL " + target

            for pos, entry in enumerate(source.strip().split("})")):
                if not entry:
                    continue

                self.add_alignment(line, pos, target, entry)

            metadata = infile.readline()
            line += 1

            if line % 100 == 0:
                print("Reading alignment %i" % line)

            if self._limit > 0 and line > self._limit:
                break

    def read_vocab(self, filename):
        d = dict((int(id), word) for _id, word, count in \
                  map(string.split, open(filename).readlines()))
        d[0] = "NULL"

        return d

    cpdef add_lexical(self, src, tgt, prob):
        self._lexical[src][tgt] = prob

    cpdef read_lexical(self, source_voc, target_voc, word_trans):
        sw = self.read_vocab(source_voc)
        tw = self.read_vocab(target_voc)

        for ii in open(word_trans):
            ii, jj, prob = ii.split()
            self.add_lexical(sw[int(ii)], tw[int(jj)], float(prob))

    cpdef word_candidate(self, src_word):
        if self._random_stub:
            return sets.dict_sample(self._lexical[src_word],
                               self._random_stub.pop())
        else:
            if src_word in self._lexical:
                return sets.dict_sample(self._lexical[src_word])
            else:
                logging.log(logging.DEBUG, "%s not in lexical lookup" % src_word)
                return None


    cpdef get_alignment(self, sent_id):
        return self._aligns[sent_id]

    cpdef reconstructed_source(self, _id, prefix, verbs):
        verb_pos = self.first_verb(id)
        if verbs and verb_pos != -1:
            reconstruction = prefix[:]
            for ii in xrange(len(prefix), verb_pos):
                reconstruction.append(UNKNOWN_WORD)
            reconstruction += verbs[:]
            return reconstruction
        elif verbs:
            return prefix + list(verbs)
        else:
            return prefix

    cpdef generate_candidate(self, src, corr, hyp, algn):
        """
        Given a coverage vector, generate candidate translations.

        Returns two dictionaries.  The first gives the probability of each
        translation at a source-side index.  The second gives the translated
        words.
        """

        tgt = algn.target
        assert tgt[0] == "NULL", "Target sentence must begin with NULL: %s" \
            % " ".join(tgt)

        assert src == [] or src[0] != "NULL", \
            "We add NULL so you don't have to"
        src = ["NULL"] + src

        trans = Translation()

        for ii in hyp:
            # We can get a "too big position if we've predicted a verb
            # where the sentence didn't originally have one.  Just
            # stick that to the end.
            position = ii + 1
            if position >= len(src):
                position = len(src) - 1

            src_word = src[position]

            logging.log(logging.DEBUG, "HYP Position: %i (orig: %i) is %s (%s)" % \
                        (position, ii, src_word, list(enumerate(src))))
            for jj in hyp[ii]:
                cand = self.word_candidate(src_word)
                try:
                    if cand != None:
                        trans.add(jj, cand, log(self._lexical[src_word][cand]))
                except ValueError:
                    print("Math domain error when looking up %s %s" % (src_word, cand))

        for ii in corr:
            for jj in corr[ii]:
                trans.add(jj, tgt[jj], 0.0)

        return trans

    cpdef coverage_vectors(self, _id, src_wrd, src_vrb, contig=True):
        correct_coverage = defaultdict(set)
        hypothesis_coverage = defaultdict(set)

        assert src_wrd == [] or src_wrd[0] != "NULL", \
            "We add NULL so you don't have to"

        # If this sentence doesn't actually have a verb, just treat it as a tacked on string
        if self.first_verb(id) <= 0 and src_vrb:
            src_verb = ()
            src_wrd = src_wrd + list(src_vrb)

        # Build coverage vectors so we know what to translate
        # After including NULL, we index starting from 1
        positions = [(0, "NULL")] + list(enumerate(src_wrd, 1))

        # We now add the verbs.  We index the verb either for where
        # the verb should be or from the first open slot
        verb_position = self.first_verb(id)
        if verb_position > -1:
            positions += list(enumerate(src_vrb, self.first_verb(id)))

        algn = self.get_alignment(id)
        logging.log(logging.DEBUG, "POSITIONS: %s" % str(positions))
        for ii, ww in positions:
            # logging.log(logging.DEBUG, "POS: %i\tWRD: %s\tALG: %s" % (ii, ww, algn.get_word(ii)))
            for jj in algn.get_edges(ii):
                if algn.get_word(ii) == ww:
                    correct_coverage[ii].add(jj)
                else:
                    hypothesis_coverage[ii].add(jj)

        # If the user wanted us to restrict to contiguous spans, do that
        # filtering now
        if contig:
            correct_coverage, hypothesis_coverage = \
                self.contiguous_filter(correct_coverage, hypothesis_coverage, algn)

        return correct_coverage, hypothesis_coverage

    def translate(self, _id, words, verbs, num_cands=1):
        """
        Given words and verbs (verbs assumed to be at end of sentence),
        generate an iterator over translation candidates (not sorted).  Yields
        translation and probability.

        If correct inputs are given, gives true translation with probability
        1.0.  Otherwise, generates candidates from lexical translation
        probabilities *in the place* of the true words.
        """

        correct_coverage, hypothesis_coverage = \
            self.coverage_vectors(_id, words, verbs, self._contiguous)
        align = self.get_alignment(id)

        logging.log(logging.DEBUG, "COR: %s HYP: %s" % (correct_coverage, hypothesis_coverage))

        # Check to see if we've already translated a correct sentence
        if not hypothesis_coverage:
            logging.log(logging.DEBUG, "Deterministic translation")
            key = "%s|%s" % ("~".join(words), "~".join(verbs))

            if self._cache_id != id:
                self._cache = {}
                self._cache_id = id
            if not key in self._cache:
                recon_src = self.reconstructed_source(_id, words, verbs)
                self._cache[key] = self.generate_candidate(recon_src, correct_coverage,
                                          hypothesis_coverage, align)
            yield self._cache[key]
        else:
            logging.log(logging.DEBUG, "Random translation")
            recon_src = self.reconstructed_source(_id, words, verbs)

            for ii in xrange(num_cands):
                yield self.generate_candidate(recon_src, correct_coverage,
                                              hypothesis_coverage, align)

    cpdef top(self, _id, words, verbs, num_cands=3):
        best_prob = float("-inf")
        best_trans = None

        assert num_cands >= 1, "Must generate at least one candidate"

        logging.log(logging.DEBUG, "Translating SNT: %i WRD: %s VRB: %s" % \
                    (_id, str(words), str(verbs)))
        for tt in self.translate(_id, words, verbs, num_cands):
            probs = tt._probs
            prob = sum(probs.values())
            if prob > best_prob:
                best_prob = prob
                best_trans = tt

        #  assert Translation.check_trans(val)
        return best_trans

    def admit_contiguous(self, target_present, alignment):
        """
        This function doesn't use the alignment argument, but the monotone sublcass does, so don't remove it to simplify the interface.

        The counter-intuitive convention is if the return is empty, then everything is contiguous.
        """
        if target_present:
            max_index = max(target_present)
            missing = set(range(1, max_index)) - target_present
            if missing:
                return set(xrange(min(missing)))
            else:
                return set([])
        else:
            return set([])

    def contiguous_filter(self, correct, hypothesis, alignment):
        # Find the gap between last words and the words present
        target_present = set()
        for ii in correct:
            for jj in correct[ii]:
                target_present.add(jj)
        for ii in hypothesis:
            for jj in hypothesis[ii]:
                target_present.add(jj)

        admit = self.admit_contiguous(target_present, alignment)

        if admit:
            for ii, vv in correct.items():
                correct[ii] = set(x for x in vv if x in admit)
                if not correct[ii]:
                    del correct[ii]

            for ii, vv in hypothesis.items():
                hypothesis[ii] = set(x for x in vv if x in admit)
                if not hypothesis[ii]:
                    del hypothesis[ii]

        return correct, hypothesis

cdef class OmniscientMonotone(OmniscientTranslator):
    def __init__(self, source_voc, target_voc, word_trans, \
                 target_lm, pos_file, limit=-1, language="de"):
        # Only difference is that we must always filter contiguous
        super(OmniscientMonotone, self).__init__(source_voc, target_voc, word_trans, \
                 target_lm, pos_file, limit, contiguous=True, lang=language)

    def admit_contiguous(self, target_present, alignment):
        """
        This function doesn't use the alignment argument, but the monotone sublcass does, so don't remove it to simplify the interface.

        """

        transformed_target = {}
        for ii in target_present:
            transformed_target[alignment.monotone(ii)] = ii
        transformed_contiguous = super(OmniscientMonotone, self).\
          admit_contiguous(set(transformed_target.keys()), alignment)

        return set(alignment.reverse_monotone(x) for x in transformed_contiguous)

    cpdef generate_candidate(self, src, corr, hyp, algn):
        """
        Given a coverage vector, generate candidate translations.

        Returns two dictionaries.  The first gives the probability of each
        translation at a source-side index.  The second gives the translated
        words.
        """

        tgt = algn.target
        assert tgt[0] == "NULL", "Target sentence must begin with NULL: %s" \
            % " ".join(tgt)

        assert src == [] or src[0] != "NULL", \
            "We add NULL so you don't have to"
        src = ["NULL"] + src

        trans = Translation()

        for ii in hyp:
            try:
                # We can get a "too big position if we've predicted a verb
                # where the sentence didn't originally have one.  Just
                # stick that to the end.
                position = ii + 1
                if position >= len(src):
                    position = len(src) - 1

                src_word = src[position]

                for jj in hyp[ii]:
                    cand = self.word_candidate(src_word)
                    try:
                        if cand != None:
                            trans.add(algn.monotone(jj), cand,
                                  log(self._lexical[src_word][cand]))
                    except ValueError:
                        print (" math domain error")
            except IndexError:
                print (" list index out of range:")

        for ii in corr:
            for jj in corr[ii]:
                trans.add(algn.monotone(jj), tgt[jj], 0.0)

        return trans


if __name__ == "__main__":
    from lib import flags

    flags.define_string("align", None, "Alignment file from GIZA++")
    # flags.define_string("svoc", None, "Source language vocabulary file")
    # flags.define_string("tvoc", None, "Target language vocabulary file")
    # flags.define_string("lexical", None, "Lexical translation prob table")
    flags.define_string("tlm", None, "Target language language model")
    flags.define_string("pos", None, "Part of speech input file")
    flags.define_bool("generate_cache", True, "Store all alignments as pickle")
    flags.define_string("cache_dir", None, "Where we store alignment pickles")
    #flags.define_string("lang", "de", "Language")
    #flags.define_int("limit", -1, "How many alignments to load, -1 for all")

    flags.InitFlags()

    logging.basicConfig(filename='translate.log', level=logging.DEBUG)


    #translator = CdecTranslator(grammar_file_gz="/Users/alvin/research/corpora/reuters/split/reuters-all.grammars")
    #translator = OnlineCdecTranslator()
    translator = JoshuaTranslator()
    #ocrpus = Corpus(flags.pos)

    # If we're not generating the cache, put on a show and show all of the
    # incremental translations
    # if not flags.generate_cache:
    #     for ii in corpus.get_fold("test1"):
    #         try:
    #             ot.load_single_alignment(flags.cache_dir, ii.id)
    #             for pre, verb in ii.observations():
    #                 print("-------------------------")
    #                 print("ID:\t%i" % ii.id)
    #                 print("DE:\t%s" % " ".join(ii.de))
    #                 print("EN:\t%s" % ii.en)
    #                 print("OBS:\t%s\t|\t%s" % (" ".join(pre), " ".join(verb)))
    #                 print("TRANS:\t" + \
    #                           " ".join(ot.top(ii._id, pre, verb).as_list()))
    #         except IOError:
    #             print("Error loading alignment for %i (did you write pickles?)" \
    #                       % ii.id)

    corpus_filename = u""
    filereader = UntaggedJapaneseFileReader(corpus_filename)
    context = []
    label = u""
    for line in filereader:
        context = filereader.get_context_tokens(line)
        label = filereader.get_label(line)
        

