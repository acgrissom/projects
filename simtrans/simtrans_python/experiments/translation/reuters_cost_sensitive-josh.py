# coding: utf-8
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

import pyximport; pyximport.install()

import logging, time, pickle
from csv import DictWriter

#from searn_harness.test_optimal_policy import cvnwc_inputs
from searn_harness.cost_sensitive import Searn
from searn_harness.prediction import LangModNextWord, LangModVerbPredictor, SQLVerbPredictor
from searn_harness.corpus import Corpus
from searn_harness.omniscient_translator import OmniscientTranslator, Alignment
from searn_harness.policy import OptimalPolicy, State, VALID_ACTIONS, BatchPolicy, SentenceStateLattice, Policy, InterpolatePolicy
from searn_harness.parallel_corpus import ParallelInstance

ib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
import logging, pickle, sys
from csv import DictWriter
from searn_harness.prediction import LangModNextWord, SQLVerbPredictor, ConstantPredictor, LangModVerbPredictor, VWOAAVerbPredictor, LuceneNextWordPredictor
from searn_harness.corpus import Corpus
#from searn_harness.translation import CdecTranslator, OnlineCdecTranslator
from searn_harness.translation import SequenceTranslation, JoshuaTranslator
from searn_harness.policy import OptimalPolicy, State, VALID_ACTIONS, BatchPolicy, CommitPolicy, SentenceStateLattice, Policy, InterpolatePolicy
from searn_harness.parallel_corpus import ParallelInstance
from searn_harness.cost_sensitive import ClassifierPolicy, InstanceFactory
from searn_harness.filereaders import UntaggedJapaneseFileReader, PlainUnicodeFileReader, CdecParallelCorpusReader
from searn_harness.feature_extractor import JapaneseSentenceFeatureExtractor
from sentence_extraction import JapaneseUnparsedLastVerbSentenceExtractor

from lib import flags

if __name__ == "__main__":

    flags.define_int("bleu_width", 4, "Number of n-grams to consider")
    #flags.define_string("corpus", None, "Corpus for the data")
    #flags.define_string("bigram_file", None, "List of bigrams to consider")
    flags.define_string("degenerate_example", "exclude", "What to do with degenerate training examples")
    #flags.define_string("source_nextword_lm", None, "Language model for next word prediction")
    #flags.define_string("source_verb_lm_dir", None, "Directory containing language models for Verb prediction")
    #flags.define_string("kenlm_verb_lm", None, "KenLM file with verb predictions")
    #flags.define_string("verb_db", None, "Database containing verb predictions")
    #flags.define_string("source_full_lm", None, "Full Source Language model  ")
    #flags.define_string("svoc", None, "Source language vocabulary file")
    #flags.define_string("tvoc", None, "Target language vocabulary file")
    #flags.define_string("lexical", None, "Lexical translation prob table")
    #flags.define_string("target_lm", None, "Target language language model")
    #flags.define_string("cache_dir", None, "Where we store alignment pickles")
    #flags.define_string("align", None, "Alignment file from GIZA++")
    flags.define_string("output", "cost_sensitive-josh.classifier", "Where we write the learned classifier")
    flags.define_string("cost_type", None, "Cost to use for training: greedy or binary")
    #flags.define_string("fold", "test0", "The fold we run on")
    flags.define_string("feat", None, "The feature lookup")
    flags.define_string("lang", "ja", "Language")

    flags.InitFlags()

    logging.basicConfig(filename='cost_sensitive_test.log', level=logging.ERROR)

    # Load next word prediction
    #nw = LangModNextWord(flags.bigram_file, flags.source_full_lm)
    nw = ConstantPredictor()
    # Load verb prediction
    # if flags.verb_db:
    #     vp = SQLVerbPredictor(flags.verb_db)
    # else:
    #     vp = LangModVerbPredictor(flags.kenlm_verb_lm, flags.source_full_lm)

    vp = VWOAAVerbPredictor(unicode(os.path.abspath("corpora/kyoto/kyoto-train.ja-oaa.model")))
    # Load translation
    #ot = OmniscientTranslator(flags.svoc, flags.tvoc, flags.lexical, flags.target_lm, flags.corpus, lang=flags.lang)
    ot = JoshuaTranslator(unicode(os.path.abspath(u"data/reuters/joshua/tune")))
            
    #corpus = Corpus(flags.corpus)

    feature_extractor=JapaneseSentenceFeatureExtractor(ngram=2,
                                                       use_position=False,
                                                       count_case_markers=False,
                                                       normalize_verb=True,
                                                       case_marker_ngrams=2)
    
    sentence_extractor = JapaneseUnparsedLastVerbSentenceExtractor(feature_extractor=feature_extractor)

    jr = UntaggedJapaneseFileReader(u"data/reuters/reuters-cdec.dev", sent_extractor=sentence_extractor)
    er = PlainUnicodeFileReader(u"data/reuters/reuters-cdec.dev")
    #reader = CdecParallelCorpusReader(u"data/reuters/reuters-cdec.dev", jr, er)
    reader = CdecParallelCorpusReader(u"reuters.dev.sample", jr, er)

    
    trn = []
    #for ii in corpus.get_fold(flags.fold):
    _id = -1
    for line in reader:
        _id +=1 
        preverb = reader.left_context_tokens(line)
        if preverb is None or len(preverb) > 25:
            continue
        true_verb = [reader.left_label(line)]
        print "".join(preverb), "".join(true_verb)
        #ot.load_single_alignment(flags.cache_dir, ii.id)
        #trn += [ParallelInstance(preverb, ii.verb, ii.en.split(), ii.id)]
        en_toks = reader.right_text(line).split()
        trn += [ParallelInstance(preverb, true_verb, en_toks, _id)]
        #TODOtest, dev = 
    s = Searn(trn, trn, trn, nw, vp, ot, flags.bleu_width, flags.degenerate_example, flags.cost_type)
    if flags.feat:
        start = time.clock()
        with open(flags.feat, 'rb') as infile:
            s._feat_look = pickle.load(infile)
        print 'loading feature takes', time.clock() - start, 'seconds'
    else:
        start = time.clock()
        s.build_features()
        # These features are dumped with the policy
        print 'building feature takes', time.clock() - start, 'seconds'

    start = time.clock()
    learned_policy = s.train_policy(1)
    print 'learning policy takes', time.clock() - start, 'seconds'

    # Save policy in the last round of searn
    learned_policy._new.dump(flags.output)
    print 'Average reward on training set (Learned): ', \
       sum(s.evaluate_policy(trn, learned_policy)) / len(trn)
  
    opt_policy = OptimalPolicy(nw, vp, int(flags.bleu_width), ot)
    reward = 0.0
    for example in trn:
       lat = opt_policy.find_optimal(example, forward_backward=False)
       reward += sum(s.evaluate_policy([example], opt_policy))
    print 'Average reward on training set (Optimal): ', \
       reward / len(trn)

    exit()

    for oo in s.evaluate_policy(trn, opt_policy):
        print 'optimal policy', oo

    batch = BatchPolicy()
    for bb in s.evaluate_policy(trn, batch):
        print 'batch policy', bb


