# coding: utf-8
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

import pyximport; pyximport.install()
import codecs
import logging, time, pickle
from csv import DictWriter

#from test_optimal_policy import cvnwc_inputs
from cost_sensitive import Searn
from prediction import LangModNextWord, LangModVerbPredictor, SQLVerbPredictor
from corpus import Corpus
from omniscient_translator import OmniscientTranslator, Alignment
from policy import OptimalPolicy, State, VALID_ACTIONS, BatchPolicy, SentenceStateLattice, Policy, InterpolatePolicy
from parallel_corpus import ParallelInstance

ib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
import logging, pickle, sys
from unicodecsv import DictWriter
from prediction import LangModNextWord, SQLVerbPredictor, ConstantPredictor, LangModVerbPredictor, VWOAAVerbPredictor, LuceneNextWordPredictor
from corpus import Corpus
#from translation import CdecTranslator, OnlineCdecTranslator
from translation import SequenceTranslation, JoshuaTranslator, JoshuaPackTranslator
from policy import OptimalPolicy, State, VALID_ACTIONS, BatchPolicy, CommitPolicy, SentenceStateLattice, Policy, InterpolatePolicy
from parallel_corpus import ParallelInstance
from cost_sensitive import ClassifierPolicy, InstanceFactory
from filereaders import UntaggedJapaneseFileReader, PlainUnicodeFileReader, CdecParallelCorpusReader, POSCSVFileReader
from feature_extractor import JapaneseSentenceFeatureExtractor
from sentence_extraction import JapaneseUnparsedLastVerbSentenceExtractor

logging.basicConfig(filename='/tmp/cost_sensitive_test.log', level=logging.ERROR)
#sys.stdout = codecs.getwriter("UTF-8")(sys.stdout)
starting_dir = os.system('pwd')
MAX_SENT_LENGTH=25
MAX_SENTENCES=2000
input_src = u"../../../../data/europarl-v7.de.2500.dev"
input_dest = u"../../../../data/europarl-v7.en.2500.dev"
input_parallel = u"../../../../data/europarl-v7.en.2500.dev.parallel"

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
    flags.define_string("output", "denews-cost_sensitive-josh-bleu4-um.classifier", "Where we write the learned classifier")
    flags.define_string("cost_type", None, "Cost to use for training: greedy or binary")
    #flags.define_string("fold", "test0", "The fold we run on")
    flags.define_string("feat", None, "The feature lookup")
    flags.define_string("lang", "de", "Language")

    flags.InitFlags()


    reader = POSCSVFileReader(u"../pos.vf.csv")
    # Load next word prediction
    #nw = LangModNextWord(flags.bigram_file, flags.source_full_lm)
    #nw = ConstantPredictor()
    nw = LuceneNextWordPredictor(suggester_filename=u"models/1M-leipzig.suggester",ngram=3)
    sys.stderr.write("INFO:Loaded suggester.\n")
    # Load verb prediction

    vp = VWOAAVerbPredictor(unicode(os.path.abspath("./models/1M-tagged.train-oaa.model")))

    # if flags.verb_db:
    #     vp = SQLVerbPredictor(flags.verb_db)
    # else:
    #     vp = LangModVerbPredictor(flags.kenlm_verb_lm, flags.source_full_lm)

    # Load translation
    #ot = OmniscientTranslator(flags.svoc, flags.tvoc, flags.lexical, flags.target_lm, flags.corpus, lang=flags.lang)
    #ot = JoshuaTranslator(unicode(os.path.abspath(u"data/reuters/joshua/tune")))
    ot = JoshuaPackTranslator(unicode(os.path.abspath(u"/fs/clip-simtrans/research/de-en-langpack/apache-joshua-de-en-2016-11-18/")),
                                  mem=u'16g',
                                  topn=1)
            

            
    #corpus = Corpus(flags.corpus)

    # feature_extractor=JapaneseSentenceFeatureExtractor(ngram=2,
    #                                                    use_position=False,
    #                                                    count_case_markers=False,
    #                                                    normalize_verb=True,
    #                                                    case_marker_ngrams=2)
    
    #sentence_extractor = JapaneseUnparsedLastVerbSentenceExtractor(feature_extractor=feature_extractor)

    #jr = UntaggedJapaneseFileReader(u"data/reuters/reuters-cdec.dev", sent_extractor=sentence_extractor)
    #er = PlainUnicodeFileReader(u"data/reuters/reuters-cdec.dev")
    #reader = CdecParallelCorpusReader(u"data/reuters/reuters-cdec.dev", jr, er)
    #reader = CdecParallelCorpusReader(u"reuters.dev.sample", jr, er)

    
    trn = []
    #for ii in corpus.get_fold(flags.fold):
    _id = -1
    fold = "train"
    num_sents = 0
    for line in reader:
        if reader.get_fold(line) != fold:
            continue
        if num_sents > MAX_SENTENCES:
            break
        line = [x.lower() for x in line]
        _id = reader.get_id(line)
        #preverb = reader.left_context_tokens(line)
        preverb = reader.get_context_tokens(line)
        if preverb is None or len(preverb) > MAX_SENT_LENGTH or len(preverb) == 0:
            continue
        num_sents += 1
        if num_sents % 100 == 0:
            sys.stderr.write("INFO: Sentence " + str(num_sents) + "\n")
        true_verb = [reader.get_verb(line)]
        #print " ".join(preverb), " ".join(true_verb)
        #ot.load_single_alignment(flags.cache_dir, ii.id)
        #trn += [ParallelInstance(preverb, ii.verb, ii.en.split(), ii.id)]
        en_toks = reader.get_target_text(line).split()
        trn += [ParallelInstance(preverb, true_verb, en_toks, _id)]
        #TODOtest, dev = 
    sys.stderr.write("INFO: Starting Searn training")
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
    print "INFO: Trained on " + str(num_sents) + " sentences."
    sys.stderr.write("Done")
    exit()
    ##Only execute below if you want stats on the training data.
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


