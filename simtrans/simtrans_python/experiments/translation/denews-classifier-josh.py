# coding: utf-8
import pyximport; pyximport.install()
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
import logging, pickle, sys

import logging
import pickle
from csv import DictWriter
#from test_optimal_policy import cvnwc_inputs
from cost_sensitive import Searn, InstanceFactory, ClassifierPolicy
from prediction import LangModNextWord, SQLVerbPredictor, LuceneNextWordPredictor, VWOAAVerbPredictor
#from corpus import Corpus
#from filreaders import POSCSVFileReader
from filereaders import POSCSVFileReader
#from translation import OmniscientTranslator, Alignment, SequenceTranslation
from translation import JoshuaPackTranslator, SequenceTranslation
from policy import OptimalPolicy, State, VALID_ACTIONS, BatchPolicy, SentenceStateLattice, Policy, InterpolatePolicy
from parallel_corpus import ParallelInstance
from feature_preprocessing import FeatureLookup, FilterFeatureLookup
#MAX_SENT_LENGTH=50
#input_src = u"../../../../data/europarl-v7.de.2500.dev"
#input_dest = u"../../../../data/europarl-v7.en.2500.dev"
#input_parallel = u"../../../../data/europarl-v7.en.2500.dev.parallel"


if __name__ == "__main__":
    from lib import flags

    flags.define_string("bleu_width", 3, "N-gram length to use in bleu scoring")
    flags.define_string("corpus", "../pos.csv", "CSV input file")
    flags.define_string("bigram_file", None, "List of bigrams to consider")
    flags.define_string("source_full_lm", None, "Language model for next word prediction")
    flags.define_string("source_verb_lm", None, "Database with verb predictions")
    flags.define_string("svoc", None, "Source language vocabulary file")
    flags.define_string("tvoc", None, "Target language vocabulary file")
    flags.define_string("lexical", None, "Lexical translation prob table")
    flags.define_string("target_lm", None, "Target language language model")
    flags.define_string("cache_dir", None, "Where we store alignment pickles")
    flags.define_string("output", None, "Where we write the learned classifier")
    flags.define_int("limit", -1, "Number of training sentences")
    flags.define_int("rounds", 5, "Number of training iterations")

    flags.InitFlags()
    logging.basicConfig(filename='classifier_test.log', level=logging.DEBUG)

    # Load next word prediction
    #nw = LangModNextWord(flags.bigram_file, flags.source_full_lm)
    nw = LuceneNextWordPredictor(suggester_filename=u"models/1M-leipzig.suggester",ngram=3)

    # Load verb prediction
    #vp = SQLVerbPredictor(flags.source_verb_lm)
    vp = VWOAAVerbPredictor(unicode(os.path.abspath("./models/1M-tagged.train-oaa.model")))

    # Load translation
    #ot = OmniscientTranslator(flags.svoc, flags.tvoc, flags.lexical, flags.target_lm, flags.corpus)
    ot = JoshuaPackTranslator(unicode(os.path.abspath(u"/fs/clip-simtrans/research/de-en-langpack/apache-joshua-de-en-2016-11-18/")),
                                  mem=u'8g',
                                  topn=1)

    #corpus = Corpus(flags.corpus)
    reader = POSCSVFileReader(u"../pos.csv")
    num_examples = 0
    fold = 'test0'
    #for ii in corpus.get_fold('test0'):
v    for line in reader:
        if reader.get_fold(line) != fold:
            continue
        line = [x.lower() for x in line]
        num_examples += 1
        if flags.limit > 0 and num_examples > flags.limit:
            break
        else:
            if num_examples % 10 == 0:
                print "Adding example %i" % num_examples

        #ot.load_single_alignment(flags.cache_dir, ii.id)
        #ref = []
        #for jj in ii.en.split():
        #    ref.append(jj)
        ref = reader.get_target_text(line).split()
        _id = reader.get_id(line)
        preverb = reader.get_context_tokens(line)
        verb = reader.get_verb(line)
        trn = [ParallelInstance(preverb, [verb], ref, _id)]

        for example in trn:
            l = SentenceStateLattice(example.id, example.src_pfx, example.src_vb, example.tgt, [ref], nw_pred=nw, vb_pred=vp, trans=ot)
            l.build_table()
            logging.debug("Done building table")
            #train = [ParallelInstance(src[:-1], src[-1:], tgt, 1)]
            action_sequence = ['COMMIT', 'VERB', 'NEXTWORD', 'WAIT']
            # Create some states to train on
            instances = []
            fl = FeatureLookup()
            s = {}
            s[-1] = State(l, SequenceTranslation(0), 0)
            for kk, aa in enumerate(action_sequence):
                s[kk] = s[kk - 1].evolve_state(aa, s[kk - 1].input_position + 1)
                costs = InstanceFactory.build_costs(s[kk])
                instances.append(InstanceFactory.fl_generate_train(s[kk], costs, fl))

    cif = InstanceFactory(fl, nw, vp, ot, flags.bleu_width)
    cp = ClassifierPolicy(cif, degenerate_option="exclude")
    cp.train(instances, flags.rounds)
    cp.dump(flags.output)
