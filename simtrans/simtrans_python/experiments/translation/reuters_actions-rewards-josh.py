# coding: utf-8
import pyximport; pyximport.install()
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
import logging, pickle, sys
from csv import DictWriter
from prediction import LangModNextWord, SQLVerbPredictor, ConstantPredictor, LangModVerbPredictor, VWOAAVerbPredictor, LuceneNextWordPredictor
from corpus import Corpus
#from translation import CdecTranslator, OnlineCdecTranslator
from translation import SequenceTranslation, JoshuaTranslator
from policy import OptimalPolicy, State, VALID_ACTIONS, BatchPolicy, CommitPolicy, SentenceStateLattice, Policy, InterpolatePolicy
from parallel_corpus import ParallelInstance
from cost_sensitive import ClassifierPolicy, InstanceFactory
from filereaders import UntaggedJapaneseFileReader, PlainUnicodeFileReader, CdecParallelCorpusReader
from feature_extractor import JapaneseSentenceFeatureExtractor
from sentence_extraction import JapaneseUnparsedLastVerbSentenceExtractor
starting_dir = os.system('pwd')
if __name__ == "__main__":
    from lib import flags
    flags.define_string("policy", u"searn", "The policy that we use")
    flags.define_string("policy_file", "cost_sensitive-josh.classifier", "The file that saves the policy")
    flags.define_string("bigram_file", None, "List of bigrams to consider")
    flags.define_string("bleu_width", 3, "BLEU length")
    flags.define_string("database_verb_lm", None, "Database with verb predictions")
    flags.define_string("kenlm_verb_lm", None, "KenLM file with verb predictions")
    flags.define_string("source_full_lm", None, "Full Source Language model  ")
    flags.define_string("corpus", None, "Corpus for the data")
    flags.define_string("output", "results/retuters_actions_rewards-searn.csv", "Where the  results are written")
    flags.define_string("target_lm", None, "Target language language model")
    flags.define_string("lang", "ja", "Language")

    flags.InitFlags()

    # Load data
    #corpus = Corpus(flags.corpus)
    feature_extractor=JapaneseSentenceFeatureExtractor(ngram=2,
                                                       use_position=False,
                                                       count_case_markers=False,
                                                       normalize_verb=True,
                                                       case_marker_ngrams=2)
    
    sentence_extractor = JapaneseUnparsedLastVerbSentenceExtractor(feature_extractor=feature_extractor)

    jr = UntaggedJapaneseFileReader(u"data/reuters/reuters-cdec.dev", sent_extractor=sentence_extractor)
    er = PlainUnicodeFileReader(u"data/reuters/reuters-cdec.dev")
    reader = CdecParallelCorpusReader(u"data/reuters/reuters-cdec.dev", jr, er)

    ot = None
    # Load translation
    if flags.policy == "monotone":
        ot = JoshuaTranslator(unicode(os.path.abspath(u"data/reuters/joshua/tune")),
                              mem=u'8g',
                              topn=1
        )
    elif flags.policy == "searn":
        #ot = JoshuaTranslator(unicode(os.path.abspath(u"data/reuters/joshua/tune")))
        ot = JoshuaTranslator(unicode(os.path.abspath(u"data/reuters/joshua/tune")),
                              mem=u'8g',
                              topn=1
        )
        
       # For the batch and monotone policy, we don't need next word predictions
    if flags.policy in ["batch", "monotone"]:
        nw = ConstantPredictor()
        vp = ConstantPredictor()
    else:
        # Load next word prediction
        #nw = LangModNextWord(flags.bigram_file, flags.source_full_lm)
        nw = LuceneNextWordPredictor(suggester_filename=u"/Users/alvin/OneDrive/research/reuters-j-3gram.suggester",ngram=3)
        sys.stderr.write("INFO:Loaded suggester.\n")
        # Load verb prediction
        if flags.database_verb_lm:
            vp = SQLVerbPredictor(flags.database_verb_lm)
        else:
            #vp = LangModVerbPredictor(flags.kenlm_verb_lm, flags.source_full_lm)
            nw = ConstantPredictor()
            #vp = VWOAAVerbPredictor(unicode(os.path.abspath("corpora/leipzig/1M-tagged.txt.train-oaa.model")))
            vp = VWOAAVerbPredictor(unicode(os.path.abspath("corpora/kyoto/kyoto-train.ja-oaa.model")))
            ot = JoshuaTranslator(unicode(os.path.abspath(u"data/reuters/joshua/tune")),
                                  mem=u'8g',
                                  topn=1)
            

    logging.basicConfig(filename='%s_action_reward.log' % flags.policy, level=logging.DEBUG)

    # Write CSV explaining what the states are doing
    reward_fieldnames = ['id', 'policy', 'position', 'prefix_length', 'total_length', 'reward', 'action', 'verb', 'pred_verb','trans']
    print("Writing output to %s" % flags.output)
    outfile = open(flags.output, 'w')
    u = DictWriter(outfile, reward_fieldnames)
    u.writerow(dict((unicode(x), unicode(x)) for x in reward_fieldnames))
    
    # Create the policy
    if flags.policy.lower().startswith("optimal"):
        policy = OptimalPolicy(nw, vp, int(flags.bleu_width), ot)
    elif flags.policy == "batch":
        policy = BatchPolicy(nw, vp, int(flags.bleu_width), ot)
    elif flags.policy == "monotone":
        policy = CommitPolicy(nw, vp, int(flags.bleu_width), ot)
    elif flags.policy == "searn":
        fl = pickle.load(open(flags.policy_file + ".fl"))
        cif = InstanceFactory(fl, nw, vp, ot, flags.bleu_width)
        policy = ClassifierPolicy(cif, degenerate_option="exclude")
        policy.read_classifier(flags.policy_file + ".weights")
    else:
        assert flags.policy in ["optimal", "batch", "monotone", "searn"], \
            "Don't know what to do with policy %s (%s)" % \
            (flags.policy, flags.policy.lower().startswith("optimal"))

    sys.stderr.write("Created policy.\n")
    policy.set_name(flags.policy)
    correct = 0
    total = 0
    # for sentence in corpus.get_fold(flags.fold):
    #     ot.load_single_alignment(flags.cache_dir, sentence.id)
    
    i = -1
    for line in reader:
        i += 1
        preverb = reader.left_context_tokens(line)
        if preverb is None or len(preverb) > 25:
            continue
        verb = reader.left_label(line).split()
        row = {}
        row['id'] = unicode(i)
        row['policy'] = policy.get_name()
        row['prefix_length'] = len(preverb)
        row['total_length'] = len(preverb) + 1
        row['verb'] = [verb]
        #print "****************************"
        #print reader.right_text(line).split()
        example = ParallelInstance(preverb, [verb], reader.right_text(line).split(),i)
        #sys.stdout.write((u"".join(preverb) + " " + verb))

        if flags.policy.lower().startswith("optimal"):
            lat = policy.find_optimal(example, forward_backward=False)
            
        elif flags.policy == "searn":
            lat = SentenceStateLattice(example.id,
                                       example.src_pfx,
                                       example.src_vb,
                                       example.tgt,
                                       example.references(),
                                       nw_pred=nw,
                                       vb_pred=vp,
                                       trans=ot)
            lat.build_table()
            #maybe you can print reward around here?
        else:
            lat = policy.build_lattice(example)
            lat.build_table(forward_backward=False)

        state = State(lat, SequenceTranslation(0), 0)

        while state.input_position < lat.source_length():
            action = policy.action(state)
            #print action
            #row['pred_verb'] = vp.predict(sentence.preverb[:state.input_position+1])
            #if state.input_position != (lat.source_length()-1) and len(sentence.verb) == 1 and row['pred_verb'][1][0] == sentence.verb[0]:
            #    correct += 1
            #total += 1
            row[u'action'] = action
            row[u'reward'] = state.reward(action)
            print row[u'reward']
            row[u'position'] = state.input_position

            next_state = state.evolve_state(action, state.input_position + 1)
            row[u'trans'] = u" ".join(state.translation.as_list())

            #row['trans'] = next_state.translation[0]
            u.writerow({unicode(k):unicode(v).encode('utf8') for k,v in row.items()})
            if state.input_position <= lat.source_length():
                state = state.evolve_state(action, state.input_position + 1)

        if example.id % 100 == 0:
            print(example.id)
            outfile.flush()
    #print total, correct
