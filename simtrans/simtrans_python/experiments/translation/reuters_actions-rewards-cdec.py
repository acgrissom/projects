# coding: utf-8
import pyximport; pyximport.install()
import logging, pickle, sys
from csv import DictWriter
from searn_harness.prediction import LangModNextWord, SQLVerbPredictor, ConstantPredictor, LangModVerbPredictor
from searn_harness.corpus import Corpus
from searn_harness.translation import CdecTranslator, OnlineCdecTranslator, SequenceTranslation
from searn_harness.policy import OptimalPolicy, State, VALID_ACTIONS, BatchPolicy, CommitPolicy, SentenceStateLattice, Policy, InterpolatePolicy
from searn_harness.parallel_corpus import ParallelInstance
from searn_harness.cost_sensitive import ClassifierPolicy, InstanceFactory
from searn_harness.filereaders import UntaggedJapaneseFileReader, PlainUnicodeFileReader, CdecParallelCorpusReader
if __name__ == "__main__":
    from lib import flags
    flags.define_string("policy", None, "The policy that we use")
    flags.define_string("policy_file", None, "The file that saves the policy")
    flags.define_string("bigram_file", None, "List of bigrams to consider")
    flags.define_string("bleu_width", 5, "BLEU length")
    flags.define_string("database_verb_lm", None, "Database with verb predictions")
    flags.define_string("kenlm_verb_lm", None, "KenLM file with verb predictions")
    flags.define_string("source_full_lm", None, "Full Source Language model  ")
    flags.define_string("corpus", None, "Corpus for the data")
    flags.define_string("output", "results/actions_rewards.csv", "Where the  results are written")
    flags.define_string("target_lm", None, "Target language language model")
    flags.define_string("lang", "ja", "Language")

    flags.InitFlags()

    # Load data
    #corpus = Corpus(flags.corpus)
    jr = UntaggedJapaneseFileReader(u"data/reuters/reuters-cdec.dev")
    er = PlainUnicodeFileReader(u"data/reuters/reuters-cdec.dev")
    reader = CdecParallelCorpusReader(u"data/reuters/reuters-cdec.dev", jr, er)

 
    # Load translation
    if flags.policy == "monotone":
        ot = OnlineCdecTranslator(config_file=u"data/reuters/reuters-cdec.train.sa.ini")
                            #grammars_dir=u"data/reuters/reuters-cdec.dev.grammars")
    else:
        ot = OnlineCdecTranslator(config_file=u"data/reuters/reuters-cdec.train.sa.ini")
                            #grammars_dir=u"data/reuters/reuters-cdec.dev.grammars")

    # For the batch and monotone policy, we don't need next word predictions
    if flags.policy in ["batch", "monotone"]:
        nw = ConstantPredictor()
        vp = ConstantPredictor()
    else:
        # Load next word prediction
        #nw = LangModNextWord(flags.bigram_file, flags.source_full_lm)
        nw = ConstantPredictor()
        # Load verb prediction
        if flags.database_verb_lm:
            vp = SQLVerbPredictor(flags.database_verb_lm)
        else:
            vp = VWOAAVerbPredictor(u"corpora/leipzig/1M-tagged.txt.train-oaa.model")

    logging.basicConfig(filename='%s_action_reward.log' % flags.policy, level=logging.DEBUG)

    # Write CSV explaining what the states are doing
    reward_fieldnames = ['id', 'policy', 'position', 'prefix_length', 'total_length', 'reward', 'action', 'verb', 'pred_verb','trans']
    print("Writing output to %s" % flags.output)
    outfile = open(flags.output, 'w')
    u = DictWriter(outfile, reward_fieldnames)
    u.writerow(dict((x, x) for x in reward_fieldnames))

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


    policy.set_name(flags.policy)
    correct = 0
    total = 0
    # for sentence in corpus.get_fold(flags.fold):
    #     ot.load_single_alignment(flags.cache_dir, sentence.id)
    
    i = 0
    for line in reader:
        preverb = reader.left_context_tokens(line)
        verb = reader.left_label(line)
        row = {}
        row['id'] = unicode(i)
        row['policy'] = policy.get_name()
        row['prefix_length'] = len(preverb)
        row['total_length'] = len(preverb) + 1
        row['verb'] = verb
        example = ParallelInstance(preverb, verb, reader.right_text(line).split(),i)

        # Don't know how to handle trained policy
        if flags.policy.lower().startswith("optimal"):
            lat = policy.find_optimal(example, forward_backward=False)
        elif flags.policy == "searn":
            lat = SentenceStateLattice(example.id, example.src_pfx, example.src_vb, example.tgt, example.references(), nw_pred=nw, vb_pred=vp, trans=ot)
            lat.build_table()
        else:
            lat = policy.build_lattice(example)
            lat.build_table(forward_backward=False)

        state = State(lat, SequenceTranslation(0), 0)

        while state.input_position < lat.source_length():
            action = policy.action(state)
            #row['pred_verb'] = vp.predict(sentence.preverb[:state.input_position+1])
            #if state.input_position != (lat.source_length()-1) and len(sentence.verb) == 1 and row['pred_verb'][1][0] == sentence.verb[0]:
            #    correct += 1
            #total += 1
            row['action'] = action
            row['reward'] = state.reward(action)
            row['position'] = state.input_position

            next_state = state.evolve_state(action, state.input_position + 1)
            row['trans'] = u" ".join(next_state.translation.as_list())
            u.writerow({unicode(k):unicode(v).encode('utf8') for k,v in row.items()})
            state = next_state

        if example.id % 100 == 0:
            print(example.id)
            outfile.flush()
    #print total, correct
