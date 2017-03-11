import pyximport
pyximport.install()
import cython
import logging
import cPickle as pickle

from parallel_corpus import ParallelInstance
from feature_preprocessing import FeatureLookup, FilterFeatureLookup
from policy import OptimalPolicy, State, VALID_ACTIONS, BatchPolicy, \
    SentenceStateLattice, Policy, InterpolatePolicy
from prediction import NextWordPredictor, VerbPredictor
from translation import Translator, SequenceTranslation

from arow import AROW

DEGENERATE_OPTIONS = ["exclude", "use"]

class ClassifierPolicy(Policy):
    def __init__(self, instance_factory, degenerate_option):
        self._classifier = AROW()
        self._instance_factory = instance_factory
        assert degenerate_option in DEGENERATE_OPTIONS, \
            "Don't know what to do with: %s" % degenerate_option
        self._deg_opt = degenerate_option

    def train(self, instances, rounds=15):
        if self._deg_opt == "exclude":
            self._classifier.train([x for x in instances if not x.degenerate], rounds=rounds)
        else:
            assert self._deg_opt == "use"
            self._classifier.train(instances, rounds=rounds)

        self._classifier.probGeneration()

    # TODO: If it's expensive to evaluate the policy, we should
    # memoize states so we don't repeat computation.
    def action(self, state):
        # If we've seen all the sentence, just commit now
        if state.input_position == state.lattice.source_length()-1:
            return "COMMIT"
        else:
            example = self._instance_factory.generate_test(state)
            pred = self._classifier.predict(example, probabilities=True)
            argmax = max(pred.label2prob, key=lambda x: pred.label2prob[x])
            logging.debug("Predicting %s with prob %f", argmax,
                          pred.label2prob[argmax])
            return argmax

    def dump(self, filename):
        with open("%s.do" % filename, 'wb') as outfile:
            pickle.dump(self._deg_opt, outfile, pickle.HIGHEST_PROTOCOL)
        # He: Save feature_lookup. I cannot load instance_factory...
        with open("%s.fl" % filename, 'wb') as outfile:
            pickle.dump(self._instance_factory._fl, outfile, pickle.HIGHEST_PROTOCOL)
        self._classifier.save("%s.weights" % filename)
        # with open("%s.if" % filename, 'wb') as outfile:
            # pickle.dump(
	    # self._instance_factory, outfile, pickle.HIGHEST_PROTOCOL)

    def read_classifier(self, filename):
        self._classifier.load(filename)

    @staticmethod
    def load(filename):
        with open("%s.if" % filename) as inst_file, open("%s.do" % filename) as do_file:
            cp = ClassifierPolicy(pickle.load(inst_file), pickle.load(do_file))
        cp.read_classifier("%s.weights" % filename)
        return cp


class InstanceFactory:

    def __init__(self, feature_lookup, nw_pred, verb_pred, trans, bleu_width):
        # These checks were good ideas, but were failing inexplicably
        # even when the types were correct.
        # assert isinstance(feature_lookup, FeatureLookup)
        # assert isinstance(nw_pred, NextWordPredictor)
        # assert isinstance(verb_pred, VerbPredictor)
        # assert isinstance(trans, Translator)

        self._nw = nw_pred
        self._vb = verb_pred
        self._trans = trans
        self._bleu_width = bleu_width
        self._fl = feature_lookup

    def lattice(self, example):
        lat = SentenceStateLattice(example.id, example.src_pfx,
                                   example.src_vb, example.tgt,
                                   example.references(), self._bleu_width,
                                   self._nw, self._vb, self._trans)
        lat.build_table()
        return lat

    @staticmethod
    # TODO: best == optimal action?
    # get rollout reward from lattice
    def build_costs(state, opt):
        costs = {}

        # weight earlier example higher
        total_length = state.lattice.source_length()
        # percentage of word left: (0, 1]
        words_left = (total_length - (state.input_position+1)) / total_length
        weight = 1 + words_left

        # weight different actions
        action_cost = {'VERB':10, 'NEXTWORD':2, 'COMMIT':1, 'WAIT':0.5}

        if not opt is None:
            opt_action = opt.action(state)
            for aa in VALID_ACTIONS:
                if aa == opt_action:
                    costs[aa] = 0
                else:
                    costs[aa] = action_cost[opt_action] * weight
                    #costs[aa] = 1.0
        else:
            for aa in VALID_ACTIONS:
                costs[aa] = state.reward(aa)

            best = max(costs.values())
            for aa in VALID_ACTIONS:
                costs[aa] = -1.0 * (costs[aa] - best)
                logging.info("cost: %f", costs[aa])


        return costs

    def run_policy(self, lattice, policy, opt = None):
        cdef int ii
        """
        Generate training examples based on the current policy
        """

        #s = State(lattice, SequenceTranslation(0), -1)
        # He: Start from state 0, always WAIT at pos -1
        # otherwise assert fails in next_trans when getting
        # non-WAIT action in state -1
        s = State(lattice, SequenceTranslation(0), 0)
        full_length = lattice.source_length()
        for ii in xrange(1, full_length):
            action = policy.action(s)
            costs = InstanceFactory.build_costs(s, opt)
            logging.debug("taking action %s at position %d", action, ii-1)
            yield self.generate_train(s, costs)
            s = s.evolve_state(action, ii)

    def generate_train(self, state, costs):
        return InstanceFactory.fl_generate_train(state, costs, self._fl, self._nw, self._vb)

    @staticmethod
    def fl_generate_train(state, costs, fl, nw=None, vb=None):
        """
        Static method to allow the generation of features given a specified
        feature lookup (useful for testing without depending on working
        FeatureLookup class).
        """
        # TODO(jbg): add flags module to control the use of features

        name = "%s-%i" % (state.lattice.id, state.input_position)
        target = state.composite_list_translation()

        source = state.source()

        features = {}
        for ii in target:
            features["TG_%s" % ii] = 1.0
        for ii in source:
            features["SRC_%s" % ii] = 1.0

        features["SRC_POS_%i" % state.input_position] = 1.0
        features["TG_POS_%i" % len(target)] = 1.0

        # use relative position
        total_length = state.lattice.source_length() * 1.0
        features["SRC_POS"] = (len(source) + 1) / total_length
        features["TG_POS"] = (len(target) + 1) / total_length

        # previous action
        # if len(state.history) != 0:
        #     features["PREV_ACT_%s" % state.history[-1]] = 1.0

        # previous word and bigram
        curr = source[-1]
        features["CURR_WORD_%s" % curr] = 1.0
        if len(source) > 1:
            prev = source[-2]
            features["PREV_WORD_%s" % prev] = 1.0
            features["CURR_BIGRAM_%s_%s" % (curr, prev)] = 1.0

        # next word prediction
        prob, word = nw.predict(source)
        features["NEXT_WORD_%s" % word] = 1.0
        features["NEXT_WORD_BIGRAM_%s_%s" % (word, curr)] = 1.0
        features["NEXT_WORD_PROB"] = prob

        # verb prediction
        #verb_list = vb.predict_set(state.lattice.full_prefix[:state.input_position+1], 2)
        #prob = [x[0] for x in verb_list]
        #verb = [x[1] for x in verb_list]
        #features["VERB_PRED_%s_%s", (verb[0], verb[1])] = 1.0
        #features["VERB_DELTA"] = prob[0] - prob[1]
        #features["VERB_PRED_%s", verb[0]] = 1.0
        #features["VERB_BIGRAM_%s_%s", (verb[0], curr)] = 1.0

        prob, verb = vb.predict(state.lattice.full_prefix[:state.input_position+1])
        verb = ' '.join([x for x in verb])
        features["VERB_PRED_%s" % verb] = 1.0
        features["VERB_BIGRAM_%s_%s" % (verb, curr)] = 1.0

        return Instance(name, costs, fl(features))

    def generate_test(self, state):
        return self.generate_train(state, costs=None)

class Searn:

    def __init__(self, train, dev, test, nw_pred, verb_pred, trans,
        bleu_width, degenerate, cost_type='binary', beta=0.8, lat_dir=None, opt=None):
        self._trn = train
        self._test = test
        self._dev = dev

        self._nw = nw_pred
        self._verb = verb_pred
        self._trans = trans

        self._bleu_width = bleu_width

        self._optimal = None
        self._beta = beta

        self._feat_look = None
        self._degenerate = degenerate

        self._cost_type = cost_type

        self._lat_dir = lat_dir
        self._opt = opt

    def run_actions(self, policy, sentence):
        inst_fac = InstanceFactory(self._feat_look, self._nw,
                self._verb, self._trans, self._bleu_width)
        lattice = inst_fac.lattice(sentence)
        #state = State(lattice, SequenceTranslation(0), -1)
        state = State(lattice, SequenceTranslation(0), 0)
        for ii in xrange(1, len(sentence)+1):
            action = policy.action(state)
            state = state.evolve_state(action, ii)
            yield action

    def build_features(self, num_features=1000, min_count=1):
        fl = FilterFeatureLookup()
        # Create an instance factory that uses all features (i.e., not using the
        # filter feature lookup)
        cif = InstanceFactory(FeatureLookup(), self._nw, self._verb,
                              self._trans, self._bleu_width)
        op = OptimalPolicy(self._nw, self._verb, self._bleu_width, self._trans)
        for ii in self._trn:
            # Get latice and optimal policy for this example
            lat = op.find_optimal(ii)
            for jj in cif.run_policy(lat, op):
                logging.debug("feature vector %s", str(jj.featureVector))
                fl.add_observation(jj.featureVector)

        self._feat_look = fl.create_lookup(num_features, min_count)

    def get_optimal(self):
        assert self._optimal, "A policy must be trained first"
        return self._optimal

    def train_policy(self, iterations):
        cdef int ii
        assert self._feat_look, "Features must be initialized"

        inst_fac = InstanceFactory(self._feat_look, self._nw, self._verb,
                                   self._trans, self._bleu_width)
        if self._opt:
            with open(self._opt, 'rb') as infile:
                policy = pickle.load(infile)
        else:
            policy = OptimalPolicy(self._nw, self._verb,
                               self._bleu_width, self._trans)
        self._optimal = policy

        if self._cost_type == 'greedy':
            opt = None
        else:
            opt = self._optimal

        # Save the lattice
        #if not self._lat_dir:
        #   for example in self._trn:
        #       lat = policy.find_optimal(example)
        #       with open("scratch/lattice/%d.pkl" % (example.id), 'wb') as outfile:
        #          pickle.dump(lat, outfile, pickle.HIGHEST_PROTOCOL-1)

        ## Save the optimal policy
        #if not self._opt:
        #   with open("scratch/opt.pkl", 'wb') as outfile:
        #       pickle.dump(policy, outfile, pickle.HIGHEST_PROTOCOL-1)

        for ii in xrange(iterations):
            logging.debug("iteration %d", ii)
            instances = []
            for example in self._trn:
                # On the first iteration, we need to figure out what optimal
                # policy is
                if ii == 0:
                    lat = policy.find_optimal(example)
                    # load lattice
                    #with open("scratch/lattice/%d.pkl" % example.id, 'rb') as infile:
                    #    lat = pickle.load(infile)
                else:
                    lat = inst_fac.lattice(example)

                for ss in inst_fac.run_policy(lat, policy, opt):
                    instances.append(ss)

            new_policy = ClassifierPolicy(inst_fac, degenerate_option=self._degenerate)
            new_policy.train(instances)
            policy = InterpolatePolicy(policy, new_policy, self._beta)
        return policy

    def evaluate_policy(self, sentences, policy):
        inst_fac = InstanceFactory(self._feat_look, self._nw,
                                   self._verb, self._trans, self._bleu_width)
        for ss in sentences:
            lattice = inst_fac.lattice(ss)
            lattice.build_table()
            state = State(lattice, SequenceTranslation(0), 0)
            for ii in xrange(1, len(ss)+1):
                action = policy.action(state)
                if ii == len(ss):
                    yield state.reward("COMMIT")
                state = state.evolve_state(action, ii)

    #There was a bug here.  inst_fac undefined.
    def dump_features_for_policy(self, sentences, policy, filename, inst_fac):
        from csv import DictWriter

        d = {}
        d["features"] = ""
        with open(filename, 'w') as outfile:
            csv_out = DictWriter(outfile, ["sent", "pos", "features", "action"])
            for ss in sentences:
                lattice = inst_fac.lattice(ss)
                d["sent"] = lattice.id
                lattice.build_table()
                state = State(lattice, SequenceTranslation(0), -1)
                for ii in xrange(len(ss)):
                    action = policy.action(state)
                    state = state.evolve_state(action, ii)
                    d["pos"] = state.input_position
                    d["source"] = state.source()
                    d["trans"] = "~".join(state.translation.as_list())
                    d["action"] = action
                    if isinstance(policy, ClassifierPolicy):
                        d["features"] = policy._instance_factory.generate_test(state).featureVector
                yield state.reward("COMMIT")


class Instance:

    def __init__(self, name, costs, features):
        self.costs = costs
        self.featureVector = features

        self.degenerate = False

        if costs != None:
            best = min(costs.values())
            worst = max(costs.values())

            if best == worst:
                degenerate = True

            assert best >= 0, "Costs cannot be negative"

            # Does AROW differentiate between best and correct?  I think they
            # are identical by definition for this use case.
            self.bestLabels = [x for x in costs if costs[x] == best]
            self.correctLabels = self.bestLabels

            self.worstLabels = [x for x in costs if costs[x] == worst]

            if best > 0:
                self.costs = dict((x, costs[x] - best) for x in costs)
            else:
                self.costs = costs

if __name__ == "__main__":
    from lib import flags
    from test_optimal_policy import cvnwc_inputs

    # Tiny test corpus
    flags.define_int("bleu_width", 3, "Number of n-grams to consider")
    flags.define_string("degenerate_example", "exclude",
                        "What to do with degenerate training examples")

    flags.InitFlags()

    logging.basicConfig(filename='cost_sens.log', level=logging.DEBUG)
    src, tgt, t, vb, nw = cvnwc_inputs()

    train = [ParallelInstance(src[:-1], src[-1:], tgt, 1)]

    s = Searn(train, train, train, nw, vb, t, flags.bleu_width,
              flags.degenerate_example)
    s.build_features()

    learned_policy = s.train_policy(15)
    opt = s.get_optimal()

    batch = BatchPolicy(nw, vb, int(flags.bleu_width), t)

    for pp in [batch, learned_policy, opt]:
        print(pp.get_name(), sum(s.evaluate_policy(train, pp)))
        print list(s.run_actions(pp, train[0]))
