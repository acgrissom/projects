
# Todo:
#
# 1.  Create a policy class that can give its loss given a
# configuration and return its action
#
# 2.  Given a loss function, create an optimal policy
#
# 3.  Create a policy interpolation class that acts like a policy but
# has multiple policies inside of it.
#
# 4.  Create a policy that is learned from a cost-sensitive classifier
#
# 5.  Generate features of a state

import pyximport; pyximport.install()
from collections import defaultdict
import logging
from random import random

from prediction import VerbPredictor, NextWordPredictor
from multi_bleu import MultiBleu
from parallel_corpus import ParallelInstance
from translation import Translation, SequenceTranslation, Translator

VALID_ACTIONS = set(["WAIT", "VERB", "NEXTWORD", "COMMIT"])



class WeightedTrainingSet:
    """
    Serves as training set for a classifier
    """

    def __init__(self):
        None


class SentenceStateLattice:
    """
    Lattice for representing the possible translation states.
    """

    def __init__(self, id, full_prefix, list full_verb, target, references,
                 width=3, nw_pred=None, vb_pred=None, trans=None):
        self.full_prefix = full_prefix
        self.full_verb = full_verb
        self.target = target
        self.references = references
        self.nw = nw_pred
        self.vb = vb_pred
        self.id = id
        self.trans = trans

        assert isinstance(width, int), "Must be int: %s" % unicode(width)
        assert isinstance(references, list), \
            "Must be list: %s" % unicode(references)
        self.bleu = MultiBleu(width, references, cache = True)

        self._states = {}
        self._backpointer = {}
        self._total_score = {}
        self._beta = None

        self._verb_guesses = None
        self._word_guesses = None
        self._commit_guesses = None

        self._policy_cost = None

    def next_trans(self, pos, action):
        """
        If we start at a state at index @pos and execute @action, what
        additional translation will we have in the next state?
        """
        assert self._commit_guesses, "Guesses aren't initialized"

        if pos >= self.source_length() or pos < 0:
            assert pos == self.source_length() and action == "COMMIT", \
                "Got action %s in position %i (len=%i)" % \
                (action, pos, self.source_length())

            return self._commit_guesses[pos]

        assert action in VALID_ACTIONS

        val = None
        if action == "COMMIT":
            val = self._commit_guesses[pos]
        elif action == "VERB":
            val = self._verb_guesses[pos]
        elif action == "WAIT":
            val = Translation()
        elif action == "NEXTWORD":
            val = self._word_guesses[pos]

        assert isinstance(val, Translation), "No translation for %s in %i" % \
            (action, pos)
        return val

    # TODO: one potential problem is that this cost might have been
    # computed for the wrong policy.  It would be nice to have a hash
    # of the policy to ensure that doesn't happen.
    def get_cost(self, state, action):
        assert self._policy_cost, "Costs not yet initialized for a policy"

        return self._policy_cost[(state.input_position, state.history)][action]

    def cost_backward(self, policy):
        assert self._forward, "Must compute forward reward first"
        self._policy_cost = defaultdict(dict)
        backward = {}
        for index, history in sorted(self._forward, reverse=True):
            # The last states are simple
            if index == self.source_length():
                backward[(index, history)] = self._forward[(index, history)]
                continue

            state = self._forward_state[(index, history)]

            # For everything else, we need to compare policy
            next_action = policy.action(state)

            # What reward do we get if we follow the policy in this state
            policy_value = backward[(index + 1, history + (next_action,))]
            backward[(index, history)] = policy_value

            # Compute cost vs. all other possible actions
            if index == self.source_length() - 1:
                # The only possible action is commit, so the cost will be zero
                assert next_action == "COMMIT", "Policy must commit after final word"
                self._policy_cost[(index, history)][next_action] = 0.0
            else:
                for aa in VALID_ACTIONS:
                    if aa == next_action:
                       self._policy_cost[(index, history)][aa] = 0.0
                    else:
                        self._policy_cost[(index, history)][aa] = backward[(index + 1, history + (aa,))] - policy_value

    def source(self, cutoff):
        pl = len(self.full_prefix)
        if cutoff < len(self.full_prefix):
            return self.full_prefix[:cutoff]
        else:
            return self.full_prefix + self.full_verb[:(cutoff - pl)]

    def source_length(self):
        return len(self.full_prefix) + len(self.full_verb)

    def add_guess(self, index, list prefix, list verb):
        start = len(self.full_prefix)
        prob, verb_guess = self.vb.predict(prefix)
        if type(verb) is unicode:
            verb = [verb]
        if type(verb_guess) is unicode:
            verb_guess = [verb_guess]
        # optimal verb predictor
        #prob = 1.0
        #verb_guess = self.full_verb

        logging.debug("GUESS %i %s %s", index, prefix, verb)

        if index >= start:
            prob, nw = self.nw.predict(prefix + verb)

            self._commit_guesses[index] = self.trans.top(self.id, prefix,
                                                         verb)
            if verb_guess:
                self._verb_guesses[index] = self.trans.top(self.id, prefix,
                                                            verb_guess)
            else:
                self._verb_guesses[index] = Translation()

            if nw:
                self._word_guesses[index] = self.trans.top(self.id, prefix,
                                                            verb + [nw])
            else:
                self._word_guesses[index] = Translation()
        else:
            prob, nw = self.nw.predict(prefix)

            self._commit_guesses[index] = \
                self.trans.top(self.id, prefix, [])

            if verb_guess:
                self._verb_guesses[index] = self.trans.top(self.id, prefix,
                                                            verb_guess)
            else:
                self._verb_guesses[index] = Translation()

            if nw:
                self._word_guesses[index] = \
                    self.trans.top(self.id, prefix + [nw], [])
            else:
                self._word_guesses[index] = Translation()

        logging.debug("Guess %i: C: %s V: %s W: %s", index,
                      " ".join(self._commit_guesses[index].as_list()),
                      " ".join(self._verb_guesses[index].as_list()),
                      " ".join(self._word_guesses[index].as_list()))

        return self._commit_guesses[index], self._verb_guesses[index], \
            self._word_guesses[index]

    def fill_guesses(self):
        """
        Because we'll need what translations we'll use later on, we should form
        all the translations from the start.  This will allow us to combine
        these translations with states later.
        """

        self._verb_guesses = defaultdict(dict)
        self._word_guesses = defaultdict(dict)
        self._commit_guesses = {}

        total_length = self.source_length()
        prefix_length = len(self.full_prefix)

        # This loop can start at 0 *or* 1.  0 causes it to pass in
        # empty prefixes, but that's fine, because NEXTWORD could (in
        # theory) predict from those too.  It currently starts from 1
        # just to make things a little simpler.
        for ii in xrange(1, total_length):
            prefix = self.full_prefix[:ii]
            verb = self.full_verb[:max(0, ii - prefix_length)]
            # We substract 1 so that the data structure reflects the
            # index of the newly created state, not the number of
            # words observed.
            self.add_guess(ii - 1, prefix, verb)

        # We have a special commit guess at the end of the entire sentence
        if type(self.full_verb) is unicode:
            self.full_verb = [self.full_verb]
        full = self.trans.top(self.id, self.full_prefix, self.full_verb)
        self._commit_guesses[total_length - 1] = full
        self._commit_guesses[total_length] = full

    def cost_forward(self, state, history=(), reward=0):
        """
        Builds all the rewards that you'll store as you evaluate *all* possible
        policy transitions.  Should be first called only with initial
        state and will recursively call all reachable states.

        TODO: There's some redundancy in computing the lattice.  However, the
        lattice is looking for optimal policy, so there are some dumb
        mistakes that you could make that will only be found by running
        this.  This function should probably replace that eventually,
        but for the moment I guess we'll live with duplication.
        """

        # Add the current state's reward to the forward data structure
        #
        # This datastructure is keyed by two elements.  The first is
        # the current index of the state and the second key is the
        # history of actions that have been taken.
        self._forward[(state.input_position, history)] = reward
        self._forward_state[(state.input_position, history)] = state

        # if we've reached the end of the sentence, then we're done
        if state.input_position == self.source_length() - 1:
            # Add the final commit state
            new_reward = state.reward("COMMIT")
            self._forward[(state.input_position + 1, history + ("COMMIT",))] = new_reward
            self._forward_state[(state.input_position + 1, history + ("COMMIT",))] = \
              state.evolve_state("COMMIT", state.input_position + 1)
        else:
            for aa in VALID_ACTIONS:
                # Evolve the state
                new_state = state.evolve_state(aa, state.input_position + 1)
                # Append to the history
                new_history = history + (aa,)

                # Compute the reward
                reward = state.reward(aa)

                # Save new state and recurse
                self._forward_state[(aa, new_history)] = new_state
                self.cost_forward(new_state, new_history, reward)

    def build_entry(self, pos, action):
        """
        Create a new state after seeing pos words and having action as the
        action that got you into this state.
        """

        # By default, the best thing you could have done is wait since the first index.
        best_pos = None
        best_reward = float("-inf")
        idx = (pos, action)

        this_translation = self.next_trans(pos - 1, action)

        for prev in xrange(0, pos):
            possible_actions = VALID_ACTIONS
            if prev == self.source_length():
                possible_actions = ['COMMIT']

            for prev_ac in possible_actions:
                new_trans = SequenceTranslation(pos, self._states[(prev, prev_ac)].translation,
                                                this_translation)

                combined = self.combination_reward(dict(new_trans.translation_histogram()), pos)

                if combined > best_reward:
                    best_reward = combined
                    best_pos = (prev, prev_ac)

        assert best_pos, "We did not find a good starting state for %i %s" % (pos, action)
        self._backpointer[idx] = best_pos
        self._total_score[idx] = best_reward

        # TODO(jbg): Can we supply words and translations here to help be
        # consistent / quick?
        ns = self._states[best_pos].evolve_state(action, pos)
        self._states[idx] = ns
        self._backpointer[idx] = best_pos
        return best_pos, best_reward

    def score_histogram(self, trans_hist):
        d = {}
        for ii in trans_hist:
            point_reward = self.bleu.score(trans_hist[ii])
            d[ii] = point_reward
        return d

    def integrate_histogram(self, score_hist, pos, verbose=False):
        val = 0.0
        # Go over the source length by 1 due to state evolving
        total_length = self.source_length() + 1

        # Don't go over the total length
        last = min(pos + 1, total_length)
        if verbose:
            print ("-----------START INTEGRATION----------")
        for ii in sorted(score_hist, reverse=True):
            # We subtract one because we count all observations once already
            difference = last - ii
            if difference > 0:
                contrib = float(difference) * score_hist[ii]
                val += contrib
                if verbose:
                    print("(%f, %f) -> (%f, %f) = %f" % \
                              (last, score_hist[ii], ii,
                               score_hist[ii], contrib))
            last = ii
        if verbose:
            print ("-----------END INTEGRATION----------")

        # weigh final translation higher
        if (total_length - 1) in score_hist:
            val += 16.8 * score_hist[total_length-1]

        # Normalize by length of sentence
        # We have added one to total_length, now -1
        total = val / float(total_length - 1)
        if verbose:
            print("TOTAL: %f" % total)
        return total

    def combination_reward(self, trans_hist, pos):
        assert isinstance(trans_hist, dict), \
            "Translation history must be dictionary"

        score_hist = self.score_histogram(trans_hist)
        total_reward = self.integrate_histogram(score_hist, pos)

        return total_reward

    def base_case_init(self):
        '''
        set the base cases for the table
        '''
        logging.debug("Base case init")
        if not self._commit_guesses:
            self.fill_guesses()

        initial = State(self, SequenceTranslation(0), 0)
        self._states[(0, "")] = initial

        for aa in VALID_ACTIONS:
            idx = (0, aa)

            # The back pointer corresponds to (state, action) and encodes
            # when I was in this *state* I took this *action*.
            ns = State(self, SequenceTranslation(0), 0)
            self._states[idx] = ns

            # After only seeing the 0th word, we can have no good guess
            self._total_score[idx] = 0.0


    @property
    def initialized(self):
        """
        Returns whether translations have been saved
        """
        return self._commit_guesses != None

    def build_table(self, forward_backward=False):
        '''
        determines, for each state what previous state it should have evolved from
        '''
        # Create a place where we can store the rewards
        self._forward = {}
        self._forward_state = {}

        # Base case
        self.base_case_init()
        # Extend the base cases (which are at position 1)
        total_length = self.source_length()
        for ii in xrange(1, total_length):
            for aa in VALID_ACTIONS:
                self.build_entry(ii, aa)
        # Do the final action
        self.build_entry(total_length, "COMMIT")

        if forward_backward:
            self.cost_forward(State(self, SequenceTranslation(0), 0))

    def get_back_pointer(self, idx):
        return self._backpointer[idx]

    def get_reward(self, idx):
        return self._total_score.get(idx, 0)

    def optimal_actions(self):
        """
        Do a post-order traversal of indices to create a sequence of state
        actions
        """
        res = []

        current = (self.source_length(), "COMMIT")
        assert current in self._backpointer
        # Go until we get to the start state (where we've only seen
        # the first word)
        while current[0] != 0:
            previous = self._backpointer[current]
            res.append((current[0] - 1, current[1]))

            # We don't explicitly represent wait actions, so we must add those
            # back to the sequence, which we do through this iteration
            if previous[0] != current[0] - 1:
                for jj in xrange(current[0] - 1, previous[0], -1):
                    res.append((jj - 1, "WAIT"))

            current = previous

        res.reverse()
        return res


class State:
    """
    Represents the current translation state and generates features.  The
    state's input_position refers to having seen that many tokens in the
    source stream.
    """

    def __init__(self, lattice, translation, input_pos=0):
        self.input_position = input_pos
        self.history = ()

        assert lattice.initialized, \
            "Lattice must be initialized with translations"

        # The values are a dictionary of word positions and translations.
        assert isinstance(translation, Translation)
        self.translation = translation
        self.lattice = lattice

    def sent_id(self):
        """
        Returns the sentence id for the corresponding sentence.  Needed for the
        optimal policy.
        """

        return self.lattice.id

    def source(self):
        """
        Get the source string up to current position
        """

        return self.lattice.source(self.input_position + 1)

    def evolve_state(self, action, new_pos):
        """
        Evolves the state after the specified number of waits (default 0), and
        then executes the specified action (could be another wait).
        """
        assert action in VALID_ACTIONS, "Action %s is invalid" % action

        # No matter what, we recieve additional input (if it's available)
        #assert new_pos > self.input_position, \
        #    "New state must be later than current state: %i (was %i)" % \
        #    (new_pos, self.input_position)

        if action == "COMMIT" or action == "NEXTWORD" or action == "VERB":
            # This is the one-off translation of taking this action in a
            # given state.  Because we're moving to new_pos, this should
            # be based on new_pos -1 (as there are implicit waits after
            # the current's states position).
            trans = self.lattice.next_trans(new_pos - 1, action)

            # Combine the one-off translation into a new sequence translation
            new_trans = SequenceTranslation(new_pos, self.translation, trans)
        else:
            new_trans = self.translation

        s = State(self.lattice, new_trans, new_pos)
        s.history = tuple(self.history + (action,))
        return s

    def composite_translation(self):
        return self.translation

    def composite_list_translation(self):
        return self.translation.as_list()

    def status(self):
        """
        Returns a human readable string describing the current state
        """

        status = """
        ============
        POS: %i
        """

        status = status % (self.input_position)

        status += "        %s" % (unicode(self.translation.as_list()))

        status += "\n        ============"

        return status

    def features(self):
        # Needs to be implemented once we have a classifier
        # implementation chosen

        feat = "pos:%i translength:%i "
        feat += " ".join("EN_%s" % x for x in \
                         self.lattice.full_prefix[:self.input_position])
        feat += " ".join("DE_%s" % x for x in \
                             self.composite_list_translation())

    def translation_histogram(self, current_trans=None, current_pos=-1):
        """
        Generate a dictionary where the keys are positions and the values are
        the translation lists created at those positions.
        """

        val = dict(self.translation.translation_histogram())

        if current_trans:
            assert current_pos > 0
            if not current_pos in val:
                val[current_pos] = current_trans.as_list()

        return val

    def reward(self, action):
        """
        Given an action undertaken in this state, return the reward of that
        action.
        """

        # TODO(jbg): If this is too slow this should be memoized.

        # Check to see if this state is at the end of a sentence; then the
        # reward is going to be zero for anything but commit.

        new_pos = self.input_position + 1

        # new_pos is the next position, thus last token should be after the
        # last input word
        last_token = self.lattice.source_length()
        if new_pos == last_token and action != "COMMIT":
            logging.debug("Useless action at end %i %s", new_pos, action)
            return 0.0
        else:
            logging.debug("Action: |%s| POS: %i / %i", unicode(action), self.input_position,
                          last_token)
            ns = self.evolve_state(action, new_pos)

            # This needs to be tested to make sure that the input format is correct
            return self.lattice.combination_reward(dict(ns.translation_histogram()),
                                                   new_pos)


class RandomStub:

    def __init__(self):
        self._queue = []

    def add(self, val):
        self._queue.append(val)

    def __call__(self):
        return self._queue.pop()


class Policy:
    """
    Base case for representing a policy
    """

    def __init__(self, nw, vb, bleu_width, trans):
        # These checks were good ideas, but were failing inexplicably
        # even when the types were correct.
        # assert isinstance(nw, NextWordPredictor)
        # assert isinstance(vb, VerbPredictor)
        # assert isinstance(trans, Translator)

        self._policies = {}
        self._nw = nw
        self._vp = vb
        self._trans = trans
        self._bleu_width = bleu_width

    def action(self, state):
        None

    def set_name(self, name):
        self._name = name

    def build_lattice(self, example):
        return SentenceStateLattice(example.id, example.src_pfx, example.src_vb,
                                    example.tgt, example.references(),
                                    width=self._bleu_width, nw_pred=self._nw,
                                    vb_pred=self._vp, trans=self._trans)

    def get_name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return "UNNAMED"


class InterpolatePolicy(Policy):

    def __init__(self, old_policy, new_policy, old_weight, rnd=None):
        self._weight = old_weight

        # Don't keep a reference if the weight is degenerate
        if self._weight > 0.0:
            self._old = old_policy
        if self._weight < 1.0:
            self._new = new_policy

        if not rnd:
            self._rnd = random

    def action(self, state):
        coin_flip = self._rnd()

        if coin_flip < self._weight:
            return self._old.action(state)
        else:
            return self._new.action(state)


class OptimalPolicy(Policy):

    # Problem: does not consider multiple references
    def find_optimal(self, example, forward_backward=False):
        # These checks were good ideas, but were failing inexplicably
        # even when the types were correct.
        # assert isinstance(example, ParallelInstance)
        lattice = self.build_lattice(example)
        lattice.build_table(forward_backward=forward_backward)
        self._policies[example.id] = dict(lattice.optimal_actions())
        logging.debug("Adding policy for sent %i with states %s",
                      example.id, unicode(self._policies[example.id].keys()))
        return lattice

    def action(self, state):
        # TODO(jbg): make sure that this matches the prediction of the verb
        # predictor

        current = state.input_position
        id = state.sent_id()

        # if we have not observed any tokens, then we always wait
        if current == -1:
            return "WAIT"

        assert id in self._policies, "Sentence id %i not in policies (%s)" % \
            (id, unicode(self._policies.keys()))
        assert current in self._policies[id], \
            "State id %i not in sentence %i policies" % (id, current)

        val = self._policies[id][current]
        return val


class BatchPolicy(Policy):
    """
    Baseline method that waits until the final token in the sentence and then
    commits.
    """

    def action(self, state):
        if state.input_position == state.lattice.source_length() - 1:
            return "COMMIT"
        return "WAIT"


class CommitPolicy(Policy):
    """
    Always commits
    """
    def action(self, state):
        return "COMMIT"
