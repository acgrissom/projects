import sqlite3
import kenlm, sys, getopt
import hashlib
import codecs
from collections import defaultdict
from feature_extractor cimport *
#from wabbit_wappa import *
from bunny_lure import BinaryLogisticScorer, ClassifierScorer
from operator import itemgetter
import text_tools


from glob import glob
cdef class Predictor:
    """
    Interface for going from words to prediction.
    """

    cpdef tuple predict(self, list words):
        raise NotImplementedError
    

    cpdef str name(self):
        return "Predictor"

cdef class StubPredictor(Predictor):
    cpdef public list _stack
    """
    A utility class for writing unit tests, it holds a list of predictions and,
    when asked, returns the predictions in a LIFO fashion.
    """

    def __init__(self):
        self._stack = []

    cpdef tuple predict(self, list words):
        return 0.0, self._stack.pop()

    cpdef add(self, val):
        self._stack.append(val)

    cpdef clear(self):
        self._stack = []

    cpdef str name(self):
        return "Stub"


class MemoPredictor(Predictor):
    lookup = {}
    """
    Slightly more complicated than the StubPredictor, the MemoPredictor holds
    input/output pairs and memorizes the predictions.  Also used for testing,
    and is necessary for cases when we cannot be sure the order in which the
    predictor will be asked for predictions.
    """

    def __init__(self):
        self._lookup = {}

    def  predict(self, words):
        key = "~".join(words)
        assert key in self._lookup, "%s missing" % key
        return 0.0, self._lookup[key]

    def add(self,  words,  val):
        key = "~".join(words)
        self._lookup[key] = val

    def  name(self):
        return "Memo"


cdef class NextWordPredictor(Predictor):
    """
    Class that predicts the next word
    """

    cpdef str name(self):
        return "NextWord"


class MemoNextWordPredictor(MemoPredictor, NextWordPredictor):

    def name(self):
        return "MemoNextWord"
 

class StubNextWord(StubPredictor, NextWordPredictor):

    def name(self):
        return "StubNextWord"

import subprocess, sys, errno, codecs
cdef class LuceneNextWordPredictor(NextWordPredictor):
    cpdef unicode suggester_filename
    cpdef unicode predictor_path
    cpdef unicode lib_path
    cpdef int ngram
    cpdef lucene_proc
    def __init__(self, unicode suggester_filename,
                 int ngram=2,
                 unicode predictor_path=u"LuceneNextWordPredictor/dist",
                 unicode lib_path=u"LuceneNextWordPreditor/lib"):
                 
        self.ngram = ngram
        self.suggester_filename = suggester_filename
        cdef unicode command = unicode(os.environ["JAVA_HOME"]) + u"/bin/java "
        command += u" -cp " + lib_path + u" "
        command += u" -jar "
        command += predictor_path + u"/LuceneNextWordPredictor.jar "
        command += u" search "
        command += suggester_filename + u" "
        command += unicode(ngram) + u" "
        sys.stderr.write("Running: " + command + "\n")

        self.lucene_proc = subprocess.Popen(command,
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            #stderr=subprocess.PIPE,
                                            close_fds=True,
                                            universal_newlines=True,
                                            shell=True)
        self.lucene_proc.stdout = codecs.getwriter('utf8')(self.lucene_proc.stdout)
        self.lucene_proc.stdin = codecs.getwriter('utf8')(self.lucene_proc.stdin)
        

    cpdef tuple predict(self, list words):
        print "words:",words
        cdef unicode prefix = u" ".join(words) + u" "
        cdef unicode word
        cdef double prob
        self.lucene_proc.stdin.write((prefix + u"\n"))
        self.lucene_proc.stdin.flush()
        self.lucene_proc.stdout.flush()
        word = unicode(self.lucene_proc.stdout.readline().strip().decode('utf8'))
        #print "*********************"
        line2 = self.lucene_proc.stdout.readline().strip()
        #print line2
        prob = float(line2)
        return prob, word

    cpdef str name(self):
        return "LuceneNextWord";

"""Uses language model to make predictions directly. Untested."""
cdef class KenLMNextWordPredictor(NextWordPredictor):
    cpdef _lm
    def __init__(self, unicode lm_file):
        self. _lm = kenlm.LanguageModel(lm_file)

    cpdef tuple predict(self, list words):
        cdef list scores = []
        cdef unicode sent = u" ".join(words + [ii])
        scores.append((self._lm.score(sent), ii))

        if scores:
            return max(scores)
        else:
            return 0, u''

        

"""Uses a file list of frequent bigrams"""
cdef class LangModNextWord(NextWordPredictor):
    cpdef _bigrams
    cpdef _lm
    def __init__(self, bigram_file, lm_file):
        self._bigrams = defaultdict(set)
        try:
            for ii in open(bigram_file):
                first, second = ii.split()

                self._bigrams[first].add(second)
        except IOError:
            print("empty file %s" % bigram_file)

        try:
            self._lm = kenlm.LanguageModel(lm_file)
        except IOError:
            print("empty file %s" % lm_file)


    cpdef tuple predict(self, list words):
        cdef list scores = []
        cdef unicode sent
        for ii in self._bigrams[words[-1]]:
            sent = u" ".join(words + [ii])
            scores.append((self._lm.score(sent), ii))

        if scores:
            return max(scores)
        else:
            return 0, u''

    cpdef float lm_score(self, list words):
        return self._lm.score(u" ".join(words))

    cpdef float lm_score_string(self, unicode sents):
        return self._lm.score(sent)


cdef class VerbPredictor(Predictor):

    cpdef str name(self):
        return "Verb"

    cpdef tuple predict(self, list words):
        pass

cdef class ConstantPredictor(VerbPredictor):
    """
    Always predicts the empty string
    """

    cpdef str name(self):
        return "Constant"

    cpdef tuple predict(self, list words):
        return 0.0, ''


class MemoVerbPredictor(MemoPredictor, VerbPredictor):

    def name(self):
        return "Memo Verb"


cdef class LangModVerbPredictor(VerbPredictor):
    cpdef dict _verbs
    cpdef _background
    def __init__(self, directory, full_lm):
        self._verbs = {}
        for ii in glob("%s/vb-*.binary" % directory):
            verb = ii.split("vb-")[-1]
            try:
                self._verbs[verb.replace("_", " ").replace(".binary", "")] = \
                  kenlm.LanguageModel(ii)
            except RuntimeError:
                print("Error loading %s" % ii)
            except IOError:
                print("empty file %s" % ii)

        try:
            self._background = kenlm.LanguageModel(full_lm)
        except IOError:
            print("empty file %s" % full_lm)

    cpdef tuple predict(self, list words):
        scores = [(self._verbs[ii].score(" ".join(words)), ii) \
                     for ii in self._verbs]
        if scores:
            score, verb = max(scores)
            verb = verb.split(" ")
            return score, verb
        else:
            return None

cdef class ContextScorer(VerbPredictor):
    cpdef float score_context(self, unicode context, unicode verb):
        raise NotImplementedError
    cpdef dict get_context_score_for_all_verbs(self, unicode context):
        raise NotImplementedError
    cpdef list get_verbs(self):
        raise NotImplementedError

"""
(acg) I didn't want to change the old LangModVerbPredictor; so, I forked it.
"""
cdef class KenLMLanguageModelVerbScorer(ContextScorer):
    cpdef dict _verbs
    cpdef _background
    cpdef dict verb_priors
    def __init__(self, unicode directory):
        cdef unicode verb
        cdef float count
        cdef  verb_path
        cdef list verb_count_list
        cdef float total_verb_count = 0.0
        self._verbs = {}
        self.verb_priors = {}
        for verb_count in codecs.open(directory + u"/labels.txt", encoding="utf8"):
            verb_count_list = verb_count.split(u"\t")
            verb = verb_count_list[0]
            count = float(verb_count_list[1])
            self.verb_priors[verb] = count
            total_verb_count += count
            verb_path = (directory + u"/" + verb + u".arpa").encode("utf8")
            try:
                self._verbs[verb] = kenlm.LanguageModel(verb_path)
                sys.stderr.write("Loading " + verb_path + "\n")
            except RuntimeError:
                print "Error loading", verb_path
            except IOError:
                print("empty file %s" % verb_path)

        try:
            self._background = kenlm.LanguageModel(directory + "/_background.arpa")
        except IOError:
            print("empty file %s" % self._background)


        for verb in self.verb_priors.keys():
            self.verb_priors[verb] = self.verb_priors[verb] / total_verb_count
            

    cpdef tuple predict(self, list words):
        scores = [(self._verbs[ii].score(" ".join(words)) * self.verb_priors[ii], ii) \
                     for ii in self._verbs]
        if scores:
            score, verb = max(scores)
            #verb = verb.split(" ")
            return score, verb
        else:
            return None
        
    cpdef float score_context(self, unicode context, unicode verb):
        if verb not in self._verbs:
            sys.err.write("KenLMScorer:" + " Verb " + verb + " not valid.")
            return None
        return self._verbs[verb].score(context) / self._background.score(context)

    """
    Returns a verb->score mapping of for the context, for all verbs.
    """
    cpdef dict get_context_score_for_all_verbs(self, unicode context):
        cdef dict verb_context_scores = dict()
        for verb in self._verbs:
            verb_context_scores[verb] = self.score_context(context, verb)
        return verb_context_scores

    cpdef list get_verbs(self):
        return self._verbs.keys()
        # cdef list verbs = list()
        # for v in self._verbs:
        #     verbs.append(v)
        # return verbs

    
cdef class SQLVerbPredictor(VerbPredictor):
    cpdef c
    cpdef load_db(self,db_name):
        connection = sqlite3.connect(db_name)
        self.c = connection.cursor()
    def __init__(self,db_filename):
        self.load_db(db_filename)

    cpdef tuple predict(self, list words):
        sentence = "%s " % " ".join(words)
        sentence = sentence.strip()
        query = "SELECT probability, verb FROM verbs WHERE sentence = ? ORDER BY probability DESC LIMIT 1"
        query = unicode(query)
        #query = "SELECT probability, verb FROM verbs WHERE sentence = '" + sentence + "' ORDER BY PROBABILITY LIMIT 1"
        cursor = self.c.execute(query, (sentence,))
        #cursor = self.c.execute(query)
        row = cursor.fetchone()
        if not row:
            print sentence
            return float("-inf"), ''
        else:
            prob, verb = row
            #verb = tuple(verb.split())
            verb = [verb]
            verb = tuple(verb)
            return prob, verb

    def predict_set(self, list words, limit):
        sentence = "%s " % " ".join(words)
        sentence = sentence.strip()
        query = "SELECT DISTINCT probability, verb FROM verbs WHERE sentence = ? ORDER BY probability DESC LIMIT " + str(limit)
        query = unicode(query)
        #query = "SELECT probability, verb FROM verbs WHERE sentence = '" + sentence + "' ORDER BY PROBABILITY LIMIT 1"
        cursor = self.c.execute(query, (sentence,))
        #cursor = self.c.execute(query)
        rows = cursor.fetchall()
        if not rows:
            print sentence
            return float("-inf"), ''
        else:
            prob, verbs = rows
            #verb = tuple(verb.split())
            return prob, verbs

    cpdef float __get_best_answer_and_score(row):
        scores = row.split('\t')
        return scores[0]

    cpdef float __get_best_score(row):
        scores = row.split('\t')
        return scores[0].split()[1]

    cpdef unicode __get_best_answer(row):
        scores = row.split('\t')
        return scores[0].split()[0]


cdef class SQLNextWordPredictor(NextWordPredictor):

    cpdef load_db(self,db_name):
        connection = sqlite3.connect(db_name)
        self.c = connection.cursor()

    def __init__(self,db_filename):
        self.load_db(db_filename)


    cpdef tuple predict(self, list words):
        sentence = " ".join(words)

        query = "SELECT probability, next_words FROM verbs WHERE sentence = ? ORDER BY probability DESC LIMIT 1"
        cursor = self.c.execute(query, (sentence,))
        row = cursor.fetchone()
        if not row:
            return float("-inf"), ''
        else:
            prob, verb = row
            verb = tuple(verb.split())
            return prob, verb


def sql_verb_test(argv):
    if len(argv) < 2:
        print "Arguments: [database file] [sentence]"

    db_file = argv[1]
    test_query = argv[2].split(" ")
    print test_query
    verb_predictor = SQLVerbPredictor(db_file)
    #verb_predictor.load_db(db_file)
    print verb_predictor.predict(test_query)
    print verb_predictor.predict_set(test_query,5)

if __name__ == "__main__":
    sql_verb_test(sys.argv)
    from lib import flags

    flags.define_string("full_lm", None,
                        "The full language model")

    flags.define_string("verb_lm", None,
                        "The language models for verbs")

    flags.define_string("bigram_file", None,
                        "Where to read the list of candidate bigrams")

    flags.define_string("verb_pred_database", None,
                        "Where to read database for verb prediction")

    flags.InitFlags()

    vp = LangModVerbPredictor(flags.verb_lm, flags.full_lm)
    nwp = LangModNextWord(flags.bigram_file, flags.full_lm)

    if flags.verb_pred_database:
        sql_verb_test(flags.verb_pred_database)

    if flags.verb_lm:
        sent = "Ausserdem koennte ein Anti-dumping Paragraph im " + \
            "Congress aufgenommen"
        print("Verb for %s" % sent)
        print(vp.predict(sent.split()))

    if flags.full_lm:
        sent = "die arbeitsaemter haetten noch finanzreserven fuer arbeitsplatzprogramme".split()
        for ii in xrange(len(sent)):
            print("Next word after %s" % sent[:(ii + 1)])
            print(nwp.predict(sent[:(ii + 1)]))

import os
cdef class VWOAAHierarchicalPredictor(VWOAAVerbPredictor):
    cpdef dict vw_predictors
    cpdef public int max_layer
    def __init__(self,
                 unicode model_dir,
                 feature_extractor=GermanTaggedFeatureExtractor(),
                 int max_layer=0):
        self.vw_predictors = dict()
        self.max_layer = max_layer
        file_names = [fn for fn in os.listdir(model_dir) if fn.endswith(u'model')]
        #unicode prefix = u""
        if max_layer == 0:
            for f in file_names:
                prefix = f.split(u".")[0]
                num = int(prefix)
                if num > self.max_layer:
                    if max_layer == 0:
                        self.max_layer = num

        #print "*********max_layer",max_layer
        for layer in xrange(1,max_layer + 1):
            #print "-------------------------------------in for"
            model_filename = model_dir + u"/" + unicode(layer) + u".model"
            self.vw_predictors[layer] = VWOAAVerbPredictor(model_filename,
                                                          feature_extractor=feature_extractor,
                                                          labels_file=model_dir + u"/oaa-hier.labels")
            
                                                     
    cpdef close(self):
       for key in self.vw_predictors:
           self.vw_predictors[key].close()
          
    cpdef tuple predict(self, list words):
        cdef dict ns_features = dict()
        cdef unicode pred
        cdef unicode last_pred
        cdef list prev_words
        cdef subwords = list()
        cdef list hierarchical_predictions = list()
        #feeding forward predictions
        cdef int max_layer = self.max_layer
        if len(words) < max_layer:
            max_layer = len(words)
        #print "vw_predictors",self.vw_predictors
        score, pred = self.vw_predictors[len(words)].predict(words)
        return (score,pred)

    cpdef list get_hierarchical_features(self, list words):
        cdef subwords = list()
        cdef list hierarchical_predictions = list()
        cdef int max_layer = self.max_layer
        if len(words) < max_layer:
            max_layer = len(words)
        for layer_num in xrange(1,max_layer):                
            subwords = words[layer_num:]            
            ns_features = self.feature_extractor.get_features(subwords)
            if not layer_num == 0:
                ns_features["hierarchical"] = hierarchical_predictions
            pred = self.vw_predictors[layer_num].predict(ns_features)
            if not layer_num < max_layer:
                hierarchical_predictions.append(unicode(layer_num + u":" + pred))
            last_pred = pred
        return hierarchical_predictions

        
    
 

cdef class VWOAAVerbPredictor(VerbPredictor):
    cpdef public unicode model_path
    cpdef public dict verb_indices
    cpdef public classifier
    cpdef public  feature_extractor
    cpdef public vw_outfile
    def __init__(self,
                 unicode model_path,
                 feature_extractor=GermanTaggedFeatureExtractor(),
                 unicode labels_file=None 
                 ):
        self.model_path = model_path
        self.feature_extractor = feature_extractor
        self.verb_indices = {}
        if labels_file == None:
            self.load_verb_indices(model_path.replace(".model","") + ".labels")
        else:
            self.load_verb_indices(labels_file)
        self.classifier = ClassifierScorer(model_path)
        self.vw_outfile = codecs.open(model_path + ".test-audit.vw",
                                      "w", encoding="utf-8",
                                      errors='ignore')
        
    cpdef str name(self):
        return "VWOAA"
    
    cpdef tuple predict(self, list words):
        cdef dict ns_features = self.feature_extractor.get_features(words)
        cdef unicode vw_string = self.convert_features_to_vw_format(ns_features)

        #self.vw_outfile.write(vw_string + u"\n")
        #not actually a probabilityb
        #print "scoring"
        #cdef float cur_prob = self.classifier.score_prediction(vw_string)
        cdef int predicted_class_num = self.classifier.classify(vw_string)
        #print self.verb_indices[predicted_class_num]
        #return cur_prob, self.verb_indices[predicted_class_num] stalling for some reason
        return 1, self.verb_indices[predicted_class_num]


    cpdef unicode convert_features_to_vw_format(self, dict ns_features):
        cdef unicode instance = u"1"
        for nspace in ns_features.keys():
            instance +=  u" |" + nspace + u" "
            instance += u" ".join(ns_features[nspace])
        return instance

    cdef load_verb_indices(self, unicode filename):
        self.verb_indices = {}
        in_file = codecs.open(filename, 'r',encoding="utf-8")
        for line in in_file:
            label = line.split("\t")[0].strip()
            idx = int(line.split("\t")[1])
            self.verb_indices[idx] = label
    cpdef close(self):
        self.classifier.close()
        self.vw_outfile.close()

            
cdef class VWBinaryVerbPredictor(VerbPredictor):
    cpdef public model_path
    cpdef list verb_list
    cpdef verb_indices #not necessary for binary, but loaded anyway
    cpdef classifier
    cpdef public feature_extractor
    cpdef sent_proc
    cpdef int label_ngram
    def __init__(self, unicode model_path,
                 feature_extractor=GermanTaggedFeatureExtractor(),
                 int label_ngram=1):
        self.model_path = model_path
        self.classifier = BinaryLogisticScorer(model_path)
        self.feature_extractor = feature_extractor
        self.load_verb_list(model_path.replace(".model",".labels"))
        self.sent_proc = text_tools.SentenceProcessor()

        
    cpdef str name(self):
        return "VWBinaryVerbPredictor"

    """
    Returns a list of (prediction,score) tuples in order or rank
    """
    cpdef list rank_predictions(self, list words, list choices):
        cdef unicode choice
        cdef float score
        cdef list ranked_list = []
        cdef tuple result
        for choice in choices:
            score = self.score(words, choice)
            result = (score, choice)
            ranked_list.append(result)
        ranked_list.sort(key=itemgetter(0), reverse=True)
        return ranked_list


    cpdef tuple predict(self, list words):
        #return vw.get_prediction(jfe.get_features("".join(words)))
        cdef unicode best_verb = self.verb_list[0]
        cdef unicode current_verb
        cdef float best_prob = -1000000
        cdef float curr_prob
        cdef int predicted_class
        cdef list vw_input
        cdef unicode vw_string
        for current_verb in self.verb_list:
            vw_string = self.convert_example_to_vw_format(words, current_verb)
            cur_prob = self.classifier.score_prediction(vw_string)
            predicted_class = self.classifier.classify(vw_string)
            #sys.stderr.write(str(cur_prob) + "\n")

            if predicted_class < 0:
                cur_prob = -cur_prob
            if cur_prob > best_prob:
                    best_verb = current_verb
                    best_prob = cur_prob
                
        return  best_prob, best_verb

    cpdef float score(self, list words, unicode label):
        vw_string =  self.convert_example_to_vw_format(words, label)
        cdef int prediction = self.classifier.classify(vw_string)
        cdef float cur_prob = self.classifier.score_prediction(vw_string)
        if prediction < 0:
            cur_prob = -cur_prob
        return cur_prob
    

    cpdef unicode convert_example_to_vw_format(self, list words, unicode label_guess):
        cdef dict ns_features = self.feature_extractor.get_features(words)
        cpdef unicode instance = u"1"
        for nspace in ns_features.keys():
            instance +=  u" |" + nspace + u" "
            instance += u" ".join(ns_features[nspace]) + u" "
            instance += u"|verb " + label_guess + " "
            if self.label_ngram > 1:
                instance += " ".join(self.sent_proc.get_ngram_strings_from_text(label_guess, self.label_ngram))
        return instance

    """
    Called upon class creation.  Loads verbs from same directory as model.
    Replaces list if called multiple times.
    """
    cdef load_verb_list(self, unicode filename):
        sys.stderr.write(u"Loading verbs from " + filename + u"\n")
        in_file = codecs.open(filename, 'r',encoding="utf-8")
        self.verb_list = []
        self.verb_indices = {}
        cdef list verb_list = []
        for line in in_file:
            label = line.split("\t")[0].strip()
            idx = int(line.split("\t")[1])
            self.verb_indices[idx] = label
            verb_list.append(label)
        self.verb_list = verb_list



cdef class VWBinaryMostCommonVerbPredictor(VWBinaryVerbPredictor):
    # cpdef model_path
    # cpdef list verb_list
    # cpdef classifier
    # cpdef feature_extractor
    cpdef unicode most_common_label
    def __init__(self,
                 unicode model_path,
                 list sorted_verb_list,
                 feature_extractor=GermanTaggedFeatureExtractor()):
        super(VWBinaryMostCommonVerbPredictor, self).__init__(model_path,
                                              feature_extractor=feature_extractor)
        #Call to super over-writes with a file load.
        self.verb_list = sorted_verb_list
        self.most_common_label = self.verb_list[0]

        sys.stderr.write("\nMost common verb:" + self.most_common_label + "\n")
        
        
    cpdef str name(self):
        return "VWBinaryMostCommon"
    
    cpdef tuple predict(self, list words):
        return 1.0, self.most_common_label

from sklearn.externals import joblib
cdef class SKLSVMVerbPredictor(VerbPredictor):
    cpdef unicode model_path
    cpdef dict verb_indices
    cpdef classifier
    cpdef feature_extractor
    
    def __init__(self,
                 unicode model_path,
                 feature_extractor=GermanTaggedFeatureExtractor(),
                 ):
        self.model_path = model_path
        self.feature_extractor = feature_extractor
        #vw = VW(loss_function='logistic', i = model_path)
        self.verb_indices = {}
        self.load_verb_indices(model_path.replace("-svm.pkl","") + ".labels")
        sys.stderr.write("Loading classifier: " + model_path + "\n")
        self.classifier = joblib.load(model_path)
        
    cpdef str name(self):
        return "SKLearnSVMVerbPredictor"
    
    cpdef tuple predict(self, list words):
        #cdef unicode preverb = u" ".join(words)
        cdef dict ns_features = self.feature_extractor.get_features_from_tagged(words)
        cdef unicode instance_string = self.convert_to_instance(ns_features)
        
        cdef float cur_prob = 1.0 #not meaningful here
        cdef int predicted_class_num = self.classifier.predict(instance_string)
        #This might need to return the first index of the array
        return cur_prob, self.verb_indices[predicted_class_num]

    cpdef unicode convert_to_instance(self, dict ns_features):
        cdef unicode instance = u""
        #SKL doesn't have namespaces; just combine them with a dash

        for nspace in ns_features.keys():
            for feature in ns_features[nspace]:
                instance += nspace + u"-" + feature + u" "
        return instance            
                    

    cdef load_verb_indices(self, unicode filename):
        in_file = codecs.open(filename, 'r',encoding="utf-8")
        for line in in_file:
            label = line.split("\t")[0]
            idx = int(line.split("\t")[1])
            self.verb_indices[idx] = label
