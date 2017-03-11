# -*- coding: utf-8 -*-
from distutils.core import setup
from bunny_lure import *
from collections import defaultdict
from itertools import chain
import codecs
from feature_extractor import *
import text_tools
#from Cython.Build import cythonize

from filereaders import *
from vowpal_porpoise import VW
import random
import kenlm
import os
from subprocess import Popen, PIPE, STDOUT

cdef class Trainer:
    def __init__(self):
       raise NotImplementedError("Trainer must be subclassed.")

    cpdef str name(self):
        return "Trainer"

    cpdef add_example(self, list words, unicode true_label):
        raise NotImplementedError("add_example must be subclassed.")

    cpdef add_examples_from_file(self,  file_reader):
        for line in file_reader:
            self.add_example(file_reader.get_features(line)[""],
                        file_reader.get_class(line))

            
    cpdef save_model(self, unicode filename):
        raise NotImplementedError("Must be subclassed.")


cdef class FeaturedTrainer(Trainer):
    cpdef add_featured_example(self, dict namespace_features):
        raise NotImplementedErroror


cdef class VWBinaryMultiClassTrainer(Trainer):
    cpdef list all_labels
    cpdef dict all_classes_dict
    cpdef vw
    cpdef vw_outfile
    cpdef feature_extractor
    
    def __init__(self,
                 unicode model_name,
                 list all_classes,
                 unicode function=None,
                 float learning_rate=0.05,
                 float l1=0,
                 float l2=0,
                 int passes=2,
                 feature_extractor=GermanTaggedFeatureExtractor(),
                 quadratic='pv',
                 cubic=None,
                 int bits=30,
                 bint audit=False,
                 unicode extra_arguments=u""
                 ):
        base_command = "vw --kill_cache "
        base_command += " -b " + unicode(bits)
        if cubic is not None:
            base_command += " --cubic " + unicode(cubic)
        base_command += " " + extra_arguments
        
        
        #TODO(acg) Implement auditing for binary. Jst copy/past from OAA
        self.vw = VW(moniker=model_name,
                     vw=base_command,
                     passes=passes,
                     loss=function,
                     learning_rate=learning_rate,
                     l1=l1,
                     l2=l2,
                     quadratic=quadratic,
                     subsentence_train=True,
                     #ngram=2
                     )
        self.feature_extractor = feature_extractor
        self.vw_outfile = codecs.open(model_name + u".vw",
                                      "w",
                                      encoding="utf-8",
                                      errors='ignore')
        self.all_classes_dict = dict()
        self.all_labels = all_classes
        for label in self.all_labels:
            self.all_classes_dict[label] = 1
        self.export_label_names(model_name + ".labels")

    cdef export_label_names(self, unicode labels_filename):
        of = codecs.open(labels_filename, "w",encoding="utf-8")
        for label in self.all_classes_dict:
            of.write(label + u"\t" + unicode(self.all_classes_dict[label]))
            of.write("\n")
        of.close()
        sys.stderr.write("Wrote " + labels_filename + "\n")

    cpdef str name(self):
        return "VWBinaryMultiClassTrainer"

    """
    "words" here should be the list of words (preverb)
    """
    cpdef list convert_example_to_vw_format(self, dict ns_features, unicode true_label):
        if not true_label in self.all_classes_dict:
            return []
        cdef int sub_examples = 0
        cpdef unicode current_example
        cpdef unicode current_class
        cdef unicode vw_instance
        cpdef unicode binary_label = u'unset'
        cpdef list all_instances = list()
        cdef float false_label_weight = 0.5/(float(len((self.all_classes_dict.keys()))) - 1)
        cdef float true_label_weight = 0.5
        for current_class in self.all_labels:
            vw_instance = u''
            if current_class == true_label:
                binary_label = u'1 ' + unicode(true_label_weight)
            else:
                binary_label = u'-1 ' + unicode(false_label_weight)
            #self.vw.send_example(str(binary_label), features=words + [current_class])
            vw_instance += binary_label
            for ns in ns_features.keys():
                vw_instance +=  u' |' + ns + u' '
                vw_instance += u' '.join(ns_features[ns])
                    
            vw_instance += u' |verb ' + current_class
            all_instances.append(vw_instance)
            #self.vw.push_instance(vw_instance)
        return all_instances

    cpdef save_model(self, unicode filename):
        self.vw.save_model(filename)
    

    #TODO(alvin) Get the pxd back to type the FileReader
    #or add a manual type check to prevent people
    #from sending a Python file readerx
    cpdef add_examples_from_file(self, file_reader):
        cdef int total_read = 0
        cdef int sub_examples = 0
        cdef instance_file
        cpdef unicode current_example
        cpdef unicode current_class
        cpdef unicode true_label
        cdef unicode vw_instance = None
        cpdef unicode binary_label = u'unset'
        cpdef list words
        sys.stderr.write("Training...\n")
        with self.vw.training():
            for line in file_reader:
                total_read += 1
                ns_features = file_reader.get_features(line)
                true_label = file_reader.get_class(line)
                vw_instances = self.convert_example_to_vw_format(ns_features, true_label)
                for vw_instance in vw_instances:
                    self.vw_outfile.write(vw_instance + u"\n")
                    self.vw.push_instance(vw_instance)
                self.vw_outfile.flush()
        sys.stderr.write('Read ' + unicode(total_read) + 'sentences.\n')

cdef class VWBinaryMultipleChoiceTrainer(VWBinaryMultiClassTrainer):
    cpdef int num_negative_examples
    cpdef int label_ngram
    cpdef sent_proc
    def __init__(self,
                 unicode model_name,
                 list all_classes,
                 unicode function=u'logistic',
                 float learning_rate=0.05,
                 float l1=0,
                 float l2=0,
                 int passes=1,
                 int num_negative_examples=4,
                 int label_ngram=1,
                 unicode quadratic=None,
                 unicode cubic=None,
                 int bits = 30,
                 feature_extractor=JapaneseSentenceFeatureExtractor(),
                 extra_arguments=""
                 ):
        cdef unicode vw_command = u'vw --kill_cache -b ' + unicode(bits) + u" "
        if cubic is not None:
            vw_command += u" --cubic " + unicode(cubic)
        vw_command += u" " + extra_arguments
        self.vw = VW(moniker=model_name,
                     vw=vw_command,
                     passes=passes,
                     loss=function,
                     learning_rate=learning_rate,
                     l1=l1,
                     l2=l2,
                     quadratic=quadratic,
                     subsentence_train=False,
                     #ngram=2
                     )
        self.sent_proc = text_tools.SentenceProcessor()
        self.label_ngram = label_ngram
        self.num_negative_examples = num_negative_examples
        self.feature_extractor = feature_extractor
        self.vw_outfile = codecs.open(model_name + u".vw", "w",encoding="utf-8")
        self.all_classes_dict = dict()
        self.all_labels = all_classes
        for label in self.all_labels:
            self.all_classes_dict[label] = 1
        sys.stderr.write("Exporting labels.\n")
        self.export_label_names(model_name + ".labels")

    cpdef str name(self):
        return "VWBinaryMultipleChoiceTrainer"

    """
    "words" here should be the list of words (preverb)
    Randomly chooses from list for negative examples
    """
    cpdef list convert_example_to_vw_format(self, dict ns_features, unicode true_label, bint non_overlapping_answers=True):
        cdef list true_label_split  = true_label.split()
        cdef dict true_label_tokens = dict()
        for tok in true_label_split:
            true_label_tokens[tok] = tok
            
        cdef dict rand_indices = dict()
        cdef int random_num
        cdef list false_label_split
        while len(rand_indices) < self.num_negative_examples:
            random_num = random.randint(0,len(self.all_classes_dict.keys()) - 1)
            if not random_num in rand_indices:
                if non_overlapping_answers:
                    false_label_split = self.all_labels[random_num].split()
                    for false_token in false_label_split:
                        if false_token in true_label_tokens:
                            continue
                        else:
                            rand_indices[random_num] = random_num
                else:
                    rand_indices[random_num] = random_num
        cpdef unicode current_example
        cpdef unicode current_class
        cdef int i
        cdef unicode vw_instance
        cpdef unicode binary_label = u'unset'
        cpdef list all_instances = list()
        cdef float true_label_weight = 0.5
        cdef float false_label_weight = (1.0 - true_label_weight)/(float(self.num_negative_examples))
        cdef unicode correct_example = u'1 ' + unicode(true_label_weight)
        for ns in ns_features.keys():
            correct_example +=  u' |' + ns + u' '
            correct_example += u' '.join(ns_features[ns])
        correct_example += u' |verb ' + true_label + u" "
        if self.label_ngram > 1:
            correct_example += u" ".join(self.sent_proc.get_ngram_strings_from_text(true_label, self.label_ngram))

        for i in rand_indices.keys():
            current_class = self.all_labels[i]
            vw_instance = u''
            if current_class != true_label:
                binary_label = u'-1 ' + unicode(false_label_weight)
            #self.vw.send_example(str(binary_label), features=words + [current_class])
            vw_instance += binary_label
            for ns in ns_features:
                vw_instance +=  u' |' + ns + u' '
                vw_instance += u' '.join(ns_features[ns])
                    
            vw_instance += u' |verb ' + current_class + u" "
            if self.label_ngram > 1:
                vw_instance += " ".join(self.sent_proc.get_ngram_strings_from_text(current_class, self.label_ngram))
            all_instances.append(vw_instance)
            #self.vw.push_instance(vw_instance)
        all_instances.append(correct_example)
        #print correct_example
        return all_instances
    
cdef class EndOfSentenceTrainer:
    pass

from prediction import VWOAAHierarchicalPredictor
cdef class VWOAAHierarchicalSentenceTrainer(VWOAAMultiClassTrainer):
    cpdef unicode function
    cpdef public unicode model_dir
    cpdef float learning_rate
    cpdef float l1
    cpdef l2
    cpdef int ngram
    cpdef int passes
    cpdef unicode quadratic
    cpdef unicode cubic
    cpdef unicode base_command
    cpdef str name(self):
     return "VWOAAH"

    def __init__(self,
                 unicode model_dir,
                 list all_classes,
                 unicode function=None,
                 float learning_rate=1.0,
                 float l1=0.0,
                 float l2=0.0,
                 int ngram=1,
                 int passes=1,
                 int max_sentence_length=25, #replace with sentene extractor
                 quadratic=None,
                 cubic=None,                 
                 feature_extractor=GermanTaggedFeatureExtractor(),
                 base_command=u"vw --kill_cache",
                 int bits=30,
                 unicode extra_arguments=u""
                 ):
        base_command += u" -b " + unicode(bits)
        self.model_dir = model_dir
        self.function = function
        self.learning_rate = learning_rate
        self.l1=l1
        self.l2=l2
        self.ngram=ngram
        self.passes=passes
        self.quadratic=quadratic
        self.cubic=cubic
        self.feature_extractor=feature_extractor
        self.base_command = base_command
        self.max_sentence_length=max_sentence_length
        sys.stderr.write("Building hierarchical model.\n")
        sys.stderr.write(model_dir + "\n")
        dir = os.path.dirname(model_dir)
        try:
            os.stat(model_dir)
            sys.stderr.write("Directory exists. Deleting and recreating.\n.")
            import shutil
            shutil.rmtree(model_dir)
            os.mkdir(model_dir)
        except:
            os.mkdir(model_dir)
        
        self.all_classes_dict = dict()
        self.all_labels = all_classes
        for i in xrange(len(self.all_labels)):
            self.all_classes_dict[self.all_labels[i]] = i + 1
        self.labels_file = model_dir + "/oaa-hier.labels"
        self.export_label_names(self.labels_file)


            

    cpdef add_examples_from_file(self, file_reader):
        cpdef unicode current_example
        cpdef unicode current_class
        cpdef unicode true_label
        cdef unicode vw_instance = None
        cpdef unicode binary_label = u'unset'
        cdef list preverb
        cpdef list words
        cdef int current_layer_num
        cdef list preverb_inc#for incremental sentence training
        cpdef unicode word
        cdef vw_layer
        cdef vw_background
        cdef list previous_layers = list()
        cdef int preverb_end
        sys.stderr.write( "Training...\n")
        """
        #Step 1: Make background classifier (Skip for now)
        with self.vw.training():
            vw_background = VW(
                moniker=model_name + "_background",
                quadratic=self.quadratic,
                vw=self.base_command,
                passes=self.passes,
                loss=self.function,
                ngram=self.ngram,
                #specifying loss breaks model in some VW versions
                learning_rate=self.learning_rate,
                l1=self.l1,
                l2=self.l2,
                quadratic=self.quadratic,
                cubic=self.cubic,
                oaa=len(all_classes))
            
                             
            for line in file_reader:
                preverb = file_reader.get_context_tokens_notag(line)
                true_label = file_reader.get_class(line)
                for word in preverb:
                    ns_features = self.feature_extractor.get_features(preverb)
                    vw_instance = self.convert_example_to_vw_format(ns_features, true_label)
                    self.vw.push_instance(vw_instance)
                    self.instance_file.write(vw_instance + u"\n")
                self.instance_file.close()
        """
        cdef unicode pred
        prev_vw = None
        for current_layer_num in xrange(1, self.max_sentence_length + 1):                
            file_reader.reset()
            file_reader.set_valid_classes(self.all_labels)
            vw_layer = VW(moniker=self.model_dir + "/" + unicode(current_layer_num), #append ".model"
                         quadratic=self.quadratic,
                         vw=self.base_command,
                         passes=self.passes,
                         loss=self.function,
                         ngram=self.ngram,
                         #specifying loss breaks model in some VW versions
                         learning_rate=self.learning_rate,
                         l1=self.l1,
                         l2=self.l2,
                         quadratic=self.quadratic,
                         cubic=self.cubic,
                         oaa=len(self.all_labels))
            with vw_layer.training():
            #Step 2: Build hierarchical model.
                if current_layer_num > 1:
                    prev_layer = current_layer_num - 1
                    prev_vw = VWOAAHierarchicalPredictor(self.model_dir,
                                                             feature_extractor=self.feature_extractor,
                                                             max_layer=prev_layer)
                for line in file_reader:
                    ns_features = {}
                    preverb = file_reader.get_context_tokens_notag(line)
                    #layers start at 1, not 0
                    if not len(preverb) - current_layer_num >= 1:
                        continue
                    true_label = file_reader.get_class(line)
                    preverb = preverb[:current_layer_num]
                    ns_features = self.feature_extractor.get_features(preverb)
                    if current_layer_num > 1:
                        ns_features["hierarchical"] = list()
                        #should use entire preverb here, but be careful
                        #prev_score, prev_prediction = prev_vw.predict(preverb[prev_layer:])
                        #ns_features["hierarchical"] = prev_vw.get_hierarchical_features(preverb[prev_layer:])
                        ns_features["hierarchical"].append(unicode(current_layer_num) + u":" + prev_vw.predict(preverb[prev_layer:])[1])
                        #use previous model(s) to predict current answer

                    vw_instance = self.convert_example_to_vw_format(ns_features, true_label)
                    vw_layer.push_instance(vw_instance)
                if current_layer_num > 1:
                    prev_vw.close()
                    #self.instance_file.write(vw_instance + u"\n")
            #new_predictor = VWOAAHierarchicalPredictor(
            #vw_layer.close()


    # cpdef generate_predictor(unicode predictor_type):
    #     pass
    
                
cdef class VWOAAMultiClassTrainer(Trainer):
    cpdef public list all_labels
    cpdef public dict all_classes_dict
    #cdef public instance_file
    cpdef public unicode labels_file
    #maps labels to integers for VW
    cpdef public vw
    cpdef public feature_extractor
    cpdef public bint incremental_sentence
    cpdef public int max_sentence_length
    cpdef bint audit
    cpdef vw_audit_file
    cpdef sentence_audit_file
    cpdef unicode model_name
    def __init__(self,
                 unicode model_name,
                 list all_classes,
                 unicode function=u'logistic',
                 bint incremental_sentence=False,
                 float learning_rate=0.5,
                 float l1=0.0,
                 float l2=0.0,
                 int ngram=1,
                 int passes=1,
                 quadratic=None,
                 cubic=None,
                 int max_sentence_length=25,
                 feature_extractor=GermanTaggedFeatureExtractor(),
                 bint audit=True,
                 extra_arguments = u""
                 
                 ):
        print "Initializing VWOAA Trainer"
        base_command = 'vw --kill_cache -b 30'
        base_command += u" " + extra_arguments
        self.vw = VW(moniker=model_name,
                     quadratic=quadratic,
                     vw=base_command,
                     passes=passes,
                     loss=function,
                     ngram=ngram,
                     #specifying loss breaks model in some VW versions
                     learning_rate=learning_rate,
                     l1=l1,
                     l2=l2,
                     quadratic=quadratic,
                     cubic=cubic,
                     oaa=len(all_classes))
        self.model_name = model_name
        self.max_sentence_length = max_sentence_length
        self.incremental_sentence = incremental_sentence
        self.feature_extractor = feature_extractor
        self.all_classes_dict = dict()
        self.all_labels = all_classes
        for i in xrange(len(self.all_labels)):
            self.all_classes_dict[self.all_labels[i]] = i + 1
        self.labels_file = model_name + ".labels"
        self.export_label_names(self.labels_file)
        #instance_filename = model_name + "-oaa.vw"
        #self.instance_file = codecs.open(instance_filename, 'w', encoding="utf-8")
        self.audit = audit
        if audit:
            audit_filename = self.model_name + "-audit.vw"
            sentence_filename = self.model_name + "-sentence-audit.vw"
            self.vw_audit_file = codecs.open(audit_filename, "w", encoding="utf-8")
            self.sentence_audit_file = codecs.open(sentence_filename, "w", encoding="utf-8")
            sys.stderr.write("VW audit file: " + audit_filename + "\n")
            sys.stderr.write("Sentence audit file: " + sentence_filename + "\n")
                                             
            
        
    cdef export_label_names(self, unicode labels_filename):
        of = codecs.open(labels_filename, "w",encoding="utf-8")
        for label in self.all_classes_dict:
            of.write(label + u"\t" + unicode(self.all_classes_dict[label]))
            of.write("\n")
        of.close()
        sys.stderr.write("Wrote " + labels_filename + "\n")

    cpdef str name(self):
        return "VWOAAMultiClassTrainer"

    cpdef unicode convert_example_to_vw_format(self, dict ns_features, unicode true_label):
        cdef unicode instance =  unicode(self.all_classes_dict[true_label])
        for nspace in ns_features.keys():
            instance +=  u" |" + nspace + u" "
            instance += u" ".join(ns_features[nspace])
        return instance

    cpdef save_model(self, unicode filename):
        self.vw.save_model(filename)
    

    #TODO(alvin) Get the pxd back to type the FileReader
    #or add a manual type check to prevent people
    #from sending a Python file reader
    cpdef add_examples_from_file(self, file_reader):
        cdef bint keep_tags = True #put in function definition
        
        cdef int sub_examples = 0
        cdef int processed = 0
        cpdef unicode current_example
        cpdef unicode current_class
        cpdef unicode true_label
        cdef unicode vw_instance = None
        cpdef unicode binary_label = u'unset'
        cdef list preverb
        cpdef list words
        cdef list preverb_inc#for incremental sentence training
        cpdef unicode word
        sys.stderr.write( "Training...\n")
        with self.vw.training():
            for line in file_reader:
                if keep_tags:
                    preverb = file_reader.get_context_tokens(line)
                else:
                    preverb = file_reader.get_context_tokens_notag(line)
                if preverb is None:
                    continue
                preverb_inc = list()
                if self.incremental_sentence:
                    for word in preverb:
                        preverb_inc.append(word)
                        ns_features = self.feature_extractor.get_features(preverb_inc)
                        #ns_features["length"] = unicode(len(preverb_inc))
                        true_label = file_reader.get_class(line)
                        vw_instance = self.convert_example_to_vw_format(ns_features, true_label)
                        self.vw.push_instance(vw_instance)
                        #self.instance_file.write(vw_instance + u"\n")

                        
                else:
                    ns_features = self.feature_extractor.get_features(preverb)
                    true_label = file_reader.get_class(line)
                    vw_instance = self.convert_example_to_vw_format(ns_features, true_label)
                    self.vw.push_instance(vw_instance)
                if self.audit:
                    self.sentence_audit_file.write(file_reader.unparsed_line)
                    self.vw_audit_file.write(vw_instance + "\n")
                processed += 1
                if processed % 10000 == 0:
                    print u"Processed: ",unicode(processed)
                    #elf.vw.push_instance(vw_instance)
                    #self.instance_file.write(vw_instance + u"\n")
        #self.instance_file.close()
        print u"Trained on ", unicode(processed) + u" examples."
        if self.audit:
            self.sentence_audit_file.close()
            self.vw_audit_file.close()

import numpy
cdef class SyntheticVWOAAMultiClassTrainer(VWOAAMultiClassTrainer):
    
    cpdef prior(self, int size, float alpha):
        """
        Generate a vector of given size all filled with this alpha
        """

        return numpy.ones(size) * alpha

    cpdef generate_synthetic(self, int vocab_size=100,
                           int num_verbs=5,
                           float alpha_background=0.9,
                           float alpha_bigram=3.0,                       
                           float alpha_verb=5.0,
                           int sentences=1000,
                           int observation_length=3):

        corpus = {}

        # Create the inventory of words
        vocab = ["vocab%06i" % x for x in xrange(vocab_size)]
        verbs = ["verb%06i" % x for x in xrange(num_verbs)]

        # Distribution over verbs
        verb_dist = numpy.random.dirichlet(self.prior(num_verbs, alpha_verb))

        # generate background distribution
        word_dist = numpy.random.dirichlet(self.prior(vocab_size, alpha_background))

        #for each verb, generate bigram distribution
        bigram = defaultdict(dict)
        for verb in verbs:
            for context in chain(vocab, [None]):
                bigram[verb][context] = numpy.random.dirichlet(word_dist * alpha_bigram)

        for sentence in xrange(sentences):
            observation = []
            verb = numpy.random.choice(verbs, 1, p=verb_dist)[0]
            sentence_length = numpy.random.poisson(observation_length)
            prev = None
            for w_n in xrange(sentence_length):
                    #w_n ~ previous word's distribution (Phi_i)
                    w_n = numpy.random.choice(vocab, p=bigram[verb][prev])
                    observation.append(w_n)
                    prev = w_n
            corpus[sentence] = (unicode(verb), observation)


        return corpus

        
    cpdef add_examples_from_file(self, file_reader):
        #ignores file_reader
        corpus = self.generate_synthetic()
        cdef int sub_examples = 0
        cpdef unicode current_example
        cpdef unicode current_class
        cpdef unicode true_label
        cdef unicode vw_instance = None
        cpdef unicode binary_label = u'unset'
        cdef list words
        cdef dict ns_features = dict()
        sys.stderr.write( "Training...\n")
        with self.vw.training():
            for i in corpus:
                true_label = corpus[i][0]
                words = corpus[i][1]
                ns_features["preverb"] = context
                vw_instance = self.convert_example_to_vw_format(ns_features, true_label)
                self.vw.push_instance(vw_instance)




            

import numpy, sys
cdef class SyntheticVWBinaryMultiClassTrainer(VWBinaryMultiClassTrainer):
    cpdef public int vocab_size
    cpdef public int num_verbs
    cpdef public float alpha_background
    cpdef public float alpha_bigram                       
    cpdef public float alpha_verb
    cpdef public int num_sentences
    cpdef public int observation_length
    
    def init(unicode model_name,
             list all_classes,
             unicode function=u'logistic',
             float learning_rate=10,
             float l1=0.00001,
             int passes=2,
             int vocab_size=1000,
             int num_verbs=100,
             float alpha_background=0.9,
             float alpha_bigram=3.0,                       
             float alpha_verb=5.0,
             int num_sentences=1000,
             int observation_length=3
         ):
        all_classes = []
        self.num_verbs = num_verbs
        self.vocab_size = vocab_size
        self.alpha_background = alpha_background
        self.alpha_bigram = alpha_bigram
        self.alpha_verb = alpha_verb
        self.num_sentences = num_sentences
        self.observation_length = observation_length
        super(VWBinaryMultiClassTrainer, self).__init__(model_name,
                                                        all_classes,
                                                        function,
                                                        learning_rate,
                                                        l1,
                                                        passes)
        
        

    cpdef set_corpus_parameters(self, int vocab_size=1000,
                           int num_verbs=100,
                           float alpha_background=0.9,
                           float alpha_bigram=3.0,                       
                           float alpha_verb=5.0,
                           int num_sentences=1000,
                           int observation_length=3):
        self.num_verbs = num_verbs
        self.vocab_size = vocab_size
        self.alpha_background = alpha_background
        self.alpha_bigram = alpha_bigram
        self.alpha_verb = alpha_verb
        self.num_sentences = num_sentences
        self.observation_length = observation_length

    cpdef str name(self):
        return "SyntheticOAA"
    cpdef prior(self, int size, float alpha):
        """
        Generate a vector of given size all filled with this alpha
        """

        return numpy.ones(size) * alpha

    cpdef generate_synthetic(self):
        vocab_size = self.vocab_size
        alpha_background = self.alpha_background
        alpha_bigram = self.alpha_bigram
        alpha_verb = self.alpha_verb
        sentences = self.num_sentences
        observation_length = self.observation_length
        num_verbs = self.num_verbs

        sys.stderr.write( "num verbs:" + num_verbs + "\n")
        sys.stderr.write("sentences:" + sentences + "\n")
        sys.stderr.write("vocab size" + vocab_size + "\n")
        corpus = {}

        # Create the inventory of words
        vocab = ["vocab%06i" % x for x in xrange(vocab_size)]
        verbs = ["verb%06i" % x for x in xrange(num_verbs)]

        # Distribution over verbs
        verb_dist = numpy.random.dirichlet(self.prior(num_verbs, alpha_verb))

        # generate background distribution
        word_dist = numpy.random.dirichlet(self.prior(vocab_size, alpha_background))

        #for each verb, generate bigram distribution
        bigram = defaultdict(dict)
        self.all_classes_dict = dict()
        for verb in verbs:
            for context in chain(vocab, [None]):
                bigram[verb][context] = numpy.random.dirichlet(word_dist * alpha_bigram)

        for sentence in xrange(sentences):
            observation = []
            verb = numpy.random.choice(verbs, 1, p=verb_dist)[0]
            self.all_classes_dict[unicode(verb)] = 1
            sentence_length = numpy.random.poisson(observation_length)
            prev = None
            for w_n in xrange(sentence_length):
                    #w_n ~ previous word's distribution (Phi_i)
                    w_n = numpy.random.choice(vocab, p=bigram[verb][prev])
                    observation.append(w_n)
                    prev = w_n
            corpus[sentence] = (unicode(verb), observation)

        self.all_labels = self.all_classes_dict.keys()
        sys.stderr.write(unicode(self.all_labels) + u"\n")
        return corpus

    cpdef add_examples_from_file(self, file_reader):
        #ignores file_reader
        corpus = self.generate_synthetic()
        cdef int sub_examples = 0
        cpdef unicode current_example
        cpdef unicode current_class
        cpdef unicode true_label
        cdef list vw_instances = None
        cpdef unicode binary_label = u'unset'
        cdef list words
        cdef dict ns_features = dict()
        sys.stderr.write( "Training..." + "\n")
        with self.vw.training():
            for i in corpus:
                true_label = corpus[i][0]
                words = corpus[i][1]
                ns_features["preverb"] = words
                vw_instances = self.convert_example_to_vw_format(ns_features, true_label)

                for vw_instance in vw_instances:
                    self.vw.push_instance(vw_instance)


# class VWBinaryMultipleChoiceTrainer:
#     def __init__:
        
    
from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
cdef class SKLSVMMultiClassTrainer(Trainer):
    cpdef list all_labels
    cpdef dict all_classes_dict
    #maps labels to integers for VW
    cpdef unicode model_name
    cpdef classifier
    def __init__(self,
                 unicode model_name,
                 list all_classes,
                 #unicode function=u'hinge',
                 #float learning_rate=0.5,
                 #float l1=0.00001,
                 #int passes=10
                 ):
        self.model_name = model_name
        self.classifier = svm.SVC()
        self.all_classes_dict = dict()
        self.all_labels = all_classes
        for i in xrange(len(self.all_labels)):
            self.all_classes_dict[self.all_labels[i]] = i + 1
        self.export_label_names(model_name.replace(".model","-svm") + ".labels")

    cdef export_label_names(self, unicode labels_file):
        of = codecs.open(labels_file, 'w', encoding="utf-8")
        for label in self.all_classes_dict:
            of.write(label + u"\t" + unicode(self.all_classes_dict[label]))
            of.write("\n")
        of.close()

    cpdef str name(self):
        return "SKLearnMultiClassTrainer"

    """
    Returns an example, label (integer) pair
    """
    cpdef tuple convert_to_instance(self, dict ns_features, unicode true_label):
        cdef unicode instance = u""
        cdef int instance_num = self.all_classes_dict[true_label]
        #SKL doesn't have namespaces; just combine them with a dash
        for nspace in ns_features.keys():
            for feature in ns_features[nspace]:
                instance += nspace + u"-" + feature + u" "
        return instance, instance_num

  

    #TODO(alvin) Get the pxd back to type the FileReader
    #or add a manual type check to prevent people
    #from sending a Python file reader
    cpdef add_examples_from_file(self, file_reader):
        cdef int sub_examples = 0
        cpdef unicode current_example
        cpdef unicode current_class
        cpdef unicode true_label
        cdef unicode instance = None
        cpdef list words
        cpdef list instances = []
        cdef list int_labels = []
        #cdef int int_label
        for line in file_reader:
            ns_features = file_reader.get_features(line)
            true_label = file_reader.get_class(line)
            instance, int_label = self.convert_to_instance(ns_features, true_label)
            instances.append(instance)
            int_labels.append(int_label)

        transformer = CountVectorizer()
        sys.stderr.write("Transforming text to ints...")
        X = transformer.fit_transform(instances)
        #train model
        sys.stderr.write( "Training...\n")
        self.classifier.fit(X, int_labels)
        #save model
        sys.stderr.write( "Saving...\n")
        joblib.dump(self.classifier, self.model_name + "-svm.pkl")
        sys.stderr.write("Doing cross-validation on full context.")
        #Run sanity check
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, instances, test_size=0.2)
        clf = svm.SVC().fit(X_train, y_train)
        print "\nCV score:",str(clf.score(X_test,y_test))


"""
This creates KenLM language models with lmplz that filter based on the label.
A component used for creating multiple language models based on verbs.
"""

cdef class FilteredKenLMBuilder:
    cpdef unicode label_name
    cpdef unicode output_dir
    cpdef public unicode lmplz_path
    cpdef lmplz_proc
    def __init__(self, unicode label_name, unicode output_dir, int n):
        self.label_name = label_name
        #self.valid_classes = dict()
        #self.valid_classes[u'v1'] = 1
        #self.label_name = u'v1'
        self.output_dir = output_dir
        self.lmplz_path= u"lmplz"
        self.lmplz_proc = Popen([self.lmplz_path + ' -o ' + unicode(n)],
                                stdin=PIPE,
                                stdout=open(output_dir + u"/" + label_name + u".arpa", u"w"),
                                env=os.environ,
                                shell=True,
                                universal_newlines=True)

    cpdef is_valid_class(self, unicode class_name):
        return class_name in self.valid_classes

    cpdef process_sentence(self, unicode context, unicode label):
        if label == self.label_name:
            self.lmplz_proc.stdin.write((context + u"\n").encode("utf8"))
            #self.lmplz_proc.communicate(input=context + u"\n")

    def close_lmplz(self):
        self.lmplz_proc.stdin.close()


cdef class KenLMTrainer(Trainer):
    cpdef file_reader
    cpdef unicode output_dir
    cpdef int n
    cpdef unicode lmplz_path
    cpdef lmplz_proc

    
    def __init__(self,int n, unicode output_dir=None, unicode model_name=None):
        self.n = n
        self.output_dir = output_dir

        self.lmplz_path= u"lmplz"
        command = self.lmplz_path + ' -o ' + unicode(n)
        self.lmplz_proc = Popen(command,
                                stdin=subprocess.PIPE,
                                stderr=open(output_dir + "/multilm.log","w"),
                                stdout=open(output_dir + u"/" + model_name + u".arpa", u"w"),
                                env=os.environ,
                                shell=True,
                                universal_newlines=True,
                                close_fds=False)


    cpdef process_sentence_string(self, unicode sentence):
        self.lmplz_proc.stdin.write((sentence + u"\n").encode("utf8"))

    cpdef process_sentence(self, list words):
        cdef unicode sentence = u" ".join(words).replace(u"<s>",u"")
        self.lmplz_proc.stdin.write((sentence + u"\n").encode("utf8"))


    def close_lmplz(self):
        self.lmplz_proc.stdin.close()        

# cdef class Word2VecTrainer(Trainer):
#     cpdef unicode output_filename
#     cpdef add_examples_from_file(self, file_reader):
#         model = gensim.models.Word2Vec(file_reader, size=100, window=5, min_count=5, workers=4)
#         model.save(output_filename)
        
#     def __init__(self, unicode model_name):
#         self.output_filename = model_name + u".w2v"

    


    
""""Builds LMs simultaneously in RAM."""
cdef class MultiLMTrainer(Trainer):
    cpdef list all_labels
    cpdef list class_counts
    cpdef dict all_classes_dict #label->lmplz trainer
    cpdef unicode out_dir
    cpdef feature_extractor
    cpdef _background
    #TODO(acg) list all_classes is redundant
    def __init__(self,
                 list all_classes,
                 unicode output_dir,
                 int n=2,
                 feature_extractor=GermanTaggedFeatureExtractor(),
                 list class_counts=None
                 ):
        
        self.feature_extractor = feature_extractor
        self.all_classes_dict = dict()
        self.all_labels = all_classes
        self.class_counts = class_counts
        try:
            os.stat(output_dir)
        except:
            os.mkdir(output_dir)

        self._background = KenLMTrainer(n, output_dir=output_dir,
                                        model_name=u"_background")
        for label in self.all_labels:
            self.all_classes_dict[label] = KenLMTrainer(n, model_name=label,output_dir=output_dir)
                        

        self.export_label_names(output_dir + u"/labels.txt")
        sys.stderr.write( "Creating model in " + output_dir + "\n")

    cdef export_label_names(self, unicode labels_filename):
        #os.remove(labels_filename)
        of = codecs.open(labels_filename, "w+",encoding="utf-8", errors='ignore')
        for label_count in self.class_counts:
            label = label_count[0]
            count = label_count[1]
            of.write(label + "\t" + unicode(count))
            of.write("\n")
        of.close()
        sys.stderr.write("Wrote " + labels_filename + "\n")

    cpdef str name(self):
        return "MultiLMTrainer"

    """
    "words" here should be the list of words (preverb)
    """

    cpdef save_model(self, unicode filename):
        pass
    

    cpdef add_example(self, list words, unicode label):
        self._background.process_sentence(words)
        if label in self.all_classes_dict:
            self.all_classes_dict[label].process_sentence(words)            
    
    cpdef add_examples_from_file(self, file_reader):
        cdef list line
        cdef unicode label, context
        for line in file_reader:
            label = file_reader.get_class(line)
            self.add_example(file_reader.get_context_tokens_notag(line),
                             label)
        sys.stderr.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Closing files.\n")
        for lmplz in self.all_classes_dict:
            sys.stderr.write("^^^^^^^^^^^^^^^^^^^^^^^^^Closing " + lmplz + "\n")
            self.all_classes_dict[lmplz].close_lmplz()
        sys.stderr.write("Closing _background\n")
        self._background.close_lmplz()

        
cdef class WhenToTrustTrainer(Trainer):
    cpdef public predictor
    def __init__(self, predictor):
        self.predictor = predictor

    cpdef add_examples_from_file(self, file_reader):
        cdef bint keep_tags = True #put in function definition
        
        cdef int sub_examples = 0
        cdef int processed = 0
        cpdef unicode current_example
        cpdef unicode current_class
        cpdef unicode true_label
        cdef unicode vw_instance = None
        cpdef unicode binary_label = u'unset'
        cdef list preverb
        cdef unicode predicted_answer
        cdef float prediction_prob
        cpdef list words
        cdef list preverb_inc#for incremental sentence training
        cpdef unicode word
        sys.stderr.write( "Training...\n")
        with self.vw.training():
            for line in file_reader:
                true_label = file_reader.get_class(line)
                if keep_tags:
                    preverb = file_reader.get_context_tokens(line)
                else:
                    preverb = file_reader.get_context_tokens_notag(line)
                    
                preverb_inc = list()
                if self.incremental_sentence:
                    for word in preverb:
                        preverb_inc.append(word)
                        prediction_prob, predicted_answer = predictor.predict(preverb_inc)
                        
                        ns_features = self.predictor.feature_extractor.get_features(preverb_inc)
                        ns_features["prediction"] = []
                        ns_features["prediction"].append(predicted_answer)
                        ns_features["prediction"].append(u"score:"+unicode(prediction_prob))
                        ns_features["prediction"].append(u"length:"+unicode(len(preverb)))
                        ns_features["prediction"].append(u"true_label:"+true_label)
                        vw_instance = self.convert_example_to_vw_format(ns_features, true_label)
                        self.vw.push_instance(vw_instance)
                        #self.instance_file.write(vw_instance + u"\n")

                        
                else:

                    #ns_features[
                    true_label = file_reader.get_class(line)
                    vw_instance = self.convert_example_to_vw_format(ns_features, true_label)
                    self.vw.push_instance(vw_instance)
                if self.audit:
                    self.sentence_audit_file.write(file_reader.unparsed_line)
                    self.vw_audit_file.write(vw_instance + "\n")
                processed += 1
                if processed % 10000 == 0:
                    print u"Processed: ",unicode(processed)
                    #elf.vw.push_instance(vw_instance)
                    #self.instance_file.write(vw_instance + u"\n")
        #self.instance_file.close()
        print u"Trained on ", unicode(processed) + u" examples."
        if self.audit:
            self.sentence_audit_file.close()
            self.vw_audit_file.close()

        



# class CrowdflowerTrainer:
#     cpdef base_trainer
#     cpdef crowdflower_filename
#     cpdef feature_extractor
#     def __init__(self, base_trainer,
#                  crowdflower_filename,
#                  feature_extractor):
#         self.base_trainer = base_classifier
#         self.crowdflower_filename = crowdflower_filename
#         self.feature_extractor = feature_extractor
        
    
