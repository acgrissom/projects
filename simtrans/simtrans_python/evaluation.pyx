# -*- coding: utf-8 -*-
from sortedcontainers import SortedDict
from numpy import mean
import codecs
import pyximport
pyximport.install()
from class_shrinker import *
from sys import *
from trainers import *
import time
import numpy as np
import matplotlib.pyplot as plt
import collections
from pandas import *
#from ggplot import *

cdef class PredictionEvaluator:
    
    cdef int total_guesses
    cdef int total_correct
    cdef dict vocab_stats 
    #verb->[0:TP,1:FP,2:FN,3:self.total_guesses]

    cdef dict  global_percent_stats
    cdef dict label_distance_stats
    #self.label_distance_stats[distance][verb]=#guesses
    cdef dict label_percent_stats
    #self.label_percent_stats[percent_revealed]=#guesses

    cdef dict global_distance_correct
    cdef dict global_distance_total

    cdef dict global_percent_correct #times label correctly identified
    cdef dict global_percent_total #denominator, #total guesses at x% of sentence


    cpdef public  global_length_correct #correct guesses for sentence length
    cpdef public global_length_total   #total guesses per sentence length
    cpdef public global_case_density_correct
    cpdef public global_case_density_total
    def __init__(self):
        self.total_guesses = 0
        self.total_correct = 0
        self.vocab_stats = {}
        #verb->[0:TP,1:FP,2:FN,3:self.total_guesses]

        self.global_percent_stats = {}
        self.label_distance_stats = {}
        #self.label_distance_stats[distance][verb]=#guesses
        self.label_percent_stats = {}
        #self.label_percent_stats[percent_revealed]=#guesses

        self.global_distance_correct = {}
        self.global_distance_total = {}

        self.global_percent_correct = {} #times label correctly identified
        self.global_percent_total = {} #denominator, #total guesses at x% of sentence

        self.global_length_correct = defaultdict(int)
        self.global_length_total = defaultdict(int)

        self.global_case_density_correct = defaultdict(int)
        self.global_case_density_total = defaultdict(int)


            
    def bin_percent(self, correct_guesses_dict, total_guesses_dict, bin_size):
        #sys.stderr.write("correct_guesses_dict:" +  str(correct_guesses_dict) + "\n")
        #sys.stderr.write("self.total_guesses_dict:" +  str(total_guesses_dict) + "\n")
        cdef list binned_acc = []
        cdef list binned_labels = []
        cdef int min_val
        cdef list keys_in_range
        for min_val in xrange(0,100,bin_size):
            max_val = min_val + bin_size
            binned_labels.append(max_val)
            keys_in_range = [x for x in total_guesses_dict.keys() or x in correct_guesses_dict.keys() if x*100 >= min_val and x*100 <= max_val]
            # sys.stderr.write("min_val:" + str(min_val) + "\n")
            # sys.stderr.write("max_val:" + str(max_val) + "\n")
            # sys.stderr.write("keys_in_range:" + str(keys_in_range) + "\n")
            #print "\nkeys_in_range:",keys_in_range
            correct_guesses_list = [correct_guesses_dict[y] for y in keys_in_range]
            total_guesses_list = [total_guesses_dict[z] for z in keys_in_range]
            #print "correct_guesses_list:",correct_guesses_list
            #print "self.total_guesses_list",self.total_guesses_list
            if sum(total_guesses_list) == 0:
                sys.stderr.write("Error in bin_percent.  This shouldn't be possible. Increase bin size\n")
                binned_acc.append(0)
                #sys.exit()
            else:
                avg = float(sum(correct_guesses_list)) / float(sum(total_guesses_list))
                binned_acc.append(avg * 100.0)
            #print "binned_acc",binned_acc
        return binned_labels, binned_acc


    def bin_no_average(self, correct_guesses_dict, total_guesses_dict, bin_size):
        #sys.stderr.write("correct_guesses_dict:" +  str(correct_guesses_dict) + "\n")
        #sys.stderr.write("self.total_guesses_dict:" +  str(total_guesses_dict) + "\n")
        cdef list binned_acc = []
        cdef list binned_labels = []
        cdef int min_val
        cdef list keys_in_range
        for min_val in xrange(0,100,bin_size):
            max_val = min_val + bin_size
            binned_labels.append(max_val)
            keys_in_range = [x for x in total_guesses_dict.keys() or x in correct_guesses_dict.keys() if x*100 >= min_val and x*100 <= max_val]
            # sys.stderr.write("min_val:" + str(min_val) + "\n")
            # sys.stderr.write("max_val:" + str(max_val) + "\n")
            # sys.stderr.write("keys_in_range:" + str(keys_in_range) + "\n")
            #print "\nkeys_in_range:",keys_in_range
            correct_guesses_list = [correct_guesses_dict[y] for y in keys_in_range]
            total_guesses_list = [total_guesses_dict[z] for z in keys_in_range]
            #print "correct_guesses_list:",correct_guesses_list
            #print "self.total_guesses_list",self.total_guesses_list
            if sum(total_guesses_list) == 0:
                sys.stderr.write("Error in bin_no_average.  This shouldn't be possible. Increase bin size\n")
                binned_acc.append(0)
                sys.exit()
            else:
                acc = []
                if len(correct_guesses_dict.keys()) != len(total_guesses_dict.keys()):
                    sys.err.write("CORRECT GUESSES DOES NOT MATCH TOTAL GUESSES\n")
                    sys.exit(0)
                #avg = float(sum(correct_guesses_list)) / float(sum(total_guesses_list))
                for i in xrange(len(correct_guesses_list)):
                    acc.append(float(correct_guesses_list[i]) / float(total_guesses_list[i]) * 100.0)
            binned_acc.append(acc)
            #print "binned_acc",binned_acc
        return binned_labels, binned_acc




    def get_percent_results(self):
        cdef dict correct_guesses_unsorted_dict = self.global_percent_correct
        total_guesses_unsorted_dict = self.global_percent_total
        correct_guesses_dict = SortedDict()
        total_guesses_dict = SortedDict()
        for key in total_guesses_unsorted_dict.keys():
            correct_guesses_dict[key] = correct_guesses_unsorted_dict[key]
            total_guesses_dict[key] = total_guesses_unsorted_dict[key]

        cdef list correct_guesses_list = [correct_guesses_dict[y] for y in correct_guesses_dict]
        cdef list total_guesses_list = [total_guesses_dict[z] for z in total_guesses_dict]
        cdef list percents = []
        quotient = 0
        for a in xrange(len(total_guesses_list)):
            if total_guesses_list[a] == 0:
                sys.stderr.write("Error in get_percent_results.  This shouldn't be possible.\n")
            else:
                quotient = float(correct_guesses_list[a]) / float(total_guesses_list[a])
            percents.append(quotient)
        return percents

    def output_results(self, outfile_name, bin_size=10, average=False):
        ### First, output sentence length results (Crowdflower)
        length_accuracy_file = codecs.open(outfile_name + '_length_accuracy.csv', 'w', encoding='utf-8')
        length_accuracy_file.write("Length,Accuracy,Count\n")
        for length in sorted(self.global_length_total.keys()):
            out_string = unicode(length) + u","
            out_string += unicode(float(self.global_length_correct[length]) / float(self.global_length_total[length])) + "," + unicode(self.global_length_total[length]) + '\n'
            length_accuracy_file.write(out_string)
        length_accuracy_file.close()

        #Now, output case density
        case_density_all_file =  codecs.open(outfile_name + 'case_density_percent_all.csv', 'w', encoding="utf-8")
        case_density_binned_file =  codecs.open(outfile_name + 'case_density_percent_binned.csv', 'w', encoding="utf-8")
        case_density_all_file.write("Density,Accuracy\n")
        for percent in self.global_case_density_total.keys():
            case_density_all_file.write(unicode(percent) + u"," + unicode(float(self.global_case_density_correct[percent]) / float(self.global_case_density_total[percent])) + '\n')
                                        
        # percents = []
        # case_density_binned_file.write('Density,Accuracy')
        # for percent in sorted(self.global_case_density_total.keys()):
        #     percent = float(self.global_case_density_correct[percent]) / float(self.global_case_density_total[percent])
        #     percents.append(percent)


        # bin_labels, binned = self.bin_percent(self.global_case_density_correct,
        #                                       self.global_case_density_total,
        #                                       bin_size=bin_size)

        # case_density_binned_file.write("Density,Accuracy\n");
        # for x in xrange(len(binned)):
        #     percent = str(float((float(x) + 0.1))  / float(len(binned)))
        #     case_density_binned_file.write(percent + "," + str(binned[x]) + "\n")
        # case_density_binned_file.close()
        # case_density_all_file.close()



            
        
        ###Now, output all points
        percent_all_file =  codecs.open(outfile_name + '_percent_all.csv', 'w', encoding="utf-8")
        percent_all_file.write("Revealed,Accuracy,Count\n");
        for key in self.global_percent_correct.keys():
            percent_all_file.write(str(key) + "," + str(float(self.global_percent_correct[key]) /float( self.global_percent_total[key]) * 100.0) + "," + str(self.global_percent_total[key]) + "\n")
        bin_labels, binned = self.bin_percent(self.global_percent_correct,
                                              self.global_percent_total,
                                              bin_size=bin_size)
        #### Then, output average

        
        distance_correct_file = codecs.open(outfile_name + '_distance.csv','w', encoding="utf-8")
        percent_binned_file =  codecs.open(outfile_name + '_percent_binned.csv', 'w', encoding="utf-8")
        self.global_distance_correct, self.global_distance_total
        self.global_percent_correct, self.global_percent_total
        sys.stderr.write(unicode((u','.join(unicode(x) for x in self.global_distance_correct.keys()))))
        sys.stderr.write("\n")

        percents = []
        for dist in sorted(self.global_distance_correct.keys()):
            percent = float(self.global_distance_correct[dist]) / float(self.global_distance_total[dist])
            percents.append(percent)
        sys.stderr.write(unicode((u','.join(unicode(x) for x in percents))))

        percents = []
        for p in sorted(self.global_percent_correct.keys()):
            acc = float(self.global_percent_correct[p]) / float(self.global_percent_total[p])
            percents.append(acc)
        sys.stderr.write((unicode(u','.join(unicode(x) for x in percents))))
        sys.stderr.write('\n')

        bin_labels, binned = self.bin_percent(self.global_percent_correct, self.global_percent_total, bin_size=bin_size)
        #header = ','.join([str(x * 10) + '-' + str((x + 1) * 10) + '%' for x in range(len(binned) + 1)])
        sys.stderr.write("binned:" + str(binned) + "\n")
        #percent_binned_file.write(header + '\n')
        #percent_binned_file.write( ','.join([str(x) for x in binned]))
        percent_binned_file.write("Revealed,Accuracy\n");
        for x in xrange(len(binned)):
            percent = str(float((x + 1))  / float(len(binned)))
            percent_binned_file.write(percent + "," + str(binned[x]) + "\n")
        percent_binned_file.close()

        distance_correct_file.write(",".join([str(x) for x in range(0,20)]))
        distance_correct_file.write(",".join([str(p) for x in percents]))
        distance_correct_file.close()
        #### Then, output all results binned

        temp_labels, binned_no_avg = self.bin_no_average(self.global_percent_correct, self.global_percent_total, bin_size=5)
        bin_no_avg_file =  codecs.open(outfile_name + '_bin_no_average.csv', 'w', encoding="utf-8")
        bin_no_avg_file.write("Revealed,Accuracy\n")
        for i in xrange(len(binned_no_avg)):
            position = str(i * 100.0 / float(len(binned_no_avg)))
            for acc in binned_no_avg[i]:
                bin_no_avg_file.write(position + "," + str(acc) + "\n")
                
            #percent_binned_file.write
        bin_no_avg_file.close()
        
        return bin_labels, binned


    def count_classes(self, shrinker, reader, max_classes=50):
        shrinker.add_examples_from_file_reader(reader)
        shrinker.set_max_classes(max_classes)
        shrinker.set_exclude_class(u'')
        shrinker.recalculate()
        return shrinker.get_shrunk_class_list()

    def TP(self, label, guess, position, sentence_len, case_density=None):
        # self.total_guesses, self.vocab_stats, self.label_distance_stats, self.label_percent_stats
        # global self.global_distance_total, self.global_distance_correct
        # global self.global_percent_correct, self.global_percent_total
        # global self.total_correct

        self.total_guesses += 1
        self.total_correct += 1

        self.global_length_correct[sentence_len] += 1
        self.global_length_total[sentence_len] += 1

        if case_density is not None:
            self.global_case_density_correct[case_density] += 1
            self.global_case_density_total[case_density] += 1

        
        if not label in self.vocab_stats:
            self.vocab_stats[label] = [0,0,0,0]        
        self.vocab_stats[label][0] += 1
        self.vocab_stats[label][3] += 1

        cdef int distance = sentence_len - position
        cdef float revealed_pcnt = float(position) / float(sentence_len)
        if distance in self.global_distance_correct:
            self.global_distance_correct[distance] += 1
        else:
            self.global_distance_correct[distance] = 1

        if distance in self.global_distance_total:
            self.global_distance_total[distance] += 1
        else:
            self.global_distance_total[distance] = 1

        if revealed_pcnt in self.global_percent_correct:
            self.global_percent_correct[revealed_pcnt] += 1
        else:
            self.global_percent_correct[revealed_pcnt] = 1

        if revealed_pcnt in self.global_percent_total:
            self.global_percent_total[revealed_pcnt] += 1
        else:
            self.global_percent_total[revealed_pcnt] = 1


        if distance not in self.label_distance_stats:
            self.label_distance_stats[distance] = {}
        if label in self.label_distance_stats[distance]:
            self.label_distance_stats[distance][label] += 1
        else:
            self.label_distance_stats[distance][label] = 1

        if revealed_pcnt not in self.label_percent_stats:
            self.label_percent_stats[revealed_pcnt] = {}
        if label in self.label_percent_stats[revealed_pcnt]:
            self.label_percent_stats[revealed_pcnt][label] += 1
        else:
            self.label_percent_stats[revealed_pcnt][label] = 1

    def FP(self, label, guess, position, sentence_len, case_density=None):
        if case_density is not None:
            self.global_length_total[sentence_len] += 1
            self.global_case_density_total[case_density] += 1

        
        self.total_guesses += 1
        cdef int distance = sentence_len - position
        cdef float revealed_pcnt = float(position) / float(sentence_len)
        if label not in self.vocab_stats:
            self.vocab_stats[label] = [0,0,0,0]
        if guess not in self.vocab_stats:
            self.vocab_stats[guess] = [0,0,0,0]
        self.vocab_stats[guess][1] += 1
        self.vocab_stats[guess][3] += 1
        self.vocab_stats[label][2] += 1

        if distance not in self.global_distance_correct:
            self.global_distance_correct[distance] = 0

        if distance in self.global_distance_total:
            self.global_distance_total[distance] += 1
        else:
            self.global_distance_total[distance] = 1

        if revealed_pcnt not in self.global_percent_correct:
            self.global_percent_correct[revealed_pcnt] = 0

        if revealed_pcnt in self.global_percent_total:
            self.global_percent_total[revealed_pcnt] += 1
        else:
            self.global_percent_total[revealed_pcnt] = 1

        if distance not in self.label_distance_stats:
            self.label_distance_stats[distance] = {}
        if guess in self.label_distance_stats[distance]:
            self.label_distance_stats[distance][guess] += 1
        else:
            self.label_distance_stats[distance][guess] = 1

        if revealed_pcnt not in self.label_percent_stats:
            self.label_percent_stats[revealed_pcnt] = {}
        if guess in self.label_percent_stats[revealed_pcnt]:
            self.label_percent_stats[revealed_pcnt][guess] += 1
        else:
             self.label_percent_stats[revealed_pcnt][guess] = 1

    """
    Returns a list of lists of words, incrementally longer
    """
    def get_all_increments(self,list preverb_words):
        cdef list contexts = []
        cdef list increment = []
        cdef unicode word
        for word in preverb_words:
            #word = word.replace(" ","")
            #if len(word) > 0:
            increment.append(word)
            contexts.append(list(increment))
        return contexts

    def test_sentence_incremental(self,list word_list, unicode label, predictor):
        increments = self.get_all_increments(word_list)
        cdef int sentence_len = len(word_list)
        cdef unicode guess
        cdef float score
        for pv in increments:
            #self.total_guesses += 1
            score, guess = predictor.predict(pv)
            #if guess == label:
                #self.total_correct += 1
                #sys.stderr.write("correct guess: " + guess + "\n")
            # else:
            #     sys.stderr.write("incorrect guess: " + guess + "\t::")
            #     sys.stderr.write("should've guessed: " + label + "\n")
            if guess == label:
                self.TP(label, guess, len(pv), sentence_len)
                
            else:
                self.FP(label, guess, len(pv), sentence_len)
        #sys.stderr.write("Overall Acc:" + str(self.total_correct) + "/" + str(self.total_guesses)  + "\n")

    def test_sentence_full_context_only(self, word_list, label, predictor):
        sentence_len = len(word_list)
            #self.total_guesses += 1
        score, guess = predictor.predict(word_list)
            #if guess == label:
                #self.total_correct += 1
                #sys.stderr.write("correct guess: " + guess + "\n")
            # else:
            #     sys.stderr.write("incorrect guess: " + guess + "\t::")
            #     sys.stderr.write("should've guessed: " + label + "\n")
        if guess == label:
            self.TP(label, guess, sentence_len, sentence_len)
        else:
            self.FP(label, guess, sentence_len, sentence_len)


    def test_sentences(self, file_reader, predictor, full_context_only=False):
        last_time = time.localtime()
        correct = 0
        total = 0
        cdef list line
        cdef list preverb
        for line in file_reader:
            label = file_reader.get_class(line)
            #TODO: This returns the verb, too. Do you want this?
            #sentence = file_reader.get_entire_source_text(line)
            preverb = file_reader.get_context_tokens_notag(line)
            #preverb = file_reader.get_features(line)
            #print "preverb", preverb
            #print "label",label
            #guess, prob = predictor.predict(sentence.split())
            if full_context_only:
                self.test_sentence_full_context_only(preverb, label, predictor)
            else:
                self.test_sentence_incremental(preverb, label, predictor) 
            if total % 100 == 0:
                # print "100 sentences in ", time.localtime() - last_time, " seconds"
                # last_time = time.localtime()
                sys.stderr.write("Tested:" + str(total) + "\n")
            total += 1
        sys.stderr.write("Tested " + str(total) + " sentences\n")
        overall = 100.0 * float(self.total_correct) / float(self.total_guesses)
        sys.stderr.write("Overall accuracy:" +  str(overall) + "%")

cdef class CrowdflowerPredictionEvaluator(PredictionEvaluator):
    cpdef public CF_audit_file
    cpdef public last_answer 
    def open_audit_file_hack(self):
        self.CF_audit_file = codecs.open("/Users/alvin/CF_audit.csv", 'w', encoding='utf-8')
        self.CF_audit_file.write("fragment,prediction,label,choices,score,correct,answer_change\n")
        self.last_answer = ""

    cdef test_sentence_incremental(self,list word_list, unicode label, predictor, list choices):
        increments = self.get_all_increments(word_list)
        cdef int sentence_len = len(word_list)
        cdef unicode guess
        cdef float score


        for pv in increments:
            #self.total_guesses += 1
            #TODO(acg) implement using actual CF choices
            case_density = predictor.feature_extractor.get_case_density(pv, len(word_list))
            score, guess = predictor.rank_predictions(pv, choices)[0]
            self.CF_audit_file.write(u" ".join(pv) + u"," + guess + "," + label + u"," + u"[" + u" ".join(choices) + u"]," + unicode(score) + ",")
            if guess.replace(" ",u"") == label.replace(" ", u""):
                self.TP(label, guess, len(pv), sentence_len, case_density)
                self.CF_audit_file.write(u"CORRECT,")
            else:
                self.FP(label, guess, len(pv), sentence_len, case_density)
                self.CF_audit_file.write(u"WRONG,");
            if self.last_answer != guess:
                self.CF_audit_file.write("CHANGED\n")
            else:
                self.CF_audit_file.write("UNCHANGED\n")
            self.CF_audit_file.flush()
            self.last_answer = guess
        #sys.stderr.write("\n")
        #sys.stderr.write("Overall Acc:" + str(self.total_correct) + "/" + str(self.total_guesses)  + "\n")

    def test_sentence_full_context_only(self, word_list, label, predictor, list choices, bunsetsu_len):
        #sentence_len = len(word_list)
            #self.total_guesses += 1
        #TODO(acg) implement using actual CF choices
        score, guess = predictor.rank_predictions(word_list, choices)[0]
        case_density = predictor.feature_extractor.get_case_density(word_list, bunsetsu_len)

        if guess.replace(" ","") == label.replace(" ",""):
            self.TP(label, guess, bunsetsu_len, bunsetsu_len, case_density)
        else:
            self.FP(label, guess, bunsetsu_len, bunsetsu_len, case_density)


    def test_sentences(self, file_reader, predictor, full_context_only=False):
        last_time = time.localtime()
        correct = 0
        total = 0
        cdef list line
        cdef list preverb
        for line in file_reader:
            label = file_reader.get_class(line)
            choices = file_reader.get_choices(line)
            preverb = file_reader.get_context_tokens(line)
            num_bunsetsu = len(file_reader.get_bunsetsu_tokens(line))
            #preverb = file_reader.get_features(line)
            #print "preverb", preverb
            #print "label",label
            #guess, prob = predictor.predict(sentence.split())
            if full_context_only:
                self.test_sentence_full_context_only(preverb, label, predictor, choices, num_bunsetsu)
            else:
                self.test_sentence_incremental(preverb, label, predictor, choices) 
            if total % 100 == 0:
                # print "100 sentences in ", time.localtime() - last_time, " seconds"
                # last_time = time.localtime()
                sys.stderr.write("Tested:" + str(total) + "\n")
            total += 1
        sys.stderr.write("Tested " + str(total) + " sentences\n")
        overall = 100.0 * float(self.total_correct) / float(self.total_guesses)
        sys.stderr.write("Overall accuracy:" +  str(overall) + "%")
    
import sys

def count_classes_train(shrinker, reader, max_classes=50):
    shrinker.add_examples_from_file_reader(reader)
    shrinker.set_max_classes(max_classes)
    shrinker.set_exclude_class(u'')
    shrinker.recalculate()
    return shrinker.get_shrunk_class_list()

import sys
def train_model(model_type,modelname, in_file, max_classes):
    sys.stderr.write("Training " + in_file + "\n")
    sent_extractor = GermanLastVerbExtractor(stem_label=False)
    corpusReader = TaggedGermanFileReader(in_file,
                                          sent_extractor=sent_extractor)
    corpusReader2 = TaggedGermanFileReader(in_file,
                                           sent_extractor=sent_extractor)
    shrinker = ClassShrinker()
    all_classes = count_classes_train(shrinker, corpusReader2, max_classes=max_classes)
    corpusReader.set_valid_classes(all_classes)
    sys.stderr.write(("CLASSES: " + unicode(len(all_classes))))
    sys.stderr.write("\n")
    if model_type == "VWBinary":
        trainer = VWBinaryMultiClassTrainer(modelname, all_classes)
    elif model_type == "VWOAA":
        trainer = VWOAAMultiClassTrainer(modelname, all_classes)
    elif model_type == "SKLSVM":
        trainer = SKLSVMMultiClassTrainer(modelname, all_classes)
    trainer.add_examples_from_file(corpusReader)


def plot_lines(bin_labels, values, filename, label):
    df = DataFrame({
        "% Sentence": bin_labels,
        "Accuracy": values
        })

    #df.y = df.y.cumsum()
    #p = ggplot(aes(x='% Sentence', weight='Accuracy'), data=df)
    #print p + geom_bar()
    plt.show(True)
    #ggsave(p,"test.png")

    current_plot = plt.plot(bin_labels,
                            values,
                            label=label
    )
    #axes = plt.gca()
    #axes.set_ylim(1,100)
    plt.xlabel('Percent Revealed')
    plt.ylabel('Accuracy')
    #plt.legend(current_plot)
    #plt.title('Area of a Circle')
    plt.savefig(filename + u"-scatter.png")
    return current_plot

def plot_histogram(values, filename, label):

    #x #y
    #plt.plot([1,2,3,4], values, 'ro')
    #plt.plot(values, "ro")
    plt.hist(values, color='blue', label=label)
    #plt.axis([0, 6, 0, 20])
    plt.savefig(filename + u".png")
    
