# cython: c_string_type=unicode, c_string_encoding=utf8
"""# cython: language_level=3"""
"""# cython: profile=True"""
__author__      = "Alvin Grissom II"

#from __future__ import unicode_literals
import subprocess, sys, errno, codecs, math
import operator
"""
bunny_lure.pyx: This class wraps the Vowpal Wabbit command line interface to enable
classification and retrieving probabilities via function calls.
"""
cdef class ClassifierScorer:
    cpdef public unicode classify_command
    cpdef public unicode model_path
    cpdef public score_command
    cpdef public classify_proc
    cpdef public score_proc
    cpdef public  audit_file
    def __init__(self, unicode model_path,
                 unicode classify_command=None,
                 unicode score_command=None,
                # ngram=0
                ):
        if classify_command is None:
            #self.classify_command = u'vw  -q pv -i ' + model_path + ' -t -p /dev/stdout --binary --ngram 2'
            self.classify_command = u"vw  -i "  + model_path + u" -t -p /dev/stdout "
        else:
            self.classify_command = classify_command
        if score_command is None:
            self.score_command = u"vw -i "  + model_path + " -t -r /dev/stdout "
        else:
            self.score_command = score_command
                
        sys.stderr.write("classifier cmd: " + self.classify_command + "\n")
        sys.stderr.write("score cmd: " + self.score_command + "\n")
        print("Running " + self.classify_command + "\n")
        self.model_path = model_path
        
        self.classify_proc = subprocess.Popen(self.classify_command.split(),
                                              stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE,
                                              close_fds=True,
                                              universal_newlines=True)

        self.score_proc = subprocess.Popen(self.score_command.split(),
                                           stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           close_fds=True,
                                           universal_newlines=True)
        self.audit_file = codecs.open(model_path + '_prediction_audit.txt','w', encoding="utf-8")

        self.classify_proc.stdin.flush()
        self.score_proc.stdin.flush()
        
        self.classify_proc.stdout.flush()
        self.score_proc.stdout.flush()
        
        self.classify_proc.stderr.flush()
        self.score_proc.stderr.flush()

    """
    Returns a class: {-1, 1} for standard logistic regression
    """    
    cpdef int classify(self, unicode vw_string):
        #print "classifying ", vw_string
        cdef str out
        cdef str er
        try:
            #print u"**********CLASSIFYING",vw_string
            self.classify_proc.stdin.write((vw_string + u"\n").encode("utf8"))
            self.classify_proc.stdin.flush()
            self.classify_proc.stdout.flush()
            #print u"******************FLUSHED"
            out = self.classify_proc.stdout.readline().strip()
            #print u"#########out",out
            self.audit_file.write(out + u"\n")
            self.audit_file.flush()
        except:
            sys.stderr.write("Encountered uncaught error in Bunny Lure. Exiting.\n")
            sys.exit()
        #out, err = self.classify_proc.communicate(input=vw_string)     
        cdef int label = int(float(out))
        #For OAA, classes start at 1; for 0/1, must be
        #in {-1, 1}
        #if label == 0:
        #    label = 1
        return label
    cdef float sigmoid(self, float x):
        return 1 / (1 + math.exp(-x))

            
    """
    Should be overridden for different scoring metrics.
    By default, returns probability of logistic classification.
    """
    cpdef float score_prediction(self, unicode vw_string):
        print "in score prediction.  WARNING: Should be overwritten."
        cdef str out
        cdef str err
        cdef float score
        #self.score_proc.stdin.write(to_stdin.encode("utf-8"))
        try:
            self.score_proc.stdin.write((vw_string + u"\n").encode("utf8"))
        except:
            print("Failure in score_prediction.  Exiting.")
            sys.exit(1)
        print "flushing"
        self.score_proc.stdin.flush()
        self.score_proc.stdout.flush()
        print "flushed"
        #stalling here?
        #out_list = self.score_proc.stdout.readline().split()
        #out = out_list[0]
        # unicode ss
        # for ss in out_list:
        #     lst = ss.split(u":")
        #     pred = 
            
        out = "0.0"
        try:
            score = float(out)
            return 1.0
        except:
            sys.stderr.write(u"Caught error in score_predicton\n")
            sys.stderr.write(u"Input: " + vw_string + u"\n")
            sys.stderr.write(u"Output: " + out + u"\n")
            sys.exit(errno)
        return score

    cpdef close(self):
        self.score_proc.stdin.close()
        self.score_proc.stdout.close()
        self.score_proc.stderr.close()
        self.classify_proc.stdout.close()
        self.classify_proc.stdin.close()
        self.classify_proc.stderr.close()
        self.audit_file.close()
        try:
            self.score_proc.kill()
            self.classify_proc.kill()
        except OSError:
            pass



    cpdef unicode classify_label(self, unicode vw_string, int num_classes):
        raise NotImplementedError("TODO:Function not yet implemented.")

       
cdef class OAAScorer(ClassifierScorer):
    def __init__(self, unicode model_path, int num_classes):
        #TODO(acg) For some reason, the isn't setting the instance variables
        score_command= u'vw   -i  ' + model_path + u" -t -r /dev/stdout"
        #score_command = "java Echo"
        #score_command += u' -q pl'
        #score_command += " --oaa " + str(num_classes) + ' '
        classify_command = u"vw -i"  + model_path + u" -t -p /dev/stdout"
        #classify_command += u'--ngram 2'
        #classify_command += u' -q pl '
        
        #classify_command += "--oaa " + str(num_classes)
        super(OAAScorer, self).__init__(model_path,
            classify_command=classify_command,
            score_command=score_command
            )
        sys.stderr.write("classifier cmd: " + self.classify_command + "\n")
        sys.stderr.write("score cmd: " + self.score_command + "\n")

    cpdef list rank(self, unicode vw_string):
        cdef str out
        cdef str err
        cdef float score
        cdef list out_list
        #self.score_proc.stdin.write(to_stdin.encode("utf-8"))
        try:
            self.score_proc.stdin.write((vw_string + u"\n").encode("utf8"))
        except:
            print("Failure in score_prediction.  Exiting.")
            sys.exit(1)
        self.score_proc.stdin.flush()
        self.score_proc.stdout.flush()
        out = self.score_proc.stdout.readline()
        out_list = out.split()
        cdef str ss
        cdef list lst
        cdef int class_num
        cdef list class_scores = list()

        for ss in out_list:
            lst = ss.split(":")
            if len(lst) != 2:
                continue
            class_num = int(lst[0])
            score = float(lst[1])
            score = self.sigmoid(score)
            class_scores.append((score,class_num))
        class_scores = sorted(class_scores, key=operator.itemgetter(0), reverse=True)
        return class_scores


    cpdef int classify(self, unicode vw_string):
        cdef str out
        cdef str er
        try:
            self.classify_proc.stdin.write((vw_string + u"\n").encode("utf8"))
            #self.classify_proc.stdin.write((('%s\n' % vw_string.encode("utf8"))))
            self.classify_proc.stdin.flush()
            #self.classify_proc.stdout.flush()
            out = self.classify_proc.stdout.readline()
            self.audit_file.write(out)
            self.audit_file.flush()
        except:
            sys.stderr.write("Encountered uncaught error in Bunny Lure. Exiting.\n")
            sys.exit()
        #out, err = self.classify_proc.communicate(input=vw_string)     
        cdef int label = int(float(out.strip()))
        #For OAA, classes start at 1; for 0/1, must be
        #in {-1, 1}
        if label == 0:
            label = 1
        return label



"""
Do not use this class for multiclass classification.
"""
cdef class BinaryLogisticScorer(ClassifierScorer):

    def __init__(self, model_path):
        self.classify_command = u'vw  -i ' + model_path + ' -t -p /dev/stdout --binary'
        self.score_command= u'vw  -i ' + model_path + '  -t -r /dev/stdout' 

        super(BinaryLogisticScorer, self).__init__(model_path,
            classify_command=self.classify_command,
            score_command=self.score_command
            )
    cpdef float get_probability(self, unicode vw_string):
        cdef float prob = self.score_prediction(vw_string)
        return prob

    cpdef int classify(self, unicode vw_string):
        cdef str out
        cdef str er
        self.classify_proc.stdin.write((vw_string + u"\n").encode("utf8"))
        self.classify_proc.stdin.flush()
        out = self.classify_proc.stdout.readline()
        #self.audit_file.write(out + u"\n")
        #self.audit_file.flush()
        return int(float(out))

    cpdef float score_prediction(self, unicode vw_string):
        cdef str out
        cdef str err
        cdef float score
        #self.score_proc.stdin.write(to_stdin.encode("utf-8"))
        try:
            self.score_proc.stdin.write((vw_string + u"\n").encode("utf8"))
        except:
            print("Failure in score_prediction.  Exiting.")
            sys.exit(1)
        self.score_proc.stdin.flush()
        self.score_proc.stdout.flush()
        out = self.score_proc.stdout.readline()
        #out = out_list[0]
        # unicode ss
        # for ss in out_list:
        #     lst = ss.split(u":")
        #     pred = 
            
        try:
            score = self.sigmoid(float(out))
        except:
            sys.stderr.write(u"Caught error in score_predicton\n")
            sys.stderr.write(u"Input: " + vw_string + u"\n")
            sys.stderr.write(u"Output: " + out + u"\n")
            sys.exit(errno)
        return score
    cdef float sigmoid(self, float x):
        return 1 / (1 + math.exp(-x))

cdef class CSOAAClassifier(ClassifierScorer):
    cpdef public int num_classes
    cpdef public dict all_labels
    def __init__(self, unicode model_path, unicode classes_file):
        #classes_file is a TSV or classes and the associated number
        self.classify_command = u'vw  -i ' + model_path + ' -t -p /dev/stdout --binary'
        self.score_command= u'vw  -i ' + model_path + '  -t -r /dev/stdout' 
        self.all_labels = dict()
        all_labels_file = open("lats_tsv.txt")
        for line in all_labels_file:
            label_and_index = line.split("\t")
            self.all_labels[int(label_and_index[1])] = label_and_index[0]
        super(CSOAAClassifier, self).__init__(model_path,
                                                   classify_command=self.classify_command,
                                                   score_command=self.score_command
            )
    cdef float sigmoid(self, float x):
        return 1 / (1 + math.exp(-x))


    cpdef list rank(self, unicode vw_string):
        cdef str out
        cdef str err
        cdef float score
        cdef list out_list
        #self.score_proc.stdin.write(to_stdin.encode("utf-8"))
        try:
            self.score_proc.stdin.write((vw_string + u"\n").encode("utf8"))
        except:
            print("Failure in score_prediction.  Exiting.")
            sys.exit(1)
        self.score_proc.stdin.flush()
        self.score_proc.stdout.flush()
        out = self.score_proc.stdout.readline()
        out_list = out.split()
        cdef str ss
        cdef list lst
        cdef int class_num
        cdef list class_scores = list()

        for ss in out_list:
            lst = ss.split(":")
            if len(lst) != 2:
                continue
            class_num = int(lst[0])
            score = float(lst[1])
            score = self.sigmoid(score)
            class_scores.append((score,self.all_labels[class_num]))
        class_scores = sorted(class_scores, key=operator.itemgetter(0), reverse=True)
        return class_scores
        # try:
        #     score = self.sigmoid(float(out))
        # except:
        #     sys.stderr.write(u"Caught error in score_predicton\n")
        #     sys.stderr.write(u"------------------------------------------")
        #     sys.stderr.write(u"Input: " + vw_string + u"\n")
        #     sys.stderr.write(u"------------------------------------------")
        #     sys.stderr.write(u"Output: " + out + u"\n")
        #     sys.exit(errno)

        
    

    
        
