from collections import defaultdict
from filereaders import LabeledDataFileReader
import math
cdef class CorpusFeatures:
    def __init__(self):
        pass
    
    cpdef label_cooccurrence(self,reader):
        cdef int total_label_tokens = 0
        cdef int total_context_tokens = 0
        cdef dict label_counts = <dict>defaultdict(int)
        cdef dict  all_tokens = <dict>defaultdict(dict) 
        #cdef dict valid_classes = reader.get_valid_classes()
        cdef unicode label
        for line in reader:
            label = reader.get_class(line)
            label_counts[label] += 1
            total_label_tokens += 1
            tokens = reader.get_preverb_text(line).split()
            for token in tokens:
                total_context_tokens += 1
                if not label in all_tokens[token]:
                    all_tokens[token] = defaultdict(int)
                all_tokens[token][label] += 1
                all_tokens[token]['__total'] += 1

        cdef float pmi
        for token in all_tokens:
            #print token
            pmi = self.pmi(total_label_tokens,
                    total_context_tokens,
                    all_tokens,
                    label_counts,
                    token,
                    u"")
            if pmi != 0:
                print str(pmi)
    

    cpdef int _get_total_token_occurrences(self, unicode token, dict all_tokens):
        return all_tokens[token]["__total"]
    
    cpdef float pmi(self, int total_label_tokens,
                    int total_context_tokens,
                    dict all_tokens,
                    dict label_counts,
                    unicode token,
                    unicode label):

        #float p_token = math.log(self._get_total_token_occurrences(token,all_tokens)) - math.log(total_context_tokens)
        if(label_counts[label] == 0):
            return 0.0
        if all_tokens[token][label] == 0:
            return 0.0
        if label_counts[label] == 0:
            return 0.0
        cdef float p_label =  math.log(label_counts[label]) - math.log(total_label_tokens)
        cdef float p_token_given_label = math.log(all_tokens[token][label]) - math.log(label_counts[label])
        cdef float pmi = -(p_token_given_label - p_label)
        return pmi
