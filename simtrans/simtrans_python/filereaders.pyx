# cython: c_string_type=unicode, c_string_encoding=utf8, language_level=3
# -*- coding: utf-8 -*-o
#import pyximport; pyximport.install()
from sentence_extraction import *
#TODO(acg) using cimport causes pyximport build errors
from feature_extractor import JapaneseSentenceFeatureExtractor
import string
import codecs
import unicodecsv 
import mecab_interface
cpdef inline list utf_encode(list csv_line):
     cpdef int index
     cdef str col
     cdef unicode u_col
     for index, col in enumerate(csv_line):
        u_col = unicode(col, errors="ignore", encoding="utf-8")
        #u_col = col.encode("utf-8", errors="ignore")
        csv_line[index] = u_col
         #unicode(col,"UTF-8")
     return csv_line

"""
This class defines an iterarable over files with various attributes.
It must be subclassed.
It would have been abstract, but abstract classes are impossible in Cython
"""
cdef class FileReader:
    cpdef int line
    cpdef public unicode filename
    cdef public unicode unparsed_line

    cpdef set_filename(self, unicode filename):
        self.filename = filename
    
    def __init__(self, unicode filename):
        self.set_filename(filename)

    def __iter__(self):
        raise NotImplementedError("__iter__ should be subclassed")

    cpdef reset(self):
        self.__init__(self.filename)

    cpdef list parse_line(self, line):
        raise NotImplementedError("Subclass to parse raw text of next line.")

    #Cython requires __next__, not next()
    # def next(self):
    #     

    def __next__(self):
        raise NotImplementedError("next() should be subclassed.")

cdef class PlainUnicodeFileReader(FileReader):
    cpdef pyreader

    def __init__(self, unicode filename):
        self.filename = filename
        if self.pyreader is not None:
            self.pyreader.close()
        self.pyreader = codecs.open(filename, 'r', encoding='utf-8', errors='ignore')

    cpdef list parse_line(self, line):
        return line.split()
    
    def __iter__(self):
        return self

    def __next__(self):
        self.unparsed_line = self.pyreader.next()
        return self.parse_line(self.unparsed_line)
    
cdef class LabeledDataFileReader(FileReader):
    cpdef dict valid_classes, column_indexes
    cpdef set_valid_classes(self, list classes):
        for label in classes:
            self.valid_classes[label] = 1

    """Assumes that the labels file is in the same directory as the model,
    with the same name"""
    cpdef set_valid_classes_from_model_file(self,unicode model_filename):
        cdef unicode label_filename = model_filename.replace(u".model",".labels")
        label_file = codecs.open(label_filename, 'r', encoding='utf-8', errors='ignore')
        for line in label_file:
            label = line.split(u"\t")[0]
            self.valid_classes[label] = 1

    cpdef set_valid_classes_from_label_file(self, unicode label_filename):
        label_file = codecs.open(label_filename, 'r', encoding='utf-8', errors='ignore')
        for line in label_file:
            label = line.split(u"\t")[0]
            self.valid_classes[label] = 1

    
    """
    Returns labels not being skipped.
    """
    cpdef dict get_valid_classes(self):
        return self.valid_classes
            
    """
    Returns the class of the current line.
    """
    #@abstractmethod
    cpdef unicode get_class(self,list line_list):
        raise NotImplementedError()

    """
    Returns the features on the pased line.
    """
    #@abstractmethod
    cpdef dict get_features(self, list line_list):
        raise NotImplementedError()
    
    cpdef list get_context_tokens(self, list line_list):
        raise NotImplementedError()


"""
A file iterator for CF CSV files.
"""
cdef class JapaneseCrowdflowerReader(LabeledDataFileReader):
    cdef list header
    cdef readonly pos_csv_file
    cdef int line_num
    cdef list current_line
    cdef mecab_analyze
    def __init__(self, unicode filename):
        self.line_num = 0
        self.valid_classes = dict()
        self.column_indexes = dict()
        self.set_filename(filename)
        self.pos_csv_file = unicodecsv.reader(codecs.open(filename,'r',encoding='utf-8',errors='ignore'))
        self.header = self.next()
        self.mecab_analyze = mecab_interface.MecabInterface()
        cdef int i = 0
        #creates a dctionary mapping from column names to indices
        cdef str col
        for col in self.header:
            self.column_indexes[col] = i
            i += 1

    cpdef list get_context_tokens(self, list line_list):
       toks = self.mecab_analyze.tokenize(line_list[1].replace(u",",u""))
       cdef list new_toks = []
       for tok in toks:
           if len(tok.replace(" ","")) > 0:
               new_toks.append(tok.replace(" ", ""))
       return new_toks

    cpdef list get_bunsetsu_tokens(self, line_list):
        return line_list[1].split(u",")

    #TODO(acg) implement case density test
    # cpdef list count_case_markers(self, line_list):
    #     cdef list tokens =  self.mecab_tokenize(self.get_context_tokens(self, line_list))
    #     for tok in tokens:
    #         pass
    #     return 0

    cpdef list parse_line(self, line):
        return utf_encode(line)

    def __next__(self):
        if self.line_num == 0:
            self.line_num = self.line_num + 1
            self.current_line = self.pos_csv_file.next()
            return self.current_line
        if self.current_line is None:
            return None
        cdef list next_line = self.parse_line(self.pos_csv_file.next())
        while not next_line[0].startswith('test'):
            next_line = self.pos_csv_file.next()
            if next_line is None:
                return None
            next_line  = self.parse_line(self.pos_csv_file.next())
        #unparsed line not set here b/c it uses CSV reader

        current_line = next_line
        if next_line is None:
            return None
        self.line_num += 1
        #return next_line
        return utf_encode(next_line)

    def __iter__(self):
        return self
    
    cpdef list get_choices(self, list line_list, tokenize=True):
        cdef list choices = list()
        if tokenize:
            choices.append(u" ".join(self.mecab_analyze.tokenize((line_list[3].replace(u"。",u"")))))
            choices.append(u" ".join(self.mecab_analyze.tokenize((line_list[4].replace(u"。",u"")))))
            choices.append(u" ".join(self.mecab_analyze.tokenize((line_list[5].replace(u"。",u"")))))
            choices.append(u" ".join(self.mecab_analyze.tokenize((line_list[6].replace(u"。",u"")))))
        else:
            choices.append(line_list[3].replace(u"。",u""))
            choices.append(line_list[4].replace(u"。",u""))
            choices.append(line_list[5].replace(u"。",u""))
            choices.append(line_list[6].replace(u"。",u""))
        return choices
   
    cpdef unicode get_class(self, list line_list):
        cdef int correct_idx = int(line_list[int(self.column_indexes['correct answer'])][1])
        #s1, s2, ...
        return line_list[2 + correct_idx].replace(u"。",u"")

        
    

cdef class POSCSVFileReader(LabeledDataFileReader):
    cdef list header
    cdef readonly pos_csv_file
    cdef int class_column
    cdef unicode fold
    def __next__(self):
        if self.line == 0:
            self.line = self.line + 1
            return self.pos_csv_file.next()
        cdef list next_line = self.pos_csv_file.next()
        if len(self.valid_classes) != 0:
            while self.get_class(next_line) not in self.valid_classes.keys():
                next_line  = self.pos_csv_file.next()
                self.unparsed_line = u" ".join(next_line) #untested
                self.line += 1    
                if next_line is None:
                    return None
        return next_line
        #return utf_encode(next_line)

    cpdef unicode get_fold(self, list line_list):
        return unicode(line_list[6])

    cpdef unicode get_class(self, list line_list):
        return <unicode>line_list[self.class_column]

    """
    Returns a dictionary with the features.  The keys are namespaces; the values are features.
    """
    cpdef dict get_features(self, list line_list):
        cdef dict ns_features = dict()
        ns_features['preverb_unigrams'] = list(self.get_preverb_text(line_list).split())
        #ns_features['verb'] = list(self.get_class(line_list))
        return ns_features

    """Returns unicode string of entire source text, including verb."""
    cpdef unicode get_entire_source_text(self, list line_list):
        #return unicode(line_list[2],"UTF-8")
        return unicode(line_list[2])

    cpdef unicode get_target_text(self, list line_list):
        return unicode(line_list[1])

    cpdef unicode get_tagged_source_text(self, list line_list):
        return unicode(line_list[3])

    cpdef unicode get_preverb_text(self, list line_list):
        return unicode(line_list[4])

    cpdef unicode get_verb(self, list line_list):
        return unicode(line_list[5])

    cpdef list get_context_tokens(self, list line_list):
        return self.get_preverb_text(line_list).split()

    cpdef unicode get_id(self, list line_list):
        return unicode(line_list[0])

    """
    Sets the column to be considered as the class to classify.
    """
    def set_class_column(self, unicode column_name):
       self.class_column = self.column_indexes[column_name]

    """
    Sets filename and skips the header of pos.csv file.
    """
    def __init__(self, unicode filename):
        self.line = 0
        self.valid_classes = dict()
        self.column_indexes = dict()
        self.set_filename(filename)
        self.pos_csv_file = unicodecsv.reader(open(filename,'rb'))
        #self.pos_csv_file = unicodecsv.reader(codecs.open(filename,'r'))
        self.header = self.next()
        cdef int i = 0
        #creates a dctionary mapping from column names to indices
        #cdef str col
        for col in self.header:
            self.column_indexes[col] = i
            i += 1
        self.set_class_column('verb')

    def __iter__(self):
        return self

cdef class TaggedGermanFileReader(LabeledDataFileReader):
    cpdef filereader
    cpdef sent_extractor #determines which sentences to use (e.g., verb-final)
    cpdef feat_extractor #extracts features from context words
    cdef table

    def __init__(self, filename,
                 sent_extractor=GermanLastVerbExtractor(),
                 feat_extractor = GermanTaggedFeatureExtractor()):
        if self.filereader is not None:
            self.filereader.close()
        self.filereader = codecs.open(filename, "r",encoding="utf-8", errors='ignore')
        self.valid_classes = dict()
        self.sent_extractor = sent_extractor
        self.feat_extractor = feat_extractor
        #self.table = string.maketrans("","")
        self.filename = filename

    cpdef reset(self):
        self.__init__(self.filename,
                      self.sent_extractor,
                      self.feat_extractor)

    """
    Returns the correct label of the current line.
    """

    cpdef unicode get_class(self, list line_list):      
        return self.sent_extractor.get_context_and_label(line_list)[1]
    """
    Returns the features on the pased line.
    Strips punctuation
    # In this case, n-grams, probably
    This is confusing, because it requires a whole line.
    # """
    cpdef dict get_features(self, list line_list):
        unigrams_nopunc = self.get_context_tokens(line_list)
        ns_features = self.feat_extractor.get_features_from_tagged(unigrams_nopunc)
        return ns_features
        
    cpdef list get_context_tokens_notag(self, list line_list):
        cdef list preverb = []
        #stop before the first part of last verb sequence
        cdef unicode x
        for x in self.get_context_tokens(line_list):
            
            tok = x.split(u"_")
            if len(tok) > 1:
                if not tok[1].startswith(u"$"):
                    preverb.append(tok[0])
        return preverb
                
    cpdef list get_context_tokens(self, list line_list):
        return self.sent_extractor.get_context_and_label(line_list)[0]

    def __next__(self):
        #cdef list next_line = utf_encode(self.filereader.next().split())
        cdef list next_line = self.parse_line(self.filereader.next())
        self.line += 1
        if len(self.valid_classes) != 0:
            while self.get_class(next_line) not in self.valid_classes.keys() and (self.fold != self.get_fold(next_line)):
                self.unparsed_line = self.filereader.next()
                next_line  = self.parse_line(self.unparsed_line)
                #next_line  = utf_encode(self.filereader.next().split())
                self.line += 1
                if next_line is None:
                    return None
        return next_line

    cpdef list parse_line(self, line):
        return line.split()

    def __iter__(self):
        return self    

cdef class UntaggedJapaneseFileReader(LabeledDataFileReader):
    cpdef filereader
    cpdef sent_extractor #determines which sentences to use (e.g., verb-final)
    cpdef feat_extractor #extracts features from context words
    cdef table
    cdef bint delete_spaces

    def __init__(self, unicode filename,
                 sent_extractor=JapaneseUnparsedLastVerbSentenceExtractor()):
        self.filename = filename
        if self.filereader is not None:
            self.filereader.close()
        self.filereader = codecs.open(filename, "r",encoding="utf-8", errors='ignore')
        self.valid_classes = dict()
        self.sent_extractor = sent_extractor 
        self.feat_extractor = sent_extractor.feature_extractor
        #self.table = string.maketrans("","")

    cpdef reset(self):
        self.__init__(self.filename,
                      sent_extractor = self.sent_extractor)


        
    """
    Returns the correct label of the current line.
    """
    cpdef unicode get_class(self, list line_list):
        return self.sent_extractor.get_context_and_label(line_list)[1]
    """
    Returns the features on the pased line.
    Strips punctuation
    # In this case, n-grams, probably
    This is confusing, because it requires a whole line.
    # """
    cpdef dict get_features(self, list line_list):
        cdef int verb_idx = self.feat_extractor.get_final_verb_index(line_list)
        cdef list preverb_words = []
        if verb_idx - 1 > 0:
            preverb_words = line_list[:verb_idx-1]
        ns_features = self.feat_extractor.get_features_from_tagged(preverb_words)
        return ns_features
        
    cpdef list get_context_tokens_notag(self, list line_list):
        #this is a list, but since it's not spaced for Japanese, there should only be one element
        #raise NotImplemented
        cdef tuple context_and_class = self.sent_extractor.get_context_and_label(line_list)
        if context_and_class is not None and context_and_class[0] is not None:
            return context_and_class[0]
        return []

                
    cpdef list get_context_tokens(self, list line_list):
        return self.sent_extractor.get_context_and_label(line_list)[0]

    cpdef list parse_line(self, line):
       return self.feat_extractor.get_pos_tags(line.replace(":",""))

    def __next__(self):
        #cdef list next_line = utf_encode(self.filereader.next().split())
        cdef list next_line = self.parse_line(u"".join(self.filereader.next()))
        #print "next line",next_line
        if len(self.valid_classes) != 0:
            while self.get_class(next_line) not in self.valid_classes.keys():
                self.unparsed_line = self.filereader.next()
                next_line = self.feat_extractor.get_pos_tags(self.unparsed_line.replace(":",""))
                #self.filereader.next().split()
                #next_line  = utf_encode(self.filereader.next().split())
                self.line += 1
                if next_line is None:
                    return None
        return next_line

    def __iter__(self):
        return self    

   
import os, random
"""
Class (untested) for reading contexts from multiple files
"""

cdef class MultiClassJapaneseCorpusReader(FileReader):
    cpdef dict labels #maps lables to file descriptors
    cpdef bint shuffle
    def __init__(unicode path, bint shuffle = True):
        self.file_paths = dict()
        self.shuffle = shuffle
        paths = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]
        for path in paths:
            path_toks = path.split("/")
            label = path_tokens[len(path_toks) - 1].replace("context.text","").replace("。","")
            sys.stderr.write("Opening file for label " + label + "\n")
            self.labels[label] = codecs.open(path, mode="r", codec="utf-8", errors='ignore')
        sys.stderr.write("Opened " + len(file_paths) + " files\n.")
      
    def __next__(self):
        #cdef list next_line = utf_encode(self.filereader.next().split())
        labels_keys = [x for x in self.labels.keys()]
        cdef int rn
        if len(labels) == 0:
            return None
        if shuffle == True:
            rn = rand.random.randint(0,len(labels_keys))
            label = labels_keys[rn]
            next_line = self.labels[label].next()
            while next_line is None:
                del self.labels[label]
                del labels_keys[rn]
                rn = rand.random.randint(0,len(labels_keys))
                label = labels_keys[rn]
                next_line = self.labels[label].next()
        else:
            label = labels_keys[0]
            next_line = self.labels[label].next()
            while next_line is None:
                del self.labels[label]
                del labels_keys[0]
                next_line = labels_keys[0]
        if next_line is None:
            return None
        return next_line


    #class_name.context.text

cdef class ParallelCorpusReader:
    cpdef source_reader
    cpdef target_reader
    cpdef public list line_list_source
    cpdef public list line_list_target

    def __init__(self, source_reader, target_reader):
        self.source_reader = source_reader
        self.target_reader = target_reader

    def iter(self):
        return self

    def __next__(self):
        cdef unicode next_source = self.source_reader.next()
        cdef unicode next_target = self.target_reeader.next()
        self.line_list_source = self.left_reader.parse_line(next_source)
        self.line_list_target = self.right_reader.parse_line(next_target)
        return (self.line_list_source, self.line_list_target)

    cpdef source_context_tokens(self, tuple line_lists):
        return self.source_reader.get_context_tokens(line_lists[0])

    cpdef list target_context_tokens(self, tuple  line_lists):
       return self.right_reader.get_context_tokens(line_lists[1])

    cpdef unicode source_text(self, tuple line_lists):
        return u" ".join(line_lists[0])

    cpdef unicode target_text(self, tuple line_lists):
        return u" ".join(line_lists[1])

    cpdef unicode target_label(self, tuple line_lists):
       return self.right_reader.get_class(line_lists[1])

    cpdef unicode source_label(self, tuple line_lists):
       return self.left_reader.get_class(line_lists[0])


        
cdef class CdecParallelCorpusReader:
    cpdef filereader
    cpdef public left_reader
    cpdef public right_reader
    cpdef public list line_list_lhs
    cpdef public list line_list_rhs
    
    def __init__(self, filename, left_reader, right_reader):
        self.left_reader = left_reader
        self.right_reader = right_reader
        self.filereader = codecs.open(filename, "r",encoding="utf-8", errors='ignore')

    def __iter__(self):
        return self

    def __next__(self):
        cdef unicode  next_line = self.filereader.next()
        if next_line is None:
            return
        cdef list split_line = next_line.split(u"|||")
        self.line_list_lhs = self.left_reader.parse_line(split_line[0])
        self.line_list_rhs = self.right_reader.parse_line(split_line[1])
        return (self.line_list_lhs, self.line_list_rhs)

    cpdef left_context_tokens(self, tuple line_lists):
        return self.left_reader.get_context_tokens(line_lists[0])

    cpdef list right_context_tokens(self, tuple  line_lists):
       return self.right_reader.get_context_tokens(line_lists[1])

    cpdef unicode left_text(self, tuple line_lists):
        return u" ".join(line_lists[0])

    cpdef unicode right_text(self, tuple line_lists):
        return u" ".join(line_lists[1])

    cpdef unicode right_label(self, tuple line_lists):
       return self.right_reader.get_class(line_lists[1])

    cpdef unicode left_label(self, tuple line_lists):
       return self.left_reader.get_class(line_lists[0])


