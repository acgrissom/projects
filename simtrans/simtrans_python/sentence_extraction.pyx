#import pyximport; pyximport.install()
import string, unicodedata, sys
#from filereaders import *
from nltk.stem.snowball import SnowballStemmer
from feature_extractor import JapaneseSentenceFeatureExtractor, GermanTaggedFeatureExtractor

"""
This class is used to extract context and labels from raw (or tagged, depending on the implementation) sentences.
For example, sentences that end in verbs would return the preverb and the verb.
It is used to determine *relevant* sentences (i.e., sentences ending in verbs.)
"""
cdef class SentenceExtractor:
    cpdef int min_sentence_length
    cpdef int max_sentence_length
    cpdef dict exclude_table

    def __init__(self):
    #        exclude_table = dict.fromkeys(i for i in xrange(sys.maxunicode)
    #                                     if unicodedata.category(unichr(i)).startswith('P'))
        cdef unicode char
        self.exclude_table = dict((ord(char), u"") for char in [u":"])
        
    #TODO: alvin should probably be optimized.  This is called on every word.
    cpdef unicode remove_punctuation(self, unicode s):
        return s.translate(self.exclude_table)
            
    cpdef tuple get_context_and_label(self, list all_tokens):
        raise NotImplementedError("Must be subclassed for verb extraction strategy.")

    cpdef bint has_valid_label(self, list all_tokens):
       return self.is_verb_final(all_tokens)

    cpdef bint has_valid_sentence_length(self,list all_tokens):
        return len(all_tokens) > self.min_sentence_length and len(all_tokens) < self.max_sentence_length

    cpdef bint is_valid_sentence(self, all_tokens):
        return self.has_valid_label(all_tokens) and self.valid_sentence_length(all_tokens)

cdef class LastVerbSentenceExtractor(SentenceExtractor):
    def __init__(self):
        super(LastVerbSentenceExtractor, self).__init__()
    cpdef bint is_verb_final(self, list all_tokens):
        raise NotImplementedError("Must be subclassed.")
    cpdef bint has_valid_label(self, list all_tokens):
        return self.is_verb_final(all_tokens)

"""
For reading rew Japanese text (no spaces or tags in data)
Does parsing automatically.
TODO(acg) Lemmatization not implemented.
"""
cdef class JapaneseUnparsedLastVerbSentenceExtractor(SentenceExtractor):
    cpdef bint use_last_verb_chunk
    cpdef public feature_extractor
    def __init__(self,
                  min_sentence_length=2,
                  max_sentence_length=20,
                  use_last_verb_chunk=True,
                  feature_extractor=None
                   ):
        super(JapaneseUnparsedLastVerbSentenceExtractor, self).__init__()
        self.use_last_verb_chunk = use_last_verb_chunk
        self.min_sentence_length = 2
        self.max_sentence_length = 20
        self.feature_extractor = feature_extractor
        # if stem_label == True:
        #     self.stemmer = SnowballStemmer("german")


    """
    Returns a tuple containing (list of untagged preverb tokens, label)
    TODO(acg) You can use get_next_verb w/ the position to speed this up.
    """
    cpdef tuple get_context_and_label_from_string(self, unicode sentence):
       cdef list tagged_tokens = self.feature_extractor.get_pos_tags(sentence)
       return self.get_context_and_label(tagged_tokens)

       
    cpdef tuple get_context_and_label(self, list tagged_tokens):
       cdef list preverb
       cdef unicode verb
       cdef int verb_idx = self.feature_extractor.get_final_verb_index(tagged_tokens)
       preverb = self.feature_extractor.get_context_before_position(verb_idx, tagged_tokens)
       import sys
       if self.use_last_verb_chunk:
           label = self.feature_extractor.get_final_verb_chunk_string(tagged_tokens)
       else:
           label = self.feature_extractor.get_final_verb(tagged_tokens)
       return preverb, label
       # if self.lemmatize_label:
       #    label = self.feature_extractor.get_last_verb_lemma(tagged_tokens)
       # else:
       #     label = self.feature_extractor.get_final_verb(tagged_tokens)
           




    cpdef bint is_verb_final(self, list tagged_tokens):
        return self.feature_extractor.get_final_verb_index(tagged_tokens) > 0


cdef class GermanLastVerbExtractor(LastVerbSentenceExtractor):
    cpdef bint stem_label
    #use Snowball stemmer on verbs
    cpdef bint use_verb_sequence
    #concatenate successive final verbs. When set to false, uses first verb in sequence
    cpdef bint lemmatize_label
    #lemmatize verbs

    cpdef stemmer
    cpdef dict lemma_dict
    
    def __init__(self,
                 bint stem_label=False,
                 bint lemmatize_label=False,
                 min_sentence_length=2,
                 max_sentence_length=20,
                 bint use_verb_sequence=False,
                 dict lemma_dict = None
                  ):
        super(GermanLastVerbExtractor, self).__init__()
        self.stem_label = stem_label
        self.lemmatize_label = lemmatize_label
        self.use_verb_sequence=use_verb_sequence
        self.min_sentence_length = 2
        self.max_sentence_length = 20
        self.lemma_dict = lemma_dict
        if stem_label == True:
            self.stemmer = SnowballStemmer("german")
            
           
        
    cpdef tuple get_context_and_label(self, list tagged_tokens):
        cdef bint keep_tags = True #TODO: make this changeable
        cpdef unicode verb = u""
        cpdef list tok
        cdef list preverb = list()
        cdef unicode t
        if not self.is_verb_final(tagged_tokens):
            return (tagged_tokens, u"")
            
        cdef int i = len(tagged_tokens) - 1
        while i > 0:
            tagged_tokens[i]
            try:
                tok = tagged_tokens[i].split(u"_")
            except UnicodeDecodeError:
                sys.stderr.write("Skipping token: " + unicode(tok) + u"\n")
                i -= 1
                continue

            if len(tok) <= 1:
                i -=1
                continue
            elif tok[1].startswith(u"$"):
                i -= 1
                continue
            elif self.is_verb(tok[1]):
                if self.stem_label:
                    tok[0] = self.stemmer.stem(tok[0])
                elif self.lemmatize_label:
                        if tok[0] in self.lemma_dict:
                            tok[0] = self.lemma_dict[tok[0]]
                if self.use_verb_sequence:
                    verb = tok[0] + u" " + verb
                else:
                    verb = tok[0]
            else:
                break
            i -= 1
        for i in xrange(0, i + 1):
            try:
                tok = tagged_tokens[i].split(u"_")
            except UnicodeDecodeError:
                sys.stderr.write("Skipping token: " + str(tok))
                continue
            if len(tok) <= 1:
                i -= 1
                continue    
            elif tok[1].startswith(u"$"):
                continue
            else:
                if keep_tags:
                    t = tagged_tokens[i]
                    t = self.remove_punctuation(t)
                    preverb.append(t)
                else:
                    tok[0] = self.remove_punctuation(tok[0])
                    preverb.append(tok[0])
        return (preverb,verb.strip())

    cpdef get_last_verb_first_index(self, list tagged_tokens):
        cdef:
            int i
            int j
        while i > 0:
            tok = tagged_tokens[i].split(u"_")
            if tok[1].startswith(u"$"):
                i -= 1
            else:
                j = i;
                while self.is_verb(tok[1]):
                    tok = tagged_tokens[j].split(u"_")
                    j -=1
                return j + 1

        return i



    cpdef bint is_verb_final(self, list tagged_tokens):
        cdef int i = len(tagged_tokens) - 1
        while i > 0:
            tok = tagged_tokens[i].split(u"_")
            if tok[1].startswith(u"$"):
                i -= 1
            elif self.is_verb(tok[1]):
                return True
            else:
                break
        return False

    cpdef bint is_verb(self, unicode tag):
        return tag.startswith(u"V") or tag.startswith(u"v")




"""
Extracts sentences with verbs from a list from corpus
"""
cdef class JapaneseMultiChoiceSentenceExtractor(SentenceExtractor):
    #dictionary where keys are the surface forms of verbs from CSV
    cpdef dict valid_surface_verbs
    #dictionary where keys are the lemmas (which lemmas?)
    cpdef dict valid_lemmas
    cpdef verb_sentence_extractor

    
    # cpdef set_valid_verbs(self,
    #                       unicode crowdflower_csv_filename,
    #                       verb_sentence_extractor):
    #     #reader = JapaneseCrowdflowerReader(u"CF_parsed_sents_choices.csv")
    #     for line_list in reader:
    #         for choice in reader.get_choices(line_list):
    #             self.valid_surface_verbs[choice] = 1
                

    cpdef bint has_valid_label(self, list all_tokens):
        pass
