# cython: c_string_type=unicode, c_string_encoding=utf8
__author__ = "Alvin Grissom II"
import sys
from mecab_interface import MecabInterface
import MeCab
import text_tools
import kenlm
from prediction import KenLMLanguageModelVerbScorer
from word2vec_interface import Word2VecSearcher


#import pyximport; pyximport.install()

cdef class FeatureExtractor:
    cpdef set_use_ngram_scores(self,scorer):
        self.ngram_score_model = scorer
        
    
    cpdef dict get_features(self, list words):
        raise NotImplementedError("Must be subclassed")

"""
TODO(acg): Distinguish between arimasen, ja arimasen, etc.
"""
cdef class JapaneseSentenceFeatureExtractor(FeatureExtractor):
        # public static final String POS_POSTPOSITION = "助詞"; //じょし
        # public static final String POS_POSTPOSITION_ADVERBALIZE = POS_POSTPOSITION + ",副詞化";
        # public static final String POS_POSTPOSITION_CASEMARK = POS_POSTPOSITION + ",格助詞";
        # public static final String POS_POSTPOSITION_QUOTATIVE = POS_POSTPOSITION_CASEMARK + ",引用";
        # public static final String POS_VERB = "動詞";
        # public static final String POS_VERB_VERBSTEM = POS_VERB + ",自立";
        # public static final String POS_VERB_VERBSUFFIX = POS_VERB + ",接尾"; //せつび
        # public static final String POS_PUNCTUATION = "記号";
        # public static final String POS_PUNCTUATION_PERIOD = POS_PUNCTUATION + ",句点";
        # public static final String POS_PUNCTUATION_COMMA = POS_PUNCTUATION + ",読点"; //とうてん
        # public static final String POS_NOUN = "名詞";
        # public static final String POS_NOUN_SURUNOUN = POS_NOUN + ",サ変接続"; //さへんせつぞく
        # public static final String POS_VERB_AUX = "助動詞";
        # public static final String POS_VERB_DEPENDENT = POS_VERB + ",非自立";

    def __init__(self,
                 bint use_position=False,
                 bint use_preverb_length=False,
                 int ngram=2,
                 bint count_case_markers=False,
                 int case_marker_ngrams=0,
                 bint simplify_sentence=False,
                 bint normalize_verb=False,
                 bint skip_quotative_verbs=False,
                 bint skip_nonquotative_verbs=False,
                 bint ignore_preverb=False,
                 int w2v_features=0,
                 unicode w2v_filename=None,
                 int token_limit=-1,
                 ngram_scorer=None
             ):
        #TODO(acg) quotative skipping is a hack
        self.features = list()
        self.mecab_analyze = MecabInterface()
        self.tagger = MeCab.Tagger();
        self.ngram = ngram
        self.simplify_sentence=simplify_sentence
        self.use_position = use_position
        self.use_preverb_length = use_preverb_length
        self.count_case_markers = count_case_markers
        self.case_marker_ngrams = case_marker_ngrams
        self.ignore_preverb = ignore_preverb
        self.sentence_processor = text_tools.SentenceProcessor()
        self.normalize_verb = normalize_verb
        print "normalize...",self.normalize_verb
        self.skip_quotative_verbs = skip_quotative_verbs
        self.skip_nonquotative_verbs = skip_nonquotative_verbs
        self.token_limit = token_limit
        self.set_use_ngram_scores(ngram_scorer)
        self.word2vec_features = w2v_features
        self.word2vec_filename = w2v_filename
        if self.word2vec_features > 0:
            self.w2v_searcher = Word2VecSearcher(self.word2vec_filename)

        #The following isn't actually used.
        copulas =["そう","です","し","でし","した","た",
         "ます","ません","でしょ","う","だろ","ん","の"]
    cpdef unicode get_sentence_from_parse(self, list tagged_tokens, bint spaces=False):
        cdef unicode sentence = u""
        cdef unicode delim = u""
        if spaces:
            delim = u" "
        for tt in tagged_tokens:
            sentence += tt[0] + u" "
        return sentence.strip()


    cpdef float get_case_density(self, list words, int num_bunsetsu):
        cdef list tagged_tokens = self.get_pos_tags(u"".join(words))
        return self.get_case_density_from_tagged(tagged_tokens, num_bunsetsu)
    
    cpdef float get_case_density_from_tagged(self, list tagged_tokens, int num_bunsetsu):
        cdef float case_markers = 0.0
        for token, long_tag in tagged_tokens:
            if self.mecab_analyze.is_postposition(token, long_tag):
                if token in [u"が", u"を", u"から",u"に"]:
                    case_markers += 1
        return case_markers / float(num_bunsetsu)
    
    cpdef list drop_extra_case_marked_words(self, list tagged_tokens):
        cdef dict particles = dict() #particle -> (marked word, tag)
        cdef dict particle_words = dict()
        cdef unicode token
        cdef list simplified_sent = list()
        cdef list particle_order = list()
        cdef dict particle_indexes = dict()
        cdef unicode last_token = u""
        cdef list last_tag = list()
        cdef unicode p = u""
        cdef list long_tag
        cdef int i = 0
        for token, long_tag in tagged_tokens:
            if self.mecab_analyze.is_postposition(token, long_tag) and not token == u"の":
                if token in particle_indexes.keys():
                    del particle_order[particle_indexes[token]]
                    i = len(particle_order)
                particle_indexes[token] = i
                particle_order.append(token)
                particle_words[token] = (last_token, last_tag)
                particles[token] = (token,last_tag)
                i = len(particle_order)
            last_tag = long_tag
            last_token = token
        for p in particle_order:
            simplified_sent.append(particle_words[p])
            simplified_sent.append(particles[p])
        # for w in simplified_sent:
        #     print " ".join(w[0])
        return simplified_sent
    
    """
    If it's a suru verb, return the noun preceding it.
    The lemma version of this function should use this function
    """

    cpdef int get_final_verb_index(self, list tagged_tokens):
        cdef unicode tagged_word
        cdef unicode token
        cdef list long_tag
        cdef int i = len(tagged_tokens) -1
        #i =  self.get_index_of_start_of_post_verb_stuff(tagged_tokens) - 1
        #get first index of the last sequence of verbal tokens
        for token, long_tag in reversed(tagged_tokens):
            if i > 0:
                if self.mecab_analyze.get_base_form(token,long_tag) == u"する" or \
                   self.mecab_analyze.get_base_form(token,long_tag) == u"さ":
                    if self.mecab_analyze.is_noun(tagged_tokens[i-1][0], tagged_tokens[i-1][1]):
                        return i - 1
                    if self.mecab_analyze.is_verbal_morpheme(tagged_tokens[i-1][0], tagged_tokens[i-1][1]):
                        i -= 1
                    elif tagged_tokens[i-1][0] == u"を" and self.mecab_analyze.is_noun(tagged_tokens[i-2][0], tagged_tokens[i-2][1]):
                        return i - 2
                    elif tagged_tokens[i-1][0] == u"を" and self.mecab_analyze.is_verb(tagged_tokens[i-2][0], tagged_tokens[i-2][1]):
                        i -= 2
                    elif tagged_tokens[i-1][0] == u"と": #to suru 
                        return i - 1
                    elif tagged_tokens[i-1][0] == u"に": #ni suru
                        return i - 1
                    elif tagged_tokens[i-1][0] == u"たり": #ni suru
                        return i - 1
                    elif self.mecab_analyze.is_postposition(tagged_tokens[i-1][0], tagged_tokens[i-1][1]):
                        i -= 2
                    else:
                        #print self.get_sentence_from_parse(tagged_tokens)
                        return i
                if self.mecab_analyze.is_verbal_morpheme(tagged_tokens[i][0], tagged_tokens[i][1]):
                    while self.mecab_analyze.is_verbal_morpheme(tagged_tokens[i][0], tagged_tokens[i][1]) and i > 0:
                        if i - 2 > 0:
                            if self.mecab_analyze.is_te_connector(tagged_tokens[i-1][0], tagged_tokens[i-1][1]):
                                if self.mecab_analyze.is_verbal_morpheme(tagged_tokens[i-2][0], tagged_tokens[i-2][1]):
                                    i = i - 2
                                    continue
                        i -= 1
                    if self.mecab_analyze.is_verb(tagged_tokens[i+1][0], tagged_tokens[i+1][1]):    
                        return i + 1
                
            i -= 1
        
        # cdef int index = len(tagged_tokens)
        # for i in reversed(xrange(len(tagged_tokens) - 1)):
        #     if self.mecab_analyze.is_case_marker(tagged_tokens[i][0],tagged_tokens[i][1]):
        #         return i + 1
        #find last case
        
                
        return -1

    cpdef bint is_verb_final(self, list tagged_tokens):
        print "in is_verb_final"
        cdef int final_verb_idx = self.get_final_verb_index(tagged_tokens)
        cdef unicode token
        cdef list long_tag
        cdef int i
        if final_verb_idx < 0:
            return False
        if self.mecab_analyze.is_noun(tagged_tokens[final_verb_idx][0],tagged_tokens[final_verb_idx][1]):
            if len(tagged_tokens) == final_verb_idx + 1:
                #no suru verb
                return False
            elif self.mecab_analyze.get_dictionary_form(final_verb_idx+1,tagged_tokens) == u"する" or self.mecab_analyze.get_normalized_form(final_verb_idx+1, tagged_tokens) == u"さ":
                return True
            
        elif not self.mecab_analyze.is_verb(tagged_tokens[final_verb_idx][0],tagged_tokens[final_verb_idx][1]):
            return False

        if self.skip_quotative_verbs:
            if self.mecab_analyze.is_quotative_particle(tagged_tokens[final_verb_idx-1][0],tagged_tokens[final_verb_idx-1][1]):
                return False
        elif self.skip_nonquotative_verbs:
            if not self.mecab_analyze.is_quotative_particle(tagged_tokens[final_verb_idx-1][0],tagged_tokens[final_verb_idx-1][1]):
                return False
                
        for i in xrange(final_verb_idx, len(tagged_tokens)):
            token = tagged_tokens[i][0]
            long_tag = tagged_tokens[i][1]
            if not self.mecab_analyze.is_verbal_morpheme(token, long_tag):
                return False
        return True
                                                         
                                                         
        
    cpdef unicode get_final_quotative_verb_sequence(self, list tagged_tokens):
        cdef unicode tagged_word
        cdef unicode token
        cdef list long_tag
        cdef int i = 0
        #get first index of the last sequence of verbal tokens
        for token, long_tag in reversed(tagged_tokens):
            if self.mecab_analyze.is_quotative_particle(tagged_tokens[i][0], tagged_tokens[i-1][1]):
                return self.get_next_verb_dictionary_form(tagged_tokens, i)
            i +=1
        return None

    
    cpdef unicode get_final_verb(self, list tagged_tokens):
        cdef int idx
        cdef unicode final_verb = None
        idx = self.get_final_verb_index(tagged_tokens)
        if idx < 0:
            return None
        final_verb = tagged_tokens[idx][0]
        if idx < len(tagged_tokens) - 1:
            if final_verb == u"と":
                return u"とする"
            elif final_verb == u"に":
                return u"にする"
        if self.normalize_verb:
            #print "idx",idx
            #print "len",len(tagged_tokens)
            #print "len---",len(tagged_tokens[idx])
            final_verb = self.mecab_analyze.get_dictionary_form(tagged_tokens[idx][1])
        return final_verb

    cpdef unicode simplify_verb(self, unicode verb, bint normalize=True):
        #print "verb",verb
        cdef list tagged_verb = self.get_pos_tags(verb)
        cdef unicode token = tagged_verb[0][0]
        cdef list tag = tagged_verb[0][1]
        cdef int i = 0
        if self.mecab_analyze.is_noun(token, tag):
            return token
        if self.mecab_analyze.is_postposition(token,tag) and token == u"に":
            return u"にする"
        elif self.mecab_analyze.is_postposition(token,tag) and token == u"に":
            return u"とする"
        for token, tag in tagged_verb:
            if not self.mecab_analyze.is_verbal_morpheme(token, tag):
                i += 1
        if i == len(tagged_verb):
            i -= 1        
        if normalize:
            return self.mecab_analyze.get_normalized_form(i,tagged_verb)
        return tagged_verb[i][0]
            
            

    """
    TODO(acg): use other verb function for consistency
    this is returning a base form, not a lemma
    """
    cpdef unicode get_last_verb_lemma(self, list tagged_tokens):
        cdef unicode tagged_word
        cdef unicode token, long_tag
        cdef int i = len(tagged_tokens)
        for token, long_tag in reversed(tagged_tokens):
            if long_tag.startswith(u"動詞"):
                if i > 0:
                    if self.mecab_analyze.get_base_form(token) == u"する":
                        if self.mecab_analyze.is_noun(tagged_tokens[i - 1]):
                            return self.mecab_analyze.get_base_form(tagged_tokens[i - 1])
                    if self.mecab_analyze.get_base_form(token) == u"する":
                        if self.mecab_analyze.is_noun(tagged_tokens[i - 1]):
                            return tagged_tokens[i - 1]
                return token
            i -= 1
        return None

    cpdef dict get_features(self, list words):
        return self.get_features_from_tagged(self.get_pos_tags(u"".join(words)))

    cpdef dict get_features_from_tagged(self, list tagged_tokens):
        cdef dict ns_features = dict()
        cdef list unigrams = list()
        cdef list position_words = list()
        cdef unicode token
        cdef list temp_list
        cdef unicode temp_unicode
        cdef dict case_marked_dict = {}
        cdef list case_counts
        cdef list long_tag
        cdef int i = 0
        cdef dict ngram_sscore_dict = dict()
        cdef unicode ngram
        cdef unicode verb
        if self.token_limit > 0 and len(tagged_tokens) > self.token_limit:
            tagged_tokens = tagged_tokens[-self.token_limit:]
        if self.simplify_sentence:
            tagged_tokens = self.drop_extra_case_marked_words(tagged_tokens)
        for token, long_tag in tagged_tokens:
            unigrams.append(token)
            position_words.append(token + ":" + str(i))
            i += 1
        if not self.ignore_preverb:
            ns_features[u"preverb"] = unigrams
        if self.use_position:
            ns_features[u"location"] = position_words
        if self.use_preverb_length:
            ns_features["preverb"].append("^^length:"+ unicode(len(unigrams)))
        if self.ngram > 1:
            ngrams_list = self.sentence_processor.get_ngram_strings_from_tokens(unigrams,self.ngram)
            if self.ngram_score_model is not None:
                ns_features["z_ngram_scores"] = list()
                for ngram in ngrams_list:
                    ngram_score_dict = self.scorer.get_context_score_for_all_verbs(ngram)
                    for verb in ngram_score_dict:
                        ns_features["z_ngram_scores"].append(verb
                                                         + u"^"
                                                         + ngram + u":"
                                                         + ngram_score_dict[verb])

                #ngrams_list = [x + u":" + self.ngram_score_model.score(x) for x in ngrams_list]
            ns_features[unicode(str(self.ngram),"utf-8") + u"grams"] = ngrams_list
        if self.count_case_markers:
             case_marked_dict =  self.get_case_marked_words(tagged_tokens, True)
             case_counts = list()
        if self.count_case_markers:
             for temp_unicode, temp_list in case_marked_dict.iteritems():
                 case_counts.append(temp_unicode + ":" + unicode(len(temp_list)))
             ns_features[u"case_count"] = case_counts
        cdef unicode last_case_marker
        if self.case_marker_ngrams > 0:
            case_unigrams = self.get_case_marker_sequence(tagged_tokens)
            if len(case_unigrams) > 0:
                last_case_marker = case_unigrams[len(case_unigrams) - 1]
                ns_features[u"preverb"].append("^^last_case^" + last_case_marker)
            ngrams_list = self.sentence_processor.get_ngram_strings_from_tokens(case_unigrams,self.case_marker_ngrams)
            ns_features[u"ic"] = ngrams_list
        cdef w2v_tokens = list()
        if self.word2vec_features > 0:
            w2v_tokens = self.w2v_searcher.get_similar_tokens(unigrams, self.word2vec_features)
            ns_features['w2v'] = []            
            for f, v in w2v_tokens:
                ns_features['w2v'].append(f + u":" + unicode(v))

        return ns_features

    """
    Gets words marked by case markers. greedy=true gets multiple words
    TODO(acg) currently ignored
    """
    cpdef dict get_case_marked_words(self, list tagged_tokens, bint greedy):
        cdef unicode tagged_word
        cdef unicode token
        cdef list long_tag
        cdef unicode case_marker
        cdef dict case_marked = {}
        case_marked[u"NOM"] = []
        case_marked[u"ACC"] = []
        case_marked[u"DAT"] = []
        case_marked[u"TOP"] = []
        case_marked[u"DIR"] = []
        #case_type->tok
        cdef int i = self.get_next_case_marker_index(tagged_tokens, 0)
        if len(tagged_tokens) > 1:
            while i != -1 and i > 0 :
                case_marker = tagged_tokens[i][0]
                token = tagged_tokens[i-1][0]
                long_tag = tagged_tokens[i-1][1]
                if case_marker == u"は":
                    case_marked[u"TOP"].append(token)
                elif case_marker == u"が":
                    #print case_marked[u"NOM"]
                    case_marked[u"NOM"].append(token)
                elif case_marker == u"を":
                    case_marked[u"ACC"].append(token)
                elif case_marker == u"に":
                    #this could be directional, too
                    case_marked[u"DAT"].append(token)
                elif case_marker == u"へ":
                    case_marked[u"DIR"].append(token)
                else:
                    pass
                # elif token == u"でも":
                #     pass
                # elif token == u"のに":
                #     pass
                # elif token == u"":
                #     pass
                i = self.get_next_case_marker_index(tagged_tokens, i + 1)
        return case_marked

    cpdef bint has_next_case_marker(self, list tagged_tokens, int start_index):
        return self.get_index_of_next_case_marker(tagged_tokens, start_index) != -1
    
    """
    Approximates the function of the case marker.  This needs some more work
    to disambiguate.  For VW namespaces, the first letter needs to be unique
    """
    cdef unicode identify_case_marker(self, tuple tagged_word):
        cdef unicode token, long_tag
        token, long_tag = tagged_word
    
        if long_tag.startswith(u"助詞"):
                if token == u"は":
                    return u"TOP"
                elif token == u"が":
                    #could be "but"
                    return u"NOM"
                elif token == u"を":
                    return u"ACC"
                elif token == u"に":
                    return u"DAT"
                    #ambiguous
                elif token == u"へ":
                    return u"DIR"
                    #not a real case
                elif token == u"でも" or token == u"ても":
                    #This is an NPI
                    return u"BUT"
                elif token == u"のに":
                    return u"BUT"
                elif token == u"ば" \
                    or token == u"ら" \
                    or token == u"なら":                    
                    return u"COND"
                elif token == u"から":
                    return u"FROM"
                else:
                    return u"OTHER"
        else:
            return u""

    cpdef int get_next_case_marker_index(self, list tagged_tokens, int start_index):
        cdef unicode word
        cdef list long_tag
        cdef int i = -1
        if start_index >= 0 and start_index < len(tagged_tokens):
            i = start_index
            for word, long_tag in tagged_tokens[start_index:]:
                if self.mecab_analyze.is_case_marker(word, long_tag):
                    return i
                i += 1
        if i >= len(tagged_tokens) -1:
            return -1
        return i

    """
     Returns a list of the case markers in the sentence as they appear
    """
    cpdef list get_case_marker_sequence(self, list tagged_tokens):
        cdef list case_markers = list()
        cdef int i = self.get_next_case_marker_index(tagged_tokens, 0)
        if len(tagged_tokens) > 1:
            while i != -1 and i > 0 :
                case_markers.append(tagged_tokens[i][0])
                i = self.get_next_case_marker_index(tagged_tokens, i + 1)
        return case_markers
    
                        
    """
     Takes as input a tuple of the token and its long tag.
     These are generated in get_case_marked...()
    """        
    cpdef list get_pos_tags(self, unicode sentence):
        cdef list tags = self.tagger.parse(sentence.encode("utf8")).split("\n")
        #print (tags)
        cdef list u_tags = []
        cdef str word_tag
        cdef unicode u_tag
        cdef unicode token, long_tag
        cdef tuple token_and_tag
        for word_tag in tags:
            if word_tag == "EOS":
                break
            u_tag = unicode(word_tag,"utf-8", errors="ignore")
            #tag = word_tag.split("\t")[1]
            token, long_tag = u_tag.split(u"\t")
            token_and_tag = (token, long_tag.split(","))
            if self.mecab_analyze.is_symbol(token_and_tag[0], token_and_tag[1]):
                continue
            u_tags.append(token_and_tag)
        return u_tags
    #TODO(acg) unwritten  Maybe in mecab_interface?
    cpdef bint is_past_tense_sentence(self, list tagged_tokens):
        cdef unicode token = tagged_tokens[len(tagged_tokens)-1][0]
        cdef list parse = tagged_tokens[len(tagged_tokens)-1][1]
        return self.mecab_analyze.is_negation_morpheme(token, parse)

    cpdef unicode get_next_verb_sequence(self, list tagged_tokens, int start_pos):
    #TODO(acg) can call get_next_verb_index()
        cdef int end  = len(tagged_tokens) - 1
        cdef unicode verb = u""
        cdef int i
        cdef int j
        for i in xrange(start_pos, end):
            if tagged_tokens[i][1][0].startswith(u"動詞"):
                j = i
                while j <= end and tagged_tokens[j][1][0].startswith(u"動詞"):
                    verb += tagged_tokens[j][0]
                    j += 1
                return verb
        return None

    cpdef unicode get_next_verb_dictionary_form(self, list tagged_tokens, int start_pos):
        cdef int index = self.get_next_verb_index(tagged_tokens, start_pos)
        if index >= 0:
            return self.mecab_analyze.get_dictionary_form(tagged_tokens[index][0],tagged_tokens[index][1])
        return None

    cpdef int get_next_verb_index(self, list tagged_tokens, int start_pos):
        cdef int end  = len(tagged_tokens) - 1
        cdef unicode verb = u""
        cdef int i
        cdef int j
        for i in xrange(start_pos, end):
            if tagged_tokens[i][1][0].startswith(u"動詞"):
                return i
        return -1

    cpdef list get_context_before_position(self, int position, list tagged_tokens):
        cdef unicode tok
        cdef list tag
        cdef list preverb = [] #called preverb, but could be anything before position
        if len(tagged_tokens) == 0 or position <= 0:
            return None
        for tok, tag in tagged_tokens[0:position]:
            preverb.append(tok)
        return preverb


    cpdef list get_final_verb_chunk_tokens(self, list tagged_tokens):
        cdef int verb_idx = self.get_final_verb_index(tagged_tokens)
        if verb_idx < 0:
            return []
        cdef list verb_tokens = tagged_tokens[verb_idx:]
        return verb_tokens
        

    cpdef unicode get_final_verb_chunk_string(self, list tagged_tokens):
        cdef unicode tok
        cdef list tag
        cdef unicode chunk = u""
        if len(tagged_tokens) == 0:
            return None
        
        for tok, tag in self.get_final_verb_chunk_tokens(tagged_tokens):
            chunk += u" " + tok
        chunk = chunk.strip()
        return chunk
            


    """This function tries to guess where things that come after the final
    verb stem.  For example, vacuous copulas and expalnatory
    particles.

    """
    cpdef get_index_of_start_of_post_verb_stuff(self,
                                                list tagged_tokens):
        cdef:
            int i
            unicode tok
            list tag
            unicode last_tok
        for i in reversed(xrange(len(tagged_tokens))):
            tok, tag = tagged_tokens[i]
            if tok == u"だ" or tok == u"だっ":
                last_tok = tok
                continue
            if self.mecab_analyze.is_question_morpheme(tok, tag):
                last_tok = tok
                continue
            if i - 2 > 0:
               if tok == u"ある" or tok == u"あっ":
                    if tagged_tokens[i-1][0] == "で":
                        if self.mecab_analyze.is_verbal_morpheme(tagged_tokens[i-2][0]):
                            return len(tagged_tokens)
                        else:
                            i -= 1
               if self.mecab_analyze.is_verbal_morpheme(tagged_tokens[i-1][0], tagged_tokens[i-1][1]):
                    if tok == u"です":
                        last_tok = u"です"
                        continue
                    if tok == u"た":
                        if i - 2 > 0:
                            if tagged_tokens[i-1][0]== u"でし":
                                if self.mecab_analyze.is_verbal_morpheme(tagged_tokens[i-2][0], tagged_tokens[i-2][1]):
                                    continue
            if tok == u"う" and tag[4] == u"不変化型":
                last_tok = u"う"
                continue
            if tok == u"でしょ" or tok == u"だろ":
                last_tok = tok
                continue
            if tok == u"の" or tok == u"ん":
                if tag[0] == u"名詞":
                    last_tok = tok
                    continue
            if tok == u"た" and tag[0] == u"助動詞":
                last_tok = tok
                continue
            return i
    # cpdef list strip_postverb_copulas(self, list tagged_tokens):
    #     raise NotImplemented
    #     """
    #     で　す
    #     で　し　た
    #     である
    #     """
    
            
    cpdef bint is_negative_sentence(self, list tagged_tokens):
        """
        Possible representations:
        ない で ある
        せんでした
        なかった（です・でしょ）
        ない（です・でしょ）
        なく（て）
         """
        cdef int post_verb_start = self.get_index_of_start_of_post_verb_stuff(tagged_tokens)

        #print "last", tagged_tokens[post_verb_start-1][0]
        #if post_verb_start - 1 >= 0:
        return self.mecab_analyze.is_negation_morpheme(tagged_tokens[post_verb_start][0], tagged_tokens[post_verb_start][1])

       
    cpdef list get_sentence_properties(self, list tagged_tokens):
        pass
        
    cpdef list get_passive_stems(self, list parse):
        cdef:
            list passive_verbs = list()
            unicode current_tok
            unicode next_tok
            list properties
            list tok_parse
            list next_tok_parse
            int i = 0
        for i in xrange(len(parse) - 1):
            current_tok = parse[i]
            next_tok = parse[i + 1]
            tok_parse = current_tok.split()
            next_tok_parse = next_tok.split()
            properties = tok_parse[1].split(u",")
            if properties[0] == u"動詞" and next_tok_parse[0] == u"れ":
                if next_tok_parse[1].startswith( u"動詞"):
                    passive_verbs.append(tok_parse[0])
        return passive_verbs
    """
    Returns a list of lists of tokens, where the lists of tokens are separated by clause.
    """
    cpdef list get_clauses(self, list tagged_tokens):
        cdef unicode token
        cdef list long_tag
        cdef unicode this_clause = u""
        cdef list clauses = list()
        cdef unicode next_token = None
        cdef int i
        for i in xrange(len(tagged_tokens)):
            token = tagged_tokens[i][0]
            long_tag = tagged_tokens[i][1]
            if i < len(tagged_tokens) - 1:
                next_token = tagged_tokens[i+1][0]
            this_clause += u" " + token
            if self.mecab_analyze.is_conjunction(token, long_tag):
#                if token != u"て" and token != u"で":
                if this_clause.strip() != u"":
                    clauses.append(this_clause)
                    this_clause = u""
            if self.mecab_analyze.is_verbal_morpheme(token, long_tag):
                if next_token == u"、":
                    clauses.append(this_clause)
                    this_clause = u""
            next_token = None
        if this_clause.strip() != u"":
            clauses.append(this_clause)
        return clauses
        

        

cdef class GermanTaggedFeatureExtractor(FeatureExtractor):
    def __init__(self,
                 bint use_position=True,
                 bint use_preverb_length=False,
                 int ngram=2,
                 ngram_scorer=None,
                 int articles_ngrams=-1,
                 int tag_ngrams=-1):
        self.use_position = use_position
        self.ngram = ngram
        self.sentence_processor = text_tools.SentenceProcessor()
        self.use_preverb_length = use_preverb_length
        self.set_use_ngram_scores(ngram_scorer)
        self.articles_ngrams = articles_ngrams
        self.tag_ngrams = tag_ngrams
        
    """
   TODO(acg) This function might tagging internally.
    """
    # cpdef dict get_features(self, list untagged_words):
    #     return get_features_from_tagged(untagged_words)
        
    cpdef dict get_features(self, list untagged_words):
        return self.get_features_from_tagged(untagged_words)
        

    """
    This should only be used on the preverb.  Otherwise,
    it will return the verb along with it.
    """
    cpdef dict get_features_from_tagged(self, list tagged_tokens):
        cdef dict ns_features = dict()
        cdef list unigrams_nopunc = list()
        cdef list position_words = list()
        cdef unicode x
        cdef float i = 1.0
        cdef int tok_idx = 0
        cdef unicode temp = u""
        cdef list tok
        cdef unicode word
        cdef unicode tag = u""
        cdef list ngrams_list
        cdef list case__counts
        cdef unicode temp_unicode
        cdef list temp_list
        cdef list case_unigrams
        cdef dict case_marked_dict
        cdef dict ngram_score_dict = dict()
        cdef unicode verb 
        cdef list articles = []
        cdef list pos_sequence = []
        if self.token_limit > 0 and len(tagged_tokens) > self.token_limit:
            tagged_tokens = tagged_tokens[:-self.token_limit]
        for tok_idx in xrange(len(tagged_tokens)):
            x = tagged_tokens[tok_idx]
            tok = x.split(u"_")
            word = tok[0]
            if len(tok) > 1:
                tag = tok[1].upper()
            else:
                tag = u"UNK"
            if not tag.startswith(u"$") and word.find(u":") == -1:
                unigrams_nopunc.append(word)
                pos_sequence.append(tag)
                if self.use_position:
                    temp = word + u":" + unicode(str(i),"utf-8")
                    position_words.append(temp)
            if tag == "ART":
                articles.append(word)
            i += 1.0
        ns_features[u"preverb"] = unigrams_nopunc
        if self.use_preverb_length:
            ns_features["preverb"].append("^^length:"+ unicode(len(unigrams_nopunc)))
            
        if self.use_position:
            #ns_features[u"location"] = position_words
            ns_features[u"location"] = u"loc:" + unicode(len(unigrams_nopunc))
        if self.ngram > 1:
            ngrams_list = self.sentence_processor.get_ngram_strings_from_tokens(unigrams_nopunc,
                                                                                self.ngram,
                                                                                backoff=True)

            if self.ngram_score_model is not None:
                ns_features["z_ngram_scores"] = list()
                for ngram in ngrams_list:
                    ngram_score_dict = self.ngram_score_model.get_context_score_for_all_verbs(ngram.replace(u"_"," "))
                    for verb in ngram_score_dict:
                        ns_features["z_ngram_scores"].append(verb
                                                         + u"^"
                                                         + ngram + u":"
                                                         + unicode(ngram_score_dict[verb]))
                #ngrams_list = [x + u":" + self.ngram_prob.score(x) for x in ngrams_list]
            ns_features[unicode(str(self.ngram),"utf-8") + u"grams"] = ngrams_list
        cdef unicode article
        if self.tag_ngrams > 0:
            ns_features["tags"] = self.sentence_processor.get_ngram_strings_from_tokens(pos_sequence,
                                                                                        self.tag_ngrams,
                                                                                        backoff=True)
            if len(pos_sequence) > 1:
                ns_features["tags"] += "^last^" + pos_sequence[len(pos_sequence)-1]
                
        if self.articles_ngrams > 0:
            ns_features["articles"]  = self.sentence_processor.get_ngram_strings_from_tokens(articles,
                                                                                        self.articles_ngrams,
                                                                                        backoff=True)
            if len(articles) > 0:
                ns_features["articles"].append(u"^^last^" + articles[len(articles)-1])
       
       
            
            #print ns_features["z_ngram_scores"]
        return ns_features        
