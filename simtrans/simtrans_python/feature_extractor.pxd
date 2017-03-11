cdef class FeatureExtractor:
    cpdef sentence_processor
    cpdef int ngram
    cpdef dict get_features(self, list words)
    cpdef ngram_score_model
    cpdef set_use_ngram_scores(self, scorer)
    
cdef class JapaneseSentenceFeatureExtractor(FeatureExtractor):
    cpdef tagger
    cpdef w2v_searcher
    cdef unicode word2vec_filename
    cdef bint use_position
    cdef bint count_case_markers
    cdef list features
    cdef int case_marker_ngrams
    cpdef public bint simplify_sentence
    cpdef bint use_preverb_length
    cdef bint normalize_verb
    cdef bint skip_quotative_verbs
    cdef bint skip_nonquotative_verbs
    cdef bint ignore_preverb
    cdef int token_limit
    cdef int word2vec_features
    

    cpdef mecab_analyze
    cpdef unicode simplify_verb(self, unicode verb, bint normalize=*)
    cpdef dict  get_features(self, list untagged_words)
    cpdef dict get_features_from_tagged(self, list tagged_tokens)
    cpdef list get_pos_tags(self, unicode sentence)
    cpdef bint is_past_tense_sentence(self, list pos_tags)
    cpdef unicode get_next_verb_sequence(self, list tagged_tokens, int start_pos)
    cpdef list get_passive_stems(self, list parse)
    cpdef dict get_case_marked_words(self, list tagged_tokens, bint greedy)
    cpdef unicode get_last_verb_lemma(self, list tagged_tokens)
    cpdef unicode get_final_verb(self, list tagged_tokens)
    cpdef list get_final_verb_chunk_tokens(self, list tagged_tokens)
    cpdef unicode get_final_verb_chunk_string(self, list tagged_tokens)
    cdef unicode identify_case_marker(self, tuple tagged_word)
    cpdef list get_sentence_properties(self, list tagged_tokens)
    cpdef int get_final_verb_index(self, list tagged_tokens)
    cpdef list get_context_before_position(self, int position, list tagged_tokens)
    cpdef bint has_next_case_marker(self, list tagged_tokens, int stat_index)
    cpdef bint is_negative_sentence(self, list tagged_tokens)
    cpdef bint is_past_tense_sentence(self, list tokens_and_tags)
    cpdef int get_next_case_marker_index(self, list tagged_tokens, int start_index)
    cpdef list get_passive_stems(self, list parse)
    cpdef unicode get_final_quotative_verb_sequence(self, list tagged_tokens)
    cpdef get_index_of_start_of_post_verb_stuff(self,list tagged_tokens)
    cpdef unicode get_next_verb_dictionary_form(self, list tagged_tokens, int start_pos)
    cpdef int get_next_verb_index(self, list tagged_tokens, int start_pos)
    cpdef unicode get_sentence_from_parse(self, list tagged_tokens, bint spaces=*)
    cpdef list get_case_marker_sequence(self, list tagged_tokens)
    cpdef list get_clauses(self, list tagged_tokens)
    cpdef list drop_extra_case_marked_words(self, list tagged_tokens)
    cpdef bint is_verb_final(self, list tagged_tokens)
    cpdef float get_case_density(self, list tagged_tokens, int num_bunsetsu)
    cpdef float get_case_density_from_tagged(self, list tagged_tokens, int num_bunsetsu)

cdef class GermanTaggedFeatureExtractor(FeatureExtractor):
    cdef bint use_position
    cpdef bint use_preverb_length
    cdef int token_limit
    cpdef dict get_features(self, list untagged_words)
    cpdef dict get_features_from_tagged(self, list tagged_tokens)
    cpdef int articles_ngrams
    cpdef int tag_ngrams
    # cpdef str name(self)
    # cpdef tuple predict(self, list words)
    # cdef load_verb_indices(self, unicode filename)
    # cpdef unicode convert_features_to_vw_format(self, dict ns_features)
