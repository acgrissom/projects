
# cython: language_level=2
# cython: c_string_type=unicode, c_string_encoding=utf8
__author__ = "Alvin Grissom II"
import MeCab
cdef class MecabInterface:
    cpdef mecab
    def __init__(self):
        self.mecab = MeCab.Tagger()
    cpdef unicode get_base_form(self, unicode surface_tok, list mecab_parse):
        return mecab_parse[6]

    """
    Returns tokenized Japanese string, sans POS tags
    """
    cpdef list tokenize(self, unicode text):
        cdef:
            list mecab_output = self.mecab.parse(text.encode("utf8")).split("\n")
            str word
            unicode u_word
            list words = []
        #print(mecab_output)
        for word in mecab_output:
            words.append(word.split("\t")[0].replace("EOS","").decode("utf8"))
        return words
    # cpdef unicode get_pronunciation(self, list mecab_parse):
    #     return mecab_parse[6]

    cpdef bint is_verb(self, unicode surface_tok, list mecab_parse):
        return mecab_parse[0].startswith(u"動詞")

    cpdef bint is_noun(self, unicode surface_tok, list mecab_parse):
        return mecab_parse[0].strip().startswith(u"名詞")

    cpdef bint is_quotative_particle(self, unicode surface_tok, list mecab_parse):
        return  mecab_parse[2] == u"引用"

    cpdef bint is_negation_morpheme(self, unicode surface_tok, list mecab_parse):
        if mecab_parse[6] == u"ない" or surface_tok == u"ませ":
            #questionable いらっしゃいませ
            return True
        if surface_tok == u"ん" and mecab_parse[0] == u"助動詞":
            return True
        return False

    cpdef bint is_past_tense_morpheme(self, unicode surface_tok, list mecab_parse):
        return surface_tok == u"た" and mecab_parse[0] == u"助動詞"

    cpdef bint is_question_morpheme(self, unicode surface_tok, list mecab_parse):
            return surface_tok == u"か" and mecab_parse[1] == u"副助詞／並立助詞／終助詞"

    # cpdef bint is_passive_morpheme(self, unicode surface_tok, list mecab_parse):
    #     if surface_tok == u"さ":
    #         pass    
    cpdef bint is_verbal_morpheme(self, unicode surface_tok, list mecab_parse):
        return mecab_parse[0].endswith(u"動詞")
               
    
    cpdef bint is_symbol(self, unicode surface_tok, list mecab_parse):
        return mecab_parse[0] == u"記号"

    cpdef bint is_case_marker(self, unicode surface_tok, list mecab_parse):
        return mecab_parse[1] == u"格助詞"

    cpdef bint is_postposition(self, unicode surface_tok, list mecab_parse):
        return mecab_parse[0] == u"助詞"

    cpdef unicode get_dictionary_form(self,list mecab_parse):
        return mecab_parse[6]


    cpdef bint is_te_connector(self, unicode surface_tok, list mecab_parse):
        return surface_tok == u"て" and mecab_parse[0] == u"助詞" and mecab_parse[1] == u"接続助詞"
