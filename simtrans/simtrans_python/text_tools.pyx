# cython: c_string_type=unicode, c_string_encoding=utf8

__author__ = "Alvin Grissom II"
cdef class SentenceProcessor:
    def __init__(self):
        pass
    """Returns a list of lists of tokens
    """
    cpdef list get_ngram_lists_from_text(self, unicode text, int n):
        return self.get_ngram_lists_from_tokens(text.split(), n)
        
    cpdef list get_ngram_lists_from_tokens(self, list tokens, int n, backoff=True):
        cdef:
            list current_ngram = list()
            list ngrams = list()
            unicode token
        tokens = [u"<S>"] + tokens
        tokens.append(u"</S>")
        if backoff:
            for m in xrange(1,n + 1):
                for i in xrange(len(tokens)):
                    token = tokens[i]
                    if i + m < len(tokens):
                        current_ngram = tokens[i:i + m]
                        ngrams.append(current_ngram)
                    else:
                        break
        else:             
            for i in xrange(len(tokens)):
                token = tokens[i]
                if i + n < len(tokens):
                    current_ngram = tokens[i:i + n]
                    ngrams.append(current_ngram)
                else:
                    break
        return ngrams
        
    cpdef list get_ngram_strings_from_tokens(self, list tokens, int n, backoff=True):
        cdef:
            list ngram_lists =  self.get_ngram_lists_from_tokens(tokens, n, backoff)
            list ngram_strings = list()
        for ngl in ngram_lists:
            ngram_strings.append(u"_".join(ngl))
        return ngram_strings
                         
    cpdef list get_ngram_strings_from_text(self, unicode text, int n):
        cdef:
            list ngram_lists = self.get_ngram_lists_from_text(text, n)
            list ngram_strings = list()
        for ngl in ngram_lists:
            ngram_strings.append(u"_".join(ngl))
        return ngram_strings

    
