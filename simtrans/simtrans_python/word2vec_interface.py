# -*- coding: utf-8 -*-
from scipy.stats import logistic
import gensim
from gensim.models import Word2Vec

class Word2VecSearcher:
    filename = None
    model = None
    def __init__(self, filename):
        self.filename = filename
        self.model = Word2Vec.load(filename)

    def get_similar_tokens(self,words_list, num):
        query = []
        results = []
        for word in words_list:
            if  word in self.model.vocab:
                query.append(word)
        if len(query) > 0:
            results =  self.model.most_similar(positive=query, topn=num)

        for i in xrange(len(results)):
            results[i] = (results[i][0].replace(':','_'), logistic.cdf(results[i][1]))
                
        return results
