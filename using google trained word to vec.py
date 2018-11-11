# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:19:05 2018

@author: prabhudayala
"""

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

#limit parameter limits the total number of words from 3 million to 5000
model=KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin",binary=True,limit=5000)
#model=KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin")
print(model.wv['computer'])
print(model.wv.__getitem__('computer'))
print(model.wv.similarity('men','women'))
print(model.wv.most_similar('female'))
