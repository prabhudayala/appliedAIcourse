# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:54:39 2018

@author: prabhudayala
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle

con=sqlite3.connect('./final.sqlite')

filtered_data=pd.read_sql_query('''
select * 
from Reviews''',con)

#print(filtered_data.iloc[1].values)
#print(filtered_data['CleanedText'].values[10])
import re
def cleanhtml(sentence): #function to clean the word of any html-tags\n",
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters\n",
    cleaned = re.sub(r'[?|!|\'|\"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\\|/]',r' ',cleaned)
    return  cleaned


import gensim
from sklearn.externals import joblib
i=0
list_of_sent=[]
for sent in filtered_data['Text'].values:
    filtered_sentense=[]
    sent=cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if(cleaned_words.isalpha()):
                filtered_sentense.append(cleaned_words.lower())
            else:
                continue
    list_of_sent.append(filtered_sentense)
print(len(list_of_sent))
print(filtered_data['Text'].values[0])
print(list_of_sent[0])

w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50,workers=4)
pickle.dump(w2v_model, open('word2vec_on_amazon_food_vectors50.sav', 'wb'))
words=list(w2v_model.wv.vocab)
print(len(words))
print(w2v_model.wv.most_similar('tasty'))