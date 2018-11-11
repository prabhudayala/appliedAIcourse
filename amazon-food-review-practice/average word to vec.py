# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 00:00:08 2018

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

import pickle
w2v_model = pickle.load(open('word2vec_on_amazon_food_vectors50.sav', 'rb'))
sent_vectors=[]
for sent in list_of_sent:
    sent_vec=np.zeros(50)
    cnt_words=0
    for word in sent:
        try:
            vec=w2v_model.wv[word]
            sent_vec+=vec
            cnt_words+=1
        except:
            pass
    sent_vec/=cnt_words
    sent_vectors.append(sent_vec)

print(len(sent_vectors))
print(len(sent_vectors[0]))
pickle.dump(sent_vectors,open('avg_word2vec_on_amazon_food_vectors50.sav','wb'))