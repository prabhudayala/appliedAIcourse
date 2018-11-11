# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:49:10 2018

@author: prabhudayala
"""

import sqlite3
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#for bag of words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

con=sqlite3.connect('./final.sqlite')

filtered_data=pd.read_sql_query('''
select * 
from Reviews
                                ''',con)

print(filtered_data.iloc[1].values)
print(filtered_data['CleanedText'].values[10])

#bag of words 1gram
#count_vect=CountVectorizer()
#final_counts=count_vect.fit_transform(filtered_data['Text'].values)
#print(final_counts.get_shape())

#bag of words n gram range
#count_vect=CountVectorizer(ngram_range=(1,2))
#final_counts=count_vect.fit_transform(filtered_data['Text'].values)
#print(final_counts.get_shape())


#count_vect=CountVectorizer()
#final_counts=count_vect.fit_transform(filtered_data['CleanedText'].values)
#print(final_counts.get_shape())
#bag of words n gram range
count_vect=CountVectorizer(ngram_range=(1,2))
final_counts=count_vect.fit_transform(filtered_data['CleanedText'].values)
print(final_counts.get_shape())
