# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:25:27 2018

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

#print(filtered_data.iloc[1].values)
#print(filtered_data['CleanedText'].values[10])

#tfidf
#tf_idf_vect= TfidfVectorizer()
tf_idf_vect= TfidfVectorizer(ngram_range=(1,2))
final_tf_idf=tf_idf_vect.fit_transform(filtered_data['CleanedText'].values)
print(final_tf_idf.get_shape())

#get all features/dimentions of tfidf
features=tf_idf_vect.get_feature_names()
print(len(features))
print(features[0:100])
#get tfidf vector of a specofic review
print(final_tf_idf[3,:].toarray()[0])

#get top features for a specific review
def top_tfidf_feats(row,feature,top_n=25):
    topn_ids=np.argsort(row)[::-1][:top_n]
    top_feats=[(features[i],row[i]) for i in topn_ids]
    df=pd.DataFrame(top_feats,columns=['feature','tfidf'])
    return df

top_tfidf=top_tfidf_feats(final_tf_idf[3:5,:].toarray()[0],features)
print(top_tfidf)