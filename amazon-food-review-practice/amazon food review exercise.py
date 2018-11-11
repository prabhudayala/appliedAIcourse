# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 13:43:05 2018

@author: prabhudayala
"""

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

con=sqlite3.connect('./database.sqlite')

filtered_data=pd.read_sql_query('''
select * 
from Reviews
where Score!=3
                                ''',con)
#print(filtered_data.head())

def partition(x):
    if x<3:
        return 'negative'
    return 'positive'

actualScore=filtered_data['Score']
positiveNegative=actualScore.map(partition)
filtered_data['Score']=positiveNegative

print(filtered_data.shape)


sorted_data=filtered_data.sort_values('ProductId',axis=0)
final=sorted_data.drop_duplicates(subset={'UserId','ProfileName','Time','Text'})
print(final.shape)

final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
print(final.shape)
print(final['Score'].value_counts())