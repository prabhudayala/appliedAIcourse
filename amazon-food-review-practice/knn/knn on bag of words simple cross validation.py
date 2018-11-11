# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:28:23 2018

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
from sklearn.model_selection import train_test_split

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

print(filtered_data['Time'][1:5])
filtered_data=filtered_data.sort_values('Time')
print(filtered_data.columns)
print(filtered_data['Time'][1:5])
#print(filtered_data.iloc[1].values)
#print(filtered_data['CleanedText'].values[10])
#print(filtered_data['Text'].values[10])

#count_vect=CountVectorizer(ngram_range=(1,2))

#filtered_data=filtered_data.sort_values()

count_vect=CountVectorizer()
final_counts=count_vect.fit_transform(filtered_data['CleanedText'].values)
print(final_counts.get_shape())

X_1, X_test, y_1, y_test=train_test_split(final_counts,filtered_data['Score'],test_size=0.3,random_state=42)


X_train, X_cv, y_train, y_cv=train_test_split(X_1,y_1,test_size=0.3,random_state=42)

#y_test=y_test.values.reshape((y_test.shape[0],1))
#y_train=y_train.values.reshape((y_train.shape[0],1))
#y_cv=y_cv.values.reshape((y_cv.shape[0],1))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#for i in range(1,30,2):
#    knn=KNeighborsClassifier(n_neighbors=i)
#    knn.fit(X_train,y_train)
#    pred=knn.predict(X_cv)
#    accracy=accuracy_score(y_cv,pred,normalize=True)*float(100)
#    print("accuracy for k= %d  is %d " %(i,accracy))
print("from above i got k value as 5\n so creating a knn of k=5")
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
accracy=accuracy_score(y_test,pred,normalize=True)*float(100)
print("accuracy is %d " %(accracy))

