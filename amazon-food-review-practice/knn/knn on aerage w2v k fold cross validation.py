# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 00:01:47 2018

@author: prabhudayala
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:28:23 2018

@author: prabhudayala
"""

import sqlite3
import pandas as pd
import nltk
import numpy as np
nltk.download('stopwords')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#for bag of words

con=sqlite3.connect('./final.sqlite')

filtered_data=pd.read_sql_query('''
select * 
from Reviews
                                ''',con)

filtered_data=filtered_data.sample(n=5000)
print(filtered_data['Time'][1:5])
filtered_data=filtered_data.sort_values('Time')
print(filtered_data.columns)
print(filtered_data['Time'][1:5])
#print(filtered_data.iloc[1].values)
#print(filtered_data['CleanedText'].values[10])
#print(filtered_data['Text'].values[10])

#count_vect=CountVectorizer(ngram_range=(1,2))

#filtered_data=filtered_data.sort_values()
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

X_1, X_test, y_1, y_test=train_test_split(sent_vectors,filtered_data['Score'],test_size=0.3,random_state=42)
X_train, X_cv, y_train, y_cv=train_test_split(X_1,y_1,test_size=0.3,random_state=42)

#y_test=y_test.values.reshape((y_test.shape[0],1))
#y_train=y_train.values.reshape((y_train.shape[0],1))
#y_cv=y_cv.values.reshape((y_cv.shape[0],1))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
mylist=list(range(0,30))
neighbours=list(filter(lambda x: x %2 !=0, mylist))
cv_scores=[]
for i in neighbours:
    knn=KNeighborsClassifier(n_neighbors=i)
    scores=cross_val_score(knn,X_train,y_train,cv=10,scoring="accuracy")
    cv_scores.append(scores.mean())

print(cv_scores)
MSE=[1-x for x in cv_scores]
optimal_k=neighbours[MSE.index(min(MSE))]
print("optimal k is %d "%optimal_k)
plt.plot(neighbours,MSE)
plt.show()
knn=KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
accracy=accuracy_score(y_test,pred,normalize=True)*float(100)
print("accuracy is %d " %(accracy))
