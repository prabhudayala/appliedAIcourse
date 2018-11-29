# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 00:45:36 2018

@author: prabhudayala
"""


import sqlite3
import pandas as pd
import nltk
nltk.download('stopwords')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#for bag of words

con=sqlite3.connect('./final.sqlite')

filtered_data=pd.read_sql_query('''
select * 
from Reviews
                                ''',con)
#newDf=filtered_data.loc[filtered_data['Score']=='negative']
#df_to_be_added=newDf.sample(n=249951,replace=True)
#finaldf=pd.concat([filtered_data,df_to_be_added])
#filtered_data=finaldf
filtered_data=filtered_data.sample(n=5000,random_state=42)
print(filtered_data['Time'][1:5])
filtered_data=filtered_data.sort_values('Time')
print(filtered_data.columns)
print(filtered_data['Time'][1:5])
#print(filtered_data.iloc[1].values)
#print(filtered_data['CleanedText'].values[10])
#print(filtered_data['Text'].values[10])

#count_vect=CountVectorizer(ngram_range=(1,2))

#filtered_data=filtered_data.sort_values()


from tqdm import tqdm
import gensim

i=0
list_of_sent=[]
for sent in tqdm(filtered_data['CleanedText'].values):
    list_of_sent.append(sent.decode('utf-8').split())
w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50, workers=4)
w2v_words = list(w2v_model.wv.vocab)


model = TfidfVectorizer()
tf_idf_matrix = model.fit_transform(filtered_data['CleanedText'].values)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
feature_names = model.get_feature_names() # tfidf words/col-names

tfidf_sent_vectors = [];
row=0;
for sent in tqdm(list_of_sent): # for each review/sentence 
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]
            # to reduce the computation we are 
            # dictionary[word] = idf value of word in whole courpus
            # sent.count(word) = tf valeus of word in this review
            tf_idf = dictionary[word]*sent.count(word)
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1
print(len(tfidf_sent_vectors))
print(len(tfidf_sent_vectors[0]))

X_1, X_test, y_1, y_test=train_test_split(tfidf_sent_vectors,filtered_data['Score'],test_size=0.3,random_state=42)
X_train, X_cv, y_train, y_cv=train_test_split(X_1,y_1,test_size=0.3,random_state=42)

from sklearn.naive_bayes import GaussianNB

a=0.2
b=[0.2]
for i in range(10):
    b.append(b[i]+a)

cv_scores=[]

from sklearn.model_selection import cross_val_score
for i in b:
    clf=GaussianNB(var_smoothing=i)
    clf.fit(X_train,y_train)
    scores=cross_val_score(clf,X_train,y_train,cv=10,scoring="accuracy")
    cv_scores.append(scores.mean())
    print("for value of i=%f value is %f "%(i,scores.mean()))
MSE=[1-x for x in cv_scores]
optimal_k=b[MSE.index(min(MSE))]
print("optimal k is %d "%optimal_k)
plt.plot(b,MSE)
plt.show()

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

clf=GaussianNB(var_smoothing=1)
clf.fit(X_train,y_train)
#print(clf.class_prior_ )
#print(len(clf.theta_[0]))
#print(clf.theta_[0].argsort())
x=np.argsort(clf.theta_[0])
y=np.argsort(clf.theta_[1])
number_of_top_features_needed=5
for i in range(number_of_top_features_needed):
    print("top %d positive feature"%i)
    print(feature_names[int(x[i])])
    print("top %d negative feature"%i)
    print(feature_names[int(y[i])])
pred=clf.predict(X_test)
print("scoreeesss")
#print(f1_score(y_test,pred)*float(100))
#print(recall_score(y_test,pred)*float(100))
#print(accuracy_score(y_test,pred,normalize=True)*float(100))
#print(precision_score(y_test,pred)*float(100))
#print(confusion_matrix(y_test,pred)*float(100))
tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
print(tn, fp, fn, tp)