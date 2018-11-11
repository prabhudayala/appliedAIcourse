# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 00:05:11 2018

@author: prabhudayala
"""

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
tfidf_feat = model.get_feature_names() # tfidf words/col-names


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
