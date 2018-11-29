# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:26:56 2018

@author: prabhudayala
"""

import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

#for bag of words

con=sqlite3.connect('./final.sqlite')

filtered_data=pd.read_sql_query('''
select * 
from Reviews
                                ''',con)

def changelevels(x):
    if x=='positive':
        return 1
    return 0
filtered_data['Score']=filtered_data['Score'].apply(changelevels)

filtered_data=filtered_data.sample(n=2000,random_state=42)
filtered_data=filtered_data.sort_values('Time')


tf_idf_vect= TfidfVectorizer()
final_tf_idf=tf_idf_vect.fit_transform(filtered_data['CleanedText'].values)
feature_names=tf_idf_vect.get_feature_names()
final_tf_idf=final_tf_idf.toarray()
X_1, X_test, y_1, y_test=train_test_split(final_tf_idf,filtered_data['Score'],test_size=0.3,random_state=42)
X_train, X_cv, y_train, y_cv=train_test_split(X_1,y_1,test_size=0.3,random_state=42)
#print(X_train[:,:1])
#print(np.count_nonzero(X_train[:,4:5]))
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(penalty='l2',C=1)
model.fit(X_train,y_train)
x=model.coef_
pred=model.predict(X_test)

p=np.random.normal(loc=0.0,scale=0.1,size=X_train.shape[0])
p=p.reshape(X_train.shape[0],1)
X_train[:,4:5]+=p
#print(np.count_nonzero(X_train[:,4:5]))
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(penalty='l2',C=1)
model.fit(X_train,y_train)
y=model.coef_

print((x-y).max())

print("As i guess there is no colinearity")
print("Top 10 feature importance")

featured=np.argsort(y)
featured=featured[0]
numberoffeaturedneeded=10
featured1=featured[:numberoffeaturedneeded]
#print(featured)
for i in featured1:
    print(feature_names[i])
    print(y[0][i])
    
featured2=featured[-numberoffeaturedneeded:][::-1]
#print(featured)
for i in featured2:
    print(feature_names[i])
    print(y[0][i])
