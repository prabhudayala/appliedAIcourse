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

#filtered_data=filtered_data.sample(n=200000,random_state=42)
filtered_data=filtered_data.sort_values('Time')


tf_idf_vect= TfidfVectorizer()
final_tf_idf=tf_idf_vect.fit_transform(filtered_data['CleanedText'].values)


X_1, X_test, y_1, y_test=train_test_split(final_tf_idf,filtered_data['Score'],test_size=0.3,random_state=42)
X_train, X_cv, y_train, y_cv=train_test_split(X_1,y_1,test_size=0.3,random_state=42)


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(penalty='l2')
model.fit(X_train,y_train)
x=model.coef_
#print(np.count_nonzero(x))
#print(model.coef_)
#print(model.intercept_)
c=[0.001,0.01,0.1,1,10,100,1000]
lambda_values=[]
errors=[]
recall_scores=[]
precision_scores=[]
f1_scores=[]
for i  in c:
    model=LogisticRegression(penalty='l1',C=i,n_jobs=4)
    model.fit(X_train,y_train)
    x=model.coef_
    print(np.count_nonzero(x))
    pred=model.predict(X_test)
    lambda_values.append(1/i)
    errors.append((1-accuracy_score(y_test,pred,normalize=True))*float(100))
    recall_scores.append(recall_score(y_test,pred)*float(100))
    precision_scores.append(precision_score(y_test,pred)*float(100))
    f1_scores.append(f1_score(y_test,pred)*float(100))
for i in range(len(lambda_values)):
    print("for lambda value %f error is %f , recall score is %f , precision score is %f and f1 score is %f "%(lambda_values[i],errors[i],recall_scores[i],precision_scores[i],f1_scores[i]))
    