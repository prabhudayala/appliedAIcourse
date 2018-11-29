# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 00:45:36 2018

@author: prabhudayala
"""


import sqlite3
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split

#for bag of words
from sklearn.feature_extraction.text import CountVectorizer


con=sqlite3.connect('./final.sqlite')

filtered_data=pd.read_sql_query('''
select * 
from Reviews
                                ''',con)
#newDf=filtered_data.loc[filtered_data['Score']=='negative']
#df_to_be_added=newDf.sample(n=249951,replace=True)
#finaldf=pd.concat([filtered_data,df_to_be_added])
#filtered_data=finaldf
filtered_data=filtered_data.sample(n=5000)
#count_vect=CountVectorizer(ngram_range=(1,2))

#filtered_data=filtered_data.sort_values()

count_vect=CountVectorizer()
from tqdm import tqdm
final_counts=count_vect.fit_transform(tqdm(filtered_data['CleanedText'].values))
feature_names= count_vect.get_feature_names()
print(final_counts.get_shape())
def changelevels(x):
    if x=='positive':
        return 1
    return 0
filtered_data['Score']=filtered_data['Score'].apply(changelevels)
X_1, X_test, y_1, y_test=train_test_split(final_counts,filtered_data['Score'],test_size=0.3,random_state=42)


X_train, X_cv, y_train, y_cv=train_test_split(X_1,y_1,test_size=0.3,random_state=42)

#y_test=y_test.values.reshape((y_test.shape[0],1))
#y_train=y_train.values.reshape((y_train.shape[0],1))
#y_cv=y_cv.values.reshape((y_cv.shape[0],1))
from sklearn.naive_bayes import GaussianNB

#a=0.2
#b=[0.2]
#for i in range(10):
#    b.append(b[i]+a)
#
#cv_scores=[]
#
#from sklearn.model_selection import cross_val_score
#for i in b:
#    clf=GaussianNB(var_smoothing=i)
#    clf.fit(X_train.toarray(),y_train)
#    scores=cross_val_score(clf,X_train.toarray(),y_train,cv=10,scoring="accuracy")
#    cv_scores.append(scores.mean())
#    print("for value of i=%f value is %f "%(i,scores.mean()))
#MSE=[1-x for x in cv_scores]
#optimal_k=b[MSE.index(min(MSE))]
#print("optimal k is %d "%optimal_k)
#plt.plot(b,MSE)
#plt.show()

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

clf=GaussianNB(var_smoothing=1)
clf.fit(X_train.toarray(),y_train)
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
pred=clf.predict(X_test.toarray())
print("scoreeesss")
#print(f1_score(y_test,pred)*float(100))
#print(recall_score(y_test,pred)*float(100))
#print(accuracy_score(y_test,pred,normalize=True)*float(100))
#print(precision_score(y_test,pred)*float(100))
#print(confusion_matrix(y_test,pred)*float(100))
tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
print(tn, fp, fn, tp)
print(X_test.shape)