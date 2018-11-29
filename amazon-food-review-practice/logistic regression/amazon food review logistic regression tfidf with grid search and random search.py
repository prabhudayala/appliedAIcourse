# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:26:56 2018

@author: prabhudayala
"""

import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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
filtered_data=filtered_data.sample(n=5000,random_state=42)
filtered_data=filtered_data.sort_values('Time')


tf_idf_vect= TfidfVectorizer()
final_tf_idf=tf_idf_vect.fit_transform(filtered_data['CleanedText'].values)


X_1, X_test, y_1, y_test=train_test_split(final_tf_idf,filtered_data['Score'],test_size=0.3,random_state=42)
X_train, X_cv, y_train, y_cv=train_test_split(X_1,y_1,test_size=0.3,random_state=42)


print("\n\n\n***********Grid search CV**************")
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
tuned_parameters=[{'C': [10**-4,10**-2,1.0,10**2,10**4]}]
modelG=GridSearchCV(LogisticRegression(),tuned_parameters,scoring='f1')
modelG.fit(X_train,y_train)
print(modelG.best_estimator_)
print(modelG.score(X_test,y_test))


print("\n\n\n***********Random search CV**************")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
penalty = ['l1', 'l2']
C = uniform(loc=0, scale=4)
hyperparameters = dict(C=C, penalty=penalty)


modelR=RandomizedSearchCV(LogisticRegression(),param_distributions=hyperparameters,cv=3)
modelR.fit(X_train,y_train)
print(modelR.best_estimator_)
print(modelR.score(X_test,y_test))


print(y_test.values[500])
print(modelR.predict(X_test[500]))

