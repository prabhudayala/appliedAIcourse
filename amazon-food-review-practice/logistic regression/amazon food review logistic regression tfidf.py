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

filtered_data=filtered_data.sample(n=5000,random_state=42)
filtered_data=filtered_data.sort_values('Time')


tf_idf_vect= TfidfVectorizer()
final_tf_idf=tf_idf_vect.fit_transform(filtered_data['CleanedText'].values)


X_1, X_test, y_1, y_test=train_test_split(final_tf_idf,filtered_data['Score'],test_size=0.3,random_state=42)
X_train, X_cv, y_train, y_cv=train_test_split(X_1,y_1,test_size=0.3,random_state=42)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
print(model.coef_)
print(model.intercept_)
