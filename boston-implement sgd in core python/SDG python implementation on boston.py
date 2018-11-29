# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:51:53 2018

@author: prabhudayala
"""

from sklearn.datasets import load_boston
boston=load_boston()
X=boston.data
print(X.shape)
y=boston.target
y=y.reshape((y.shape[0],1))
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Model coeficient by ")
print(model.coef_)
print(model.intercept_)

print(model.score(X_test,y_test))


def linear_regression(X, y, m_current=0, b_current=0, epochs=1000, learning_rate=0.0001):
     N = float(len(y))
     for i in range(epochs):
          y_current = (m_current * X) + b_current
          cost = sum([data**2 for data in (y-y_current)]) / N
          m_gradient = -(2/N) * sum(X * (y - y_current))
          b_gradient = -(2/N) * sum(y - y_current)
          m_current = m_current - (learning_rate * m_gradient)
          b_current = b_current - (learning_rate * b_gradient)
     return m_current, b_current, cost
 
m,b,cost=linear_regression(X_train,y_train,learning_rate=0.000001)
print(m)
print(b)