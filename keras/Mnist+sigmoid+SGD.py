# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 22:00:58 2018

@author: prabhudayala
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 20:06:50 2018

@author: prabhudayala
"""
from  keras.utils import np_utils
from keras.datasets import mnist
import seaborn as sns
from keras.initializers import RandomNormal

(X_train,y_train) , (X_test,y_test) = mnist.load_data()
print(X_train.shape[0])
print(X_train.shape[1])
print(X_train.shape[2])

X_train=X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
X_test=X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))

print(X_train.shape[0])
print(X_train.shape[1])

#print(X_train[0])

X_train=X_train/255
X_test=X_test/255   

y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)
print(y_train[0])

from keras import Sequential
from keras.layers.core import Dense, Activation

output_dim=10
input_dim=X_train.shape[1]
batch_size=128
epoch_num=2


model=Sequential()
#model.add(Dense(512,input_dim=input_dim,activation='sigmoid'))
model.add(Dense(512,input_shape=(784,),activation='sigmoid'))
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(output_dim,activation='softmax'))
print(model.summary())


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=batch_size,epochs=epoch_num,validation_data=(X_test,y_test))
#print(model.get_weights())
#print(model.layers[0].get_weights()[1])