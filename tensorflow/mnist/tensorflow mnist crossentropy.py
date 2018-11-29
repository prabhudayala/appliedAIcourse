# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:53:53 2018

@author: prabhudayala
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
import numpy as np
print(mnist.train.images.shape[0])
print(mnist.train.images.shape[1])

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_=tf.placeholder(tf.float32,[None,10])
crossentropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
trainstep=tf.train.AdamOptimizer().minimize(crossentropy)
#tf.InteractiveSession.close()
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_x,batch_y=mnist.train.next_batch(128)
    sess.run(trainstep,feed_dict={x:batch_x,y_:batch_y})
    
correct_prediction=tf.equal(tf.argmax(y,1),tf.arg_max(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
