# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 19:20:54 2018

@author: prabhudayala
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

import matplotlib.pyplot as plt
import numpy as np
import time 
def dynamic_plot(x,y,y_1,ax,ticks,title="Default",colors=['b']):
    ax.plot(x,y,'b',label='Train loss')
    ax.plot(x,y_1,'r',label='Test loss')
    if len(x)==1:
        plt.legend()
        plt.title(title)
    plt.yticks(ticks)
    #fig.canvas.draw()

n_input=784
n_class=10
n_hidden_1=512
n_hidden_2=128


x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

keep_prob=tf.placeholder(tf.float32)
keep_prob_input=tf.placeholder(tf.float32)

weights_sgd={
        'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=0.039)),
        'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=0.055)),
        'out':tf.Variable(tf.random_normal([n_hidden_2,n_class],stddev=0.120))
        
        }


weights_relu={
        'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=0.062)),
        'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=0.055)),
        'out':tf.Variable(tf.random_normal([n_hidden_2,n_class],stddev=0.120))
        
        }

biases={
        'b1':tf.Variable(tf.random_normal([n_hidden_1])),
        'b2':tf.Variable(tf.random_normal([n_hidden_2])),
        'out':tf.Variable(tf.random_normal([n_class]))
        
        }

training_epochs = 15
learning_rate = 0.001
batch_size = 100
display_step = 1



def multilayer_perceptron(x,weights,biases):
    #print('x: ', x.get_shape(), 'W[h1]: ' ,weights['h1'].get_shape() ,'b[b1]: ',biases['b1'].get_shape())
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1=tf.nn.sigmoid(layer_1)
    
    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2=tf.nn.sigmoid(layer_2)
    
    out_layer=tf.add(tf.matmul(layer_2,weights['out']),biases['out'])
    out_layer=tf.nn.sigmoid(out_layer)
    
    
    return out_layer


#y_sgd=multilayer_perceptron(x,weights_sgd,biases)
#cost_sgd=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_sgd,labels=y_))
#
#optimizer_adam=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_sgd)
#optimizer_sgdc=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_sgd)
#
#
#with tf.Session() as sess:
#    tf.global_variables_initializer().run()
#    #fig,ax
#    xs,ytrs,ytes=[],[],[]
#    for epoch in range (training_epoch):
#        train_average_cost=0.
#        test_average_cost=0.
#        total_batch=int(mnist.train.num_examples/batch_size)
#        
#        for i in range(total_batch):
#            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
#            _,c,w=sess.run([optimizer_adam,cost_sgd,weights_sgd],feed_dict={x:batch_xs,y_:batch_ys})
#            train_average_cost+=c/total_batch
#            c=sess.run(cost_sgd,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
#            test_average_cost+=c/total_batch
#        xs.append(epoch)
#    correct_prediction=tf.equal(tf.argmax(y_sgd,1),tf.arg_max(y_,1))
#    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#    print("Accuracy:", accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

# Since we are using sigmoid activations in hiden layers we will be using weights that are initalized as weights_sgd
y_sgd = multilayer_perceptron(x, weights_sgd, biases)
cost_sgd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_sgd, labels = y_))
optimizer_adam = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_sgd)
optimizer_sgdc = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_sgd)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    xs, ytrs, ytes = [], [], []
    for epoch in range(training_epochs):
        print("epoch is: %s"%epoch)
        train_avg_cost = 0.
        test_avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c, w = sess.run([optimizer_adam, cost_sgd,weights_sgd], feed_dict={x: batch_xs, y_: batch_ys})
            train_avg_cost += c / total_batch
            c = sess.run(cost_sgd, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            test_avg_cost += c / total_batch

        xs.append(epoch)
        ytrs.append(train_avg_cost)
        ytes.append(test_avg_cost)

    correct_prediction = tf.equal(tf.argmax(y_sgd,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))