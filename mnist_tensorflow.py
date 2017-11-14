# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:37:48 2017

@author: SamKao
"""

import tensorflow as tf
import pandas as pd
import numpy as np

def contruct_network(input_size, X, y):
    with tf.device('/gpu:0'):
        X_placeholder = tf.placeholder(tf.float32, [None, input_size])
        y_placeholder = tf.placeholder(tf.float32, [None, 10])
        
        W1 = tf.Variable(tf.random_normal([input_size, 50]))
        b1 = tf.Variable(tf.zeros([1, 50]) + 0.1)
        W2 = tf.Variable(tf.random_normal([50, 10]))
        b2 = tf.Variable(tf.zeros([1, 10]) + 0.1)
        
        a1 = tf.nn.relu(tf.matmul(X_placeholder, W1) + b1)
        a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)
        prediction = tf.nn.softmax(a2)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_placeholder, logits = prediction))
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        
        init = tf.global_variables_initializer()
        
        with tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)) as sess:
            sess.run(init)
            for iters in range(1000):
                sess.run(train_step, feed_dict = {X_placeholder: X, y_placeholder: y})
                if iters % 50 == 0:
                    print(sess.run(loss, feed_dict={X_placeholder: X, y_placeholder: y}))
    return None

def main():
    #42000*784
    df = pd.read_csv('train.csv')
    X = df.drop(['label'], axis = 1)
    y = df['label']
    enconde_table = np.zeros((42000, 10))
    enconde_table[np.arange(42000), y] = 1

    contruct_network(784, X.values, enconde_table)
    return None

main()