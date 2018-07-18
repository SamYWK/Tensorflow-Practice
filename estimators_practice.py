# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:25:49 2018

@author: pig84
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

print(tf.__version__)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).batch(batch_size)
    

def main(argv):
    df = pd.read_csv('train.csv')
    X, y = df, df.pop('label')
    my_feature_columns = []
    for key in list(X):
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    # Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
    classifier = tf.estimator.DNNClassifier(
        feature_columns = my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units = [10, 10],
        # The model must choose between 10 classes.
        n_classes = 10, 
        model_dir = './saver')
    # Train the Model.
    classifier.train(
        input_fn=lambda:train_input_fn(X, y, 200),
        steps=10)

if __name__ == '__main__':
    tf.app.run()