#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:51:26 2020

@author: lorenzo
"""
import keras
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Flatten, Dense, SimpleRNN

# load data
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# images are represented as 28x28 arrays with integer pixel intensity from 0 to 255
x_train.shape
x_train.dtype

# get a validation set from the training one
# features should be scaled to be in (0,1)
x_val, x_train = x_train[:5000] / 255.0, x_train[5000:] /255.0
y_val, y_train = y_train[:5000], y_train[5000:]

# dictionary for converting numerical classes to labels
labels = ['T-shirt', 'Troursers', 'Pullover', 'Dress', 'Coat', 'Sandal',
              'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']
lab_dict = dict(zip(range(10), labels))

# plot some items
plt.figure(figsize=(5,5))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(labels[y_train[i]])
plt.show()


# baseline (54% accuracy)
model = Sequential()
model.add(Flatten(input_shape = [28,28]))
model.add(Dense(1))
model.add(Dense(10, activation= 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 20, validation_data = (x_val, y_val))

# fully connected 2 layers (88-89% accuracy on validation)
model = Sequential()
model.add(Flatten(input_shape = [28,28]))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation= 'softmax'))

# see number of parameters (approx 90k)
model.summary()

# compile & train the model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs = 30, validation_data = (x_val, y_val))

# retrieve and plot learning curves
pd.DataFrame(history.history).plot(figsize = (6,4))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# evaluate performances on test set
model.evaluate(x_test, y_test)

# make predictions on test or new data
y_proba = model.predict(x_test[:9])
y_preds = model.predict_classes(x_test[:9])

# plot actual vs predicted classes
plt.figure(figsize=(5,5))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.ylabel(labels[y_test[i]])
    plt.xlabel(labels[y_preds[i]])
plt.show()
