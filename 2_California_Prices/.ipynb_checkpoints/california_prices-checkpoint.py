#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:26:51 2020

@author: lorenzo
"""

import keras
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Flatten, Dense, SimpleRNN, Input, Concatenate

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data and split into training, validation and test set
housing = fetch_california_housing()
x_train, x_test, y_train, y_test = train_test_split(
                                housing.data, housing.target)
x_train, x_val, y_train, y_val = train_test_split(
                                x_train, y_train)

# check data dimensions
x_train.shape

# scale data
scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_val = scl.transform(x_val)
x_test = scl.transform(x_test)

# Baseline: linear regression (validation loss 0.49)
model = Sequential([
        Dense(1, input_shape = x_train.shape[1:])
        ])
model.compile(loss = 'mean_squared_error', optimizer = 'sgd')
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

# create keras model (validation loss approx 0.30)
model = Sequential([
        Dense(50, activation='relu', input_shape = x_train.shape[1:]),
        Dense(50, activation='relu'),
        Dense(1)
        ])

model.compile(loss = 'mean_squared_error', optimizer = 'sgd')
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

# test set error (approx 0.32)
mse_test = model.evaluate(x_test, y_test)

# predict on new instances
y_preds = model.predict(x_test[:5])

# learning curves
pd.DataFrame(history.history).plot(figsize = (6,4))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

#### FUNCTIONAL API ####
# Implement a wide&deep neural network
input_ = Input(shape = x_train.shape[1:])
hidden1 = Dense(50, activation='relu')(input_)
hidden2 = Dense(50, activation='relu')(hidden1)
concat = Concatenate()([input_, hidden2])
output = Dense(1)(concat)
model2 = keras.Model(inputs = [input_], outputs=[output])

model2.compile(loss = 'mean_squared_error', optimizer = 'sgd')
history2 = model2.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

# test set error (approx 0.32)
mse_test2 = model2.evaluate(x_test, y_test)

# predict on new instances
y_preds = model2.predict(x_test[:5])

# learning curves
pd.DataFrame(history2.history).plot(figsize = (6,4))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


