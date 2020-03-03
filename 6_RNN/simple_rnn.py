#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:25:33 2020

@author: lorenzo
"""

import numpy as np
from matplotlib import pyplot as plt
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Flatten, Dense, SimpleRNN


# create synthetic time-series
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time-offset1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time-offset2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)

x = generate_time_series(5, 100)

plt.plot(range(100), x[0,:,0])
plt.show()

n_steps = 50
series = generate_time_series(10000, n_steps+1)
x_train, y_train = series[:7000, :n_steps], series[:7000, -1]
x_val, y_val = series[7000:9000, :n_steps], series[7000:9000, -1]
x_test, y_test = series[9000:, :n_steps], series[9000:, -1]


# Baseline metrics
# 1) always predict last observed value
y_pred = x_val[:,-1]
np.mean(mean_squared_error(y_val, y_pred))

# 2) linear regression
model = Sequential([
        Flatten(input_shape = [n_steps, 1]),
        Dense(1)])
model.compile(loss = mean_squared_error, optimizer = 'adam')
print('\n \n *** Training linear model: *** \n')
model.fit(x_train, y_train, epochs = 20, 
          validation_data = (x_val, y_val))

# Simple RNN (1 layer, 1 unit)
model = Sequential([
        SimpleRNN(1, input_shape = [None, 1])])
model.compile(loss = mean_squared_error, optimizer = 'adam')
print('\n \n *** Training simpleRNN model: *** \n')
model.fit(x_train, y_train, epochs = 20, 
          validation_data = (x_val, y_val))

# Deep RNN (3 layers, 20-20-1 units)
model = Sequential([
        SimpleRNN(20, return_sequences = True, input_shape = [None, 1]),
        SimpleRNN(20),
        Dense(1)])
model.compile(loss = mean_squared_error, optimizer = 'adam')
print('\n \n *** Training DeepRNN model: *** \n')
model.fit(x_train, y_train, epochs = 20, 
          validation_data = (x_val, y_val))