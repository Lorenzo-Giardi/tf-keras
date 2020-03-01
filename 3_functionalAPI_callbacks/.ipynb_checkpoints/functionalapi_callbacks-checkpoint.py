#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:20:09 2020

@author: lorenzo
"""

import os
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


#### FUNCTIONAL API ####
# Multiple inputs network
input_A = Input(shape = [5], name='wide_input')
input_B = Input(shape = [6], name='deep_input')
hidden1 = Dense(50, activation='relu')(input_B)
hidden2 = Dense(50, activation='relu')(hidden1)
concat = Concatenate()([input_A, hidden2])
output = Dense(1, name='output')(concat)
model = keras.Model(inputs = [input_A, input_B], outputs=[output])

model.compile(loss = 'mse', optimizer = 'sgd')
keras.utils.plot_model(model, "multi_inputs_model.png", show_shapes=True)
model.summary()

# split data into two groups, one for each input
# dataframe has dimensions: (11610, 8)
x_train_A, x_train_B = x_train[:,:5], x_train[:,2:]
x_val_A, x_val_B = x_val[:,:5], x_val[:,2:]
x_test_A, x_test_B = x_test[:,:5], x_test[:,2:]

history = model.fit([x_train_A, x_train_B], y_train, epochs=50, 
                    validation_data=([x_val_A, x_val_B], y_val))

# Multiple outputs network
input_A = Input(shape = [5], name='wide_input')
input_B = Input(shape = [6], name='deep_input')
hidden1 = Dense(50, activation='relu')(input_B)
hidden2 = Dense(50, activation='relu')(hidden1)
concat = Concatenate()([input_A, hidden2])
main_output = Dense(1, name='main_output')(concat)
aux_output = Dense(1, name='aux_output')(hidden2)
model2 = keras.Model(inputs = [input_A, input_B], outputs=[main_output, aux_output])

model2.compile(loss = ['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer = 'sgd')
keras.utils.plot_model(model2, "multi_outputs_model.png", show_shapes=True)
model2.summary()

history = model2.fit([x_train_A, x_train_B], [y_train, y_train], epochs=50, 
                    validation_data=([x_val_A, x_val_B], [y_val, y_val]))

# When evaluating the model, the loss is split into total, main and aux
# test set error (approx 0.32)
total_mse, main_mse, aux_mse = model2.evaluate(
                    [x_test_A, x_test_B], [y_test, y_test])

# predict on new instances
y_preds_main, y_preds_aux = model2.predict([x_test_A[:5], x_test_B[:5]])


#### CALLBACKS ####
model = Sequential([
        Dense(50, activation='relu', input_shape = x_train.shape[1:]),
        Dense(50, activation='relu'),
        Dense(1)
        ])

# 1) Checkpointing
checkpoint_cb = keras.callbacks.ModelCheckpoint('my_model.h5', save_best_only = True)

# 2) Early stopping
earlystop_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights= True)


# 3) tensorboard
log_dir = os.path.join(os.curdir, 'logs')
tensorboard_cb = keras.callbacks.TensorBoard(log_dir)


# train model with callbacks
model.compile(loss = 'mse', optimizer = 'sgd')
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val),
                    callbacks=[checkpoint_cb, earlystop_cb])
model = keras.models.load_model('my_model.h5')
 