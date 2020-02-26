## Fashion MNIST
Dataset sample:
![](https://github.com/Lorenzo-Giardi/tf-keras/blob/master/1_Fashion_MNIST/df_sample.png)

### ANN with two FC hidden layers
model = Sequential() \
model.add(Flatten(input_shape = [28,28])) \
model.add(Dense(100, activation = 'relu')) \
model.add(Dense(100, activation = 'relu')) \
model.add(Dense(10, activation= 'softmax'))


Train on 55000 samples, validate on 5000 samples \
Epoch 1/30 \
55000/55000 [==============================] - 3s 59us/step - loss: 0.7803 - accuracy: 0.7439 - val_loss: 0.5355 - val_accuracy: 0.8160 \
... \
Epoch 29/30 \
55000/55000 [==============================] - 3s 57us/step - loss: 0.2568 - accuracy: 0.9068 - val_loss: 0.3125 - val_accuracy: 0.8894 \
Epoch 30/30 \
55000/55000 [==============================] - 3s 56us/step - loss: 0.2532 - accuracy: 0.9083 - val_loss: 0.3056 - val_accuracy: 0.8912

![](https://github.com/Lorenzo-Giardi/tf-keras/blob/master/1_Fashion_MNIST/learning_curves.png)
