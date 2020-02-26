### Sequential API
model = Sequential([ \
        Dense(50, activation='relu', input_shape = x_train.shape[1:]), \
        Dense(50, activation='relu'), \
        Dense(1) \
        ]) \

Epoch 50/50 \
11610/11610 [==============================] - 0s 38us/step - loss: 0.2992 - val_loss: 0.3161

![](https://github.com/Lorenzo-Giardi/tf-keras/blob/master/2_California_Prices/fc_learning_curves.png)


### Functional API
input_ = Input(shape = x_train.shape[1:]) \
hidden1 = Dense(50, activation='relu')(input_) \
hidden2 = Dense(50, activation='relu')(hidden1) \
concat = Concatenate()([input_, hidden2]) \
output = Dense(1)(concat) \
model2 = keras.Model(inputs = [input_], outputs=[output]) \

Epoch 49/50 \
11610/11610 [==============================] - 0s 39us/step - loss: 0.2992 - val_loss: 0.2809 \
Epoch 50/50 \
11610/11610 [==============================] - 0s 38us/step - loss: 0.2992 - val_loss: 0.3161 

![](https://github.com/Lorenzo-Giardi/tf-keras/blob/master/2_California_Prices/wide%26deep_learning_curves.png)
