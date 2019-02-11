import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras import optimizers
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
num_training = 50000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

print(X_train)
num_test = 5000
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

X_train = X_train.astype('float16') / 255
X_test = X_test.astype('float16') / 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=X_train.shape[1:], padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(96, (3,3), padding='same', activation='relu'))
model.add(Conv2D(96, (3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=7, validation_data = (X_test, y_test))
model.save('./model.h5')

score = model.evaluate(X_test, y_test, batch_size=256)
print(score)